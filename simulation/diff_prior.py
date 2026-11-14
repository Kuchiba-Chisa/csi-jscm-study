# -*- coding: utf-8 -*-
"""E6 前置：训练 CSI 先验（DDPM，2×32×32），并做先验检查。

用法：
  python diff_prior.py --train --epochs 200 --device cuda
  python diff_prior.py --check  --device cuda
产物：ckpt_diff/prior.pt、prior_check.json
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_experiments import load_split

T_STEPS = 500


def cosine_alpha_bar(t_steps=T_STEPS, s=0.008):
    """cos² 调度：ab_t = cos²((t/T + s)/(1+s) · π/2)。注意分母是 T，不是 1+s。"""
    n = t_steps + 1
    tt = torch.arange(n, dtype=torch.float32)
    f = torch.cos(((tt / t_steps + s) / (1 + s)) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(1e-6, 1 - 1e-6)


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class Block(nn.Module):
    def __init__(self, cin, cout, cemb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(cemb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.e(F.silu(emb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """2×32×32 的小 UNet：64 → 128 → 256，两个下采样。"""

    def __init__(self, ch=64, cemb=128):
        super().__init__()
        self.temb = TimeEmbed(cemb)
        self.inp = nn.Conv2d(2, ch, 3, padding=1)
        self.b1 = Block(ch, ch, cemb)
        self.d1 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.b2 = Block(ch * 2, ch * 2, cemb)
        self.d2 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)
        self.b3 = Block(ch * 4, ch * 4, cemb)
        self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)
        self.b4 = Block(ch * 4, ch * 2, cemb)
        self.u1 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)
        self.b5 = Block(ch * 2, ch, cemb)
        self.out = nn.Conv2d(ch, 2, 3, padding=1)

    def forward(self, x, t):
        emb = self.temb(t)
        h1 = self.b1(self.inp(x), emb)
        h2 = self.b2(self.d1(h1), emb)
        h3 = self.b3(self.d2(h2), emb)
        h = self.b4(torch.cat([self.u2(h3), h2], 1), emb)
        h = self.b5(torch.cat([self.u1(h), h1], 1), emb)
        return self.out(h)


class Diffusion:
    def __init__(self, t_steps=T_STEPS, device="cpu"):
        self.T = t_steps
        ab = cosine_alpha_bar(t_steps).to(device)
        self.ab = ab
        self.sq_ab = ab.sqrt()
        self.sq_1mab = (1 - ab).sqrt()

    def q_sample(self, x0, t, noise):
        return self.sq_ab[t][:, None, None, None] * x0 + self.sq_1mab[t][:, None, None, None] * noise

    def posterior(self, x_t, x0, t, prev_t):
        """标准 DDPM 后验均值与方差：q(x_{t-1} | x_t, x_0)。

        mean = sqrt(ab_{t-1})*beta_t/(1-ab_t) * x0 + sqrt(alpha_t)*(1-ab_{t-1})/(1-ab_t) * x_t
        var  = beta_t*(1-ab_{t-1})/(1-ab_t)
        """
        ab_t = self.ab[t][:, None, None, None]
        ab_p = self.ab[prev_t][:, None, None, None]
        beta_t = (1 - ab_t / ab_p).clamp(1e-6, 0.999)      # 1 - alpha_t
        coef_x0 = ab_p.sqrt() * beta_t / (1 - ab_t).clamp_min(1e-8)
        coef_xt = (ab_t / ab_p).sqrt() * (1 - ab_p) / (1 - ab_t).clamp_min(1e-8)
        var = beta_t * (1 - ab_p) / (1 - ab_t).clamp_min(1e-8)
        return coef_x0 * x0 + coef_xt * x_t, var


@torch.no_grad()
def sample_prior(model, diff, n, device, steps=None):
    """无条件采样（先验检查用）。标准 DDPM 祖先采样。"""
    T = steps or diff.T
    x = torch.randn(n, 2, 32, 32, device=device)
    model.eval()
    for i in reversed(range(T)):
        t = torch.full((n,), i * diff.T // T, dtype=torch.long, device=device)
        eps = model(x, t)
        x0 = ((x - diff.sq_1mab[t][:, None, None, None] * eps) /
              diff.sq_ab[t][:, None, None, None]).clamp(-6, 6)
        if i > 0:
            tp = torch.full((n,), (i - 1) * diff.T // T, dtype=torch.long, device=device)
            mean, var = diff.posterior(x, x0, t, tp)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0
    return x


def stats_of(x):
    """功率、稀疏度、相关系数，用于与真实数据比较。"""
    p = x.pow(2).mean(dim=(1, 2, 3))
    amp = x.pow(2).sum(1)
    sparsity = (amp < 0.1 * amp.mean(dim=(1, 2), keepdim=True)).float().mean().item()
    xc = x - x.mean(dim=(2, 3), keepdim=True)
    corr_ac = torch.corrcoef(torch.stack([
        xc[:, 0].mean(1).flatten(), xc[:, 0].mean(2).flatten()]))[0, 1].item()
    return {"power": float(p.mean()), "power_std": float(p.std()),
            "sparsity_frac": sparsity, "corr_ant": corr_ac,
            "absmax": float(x.abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", default="ckpt_diff/prior.pt")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    diff = Diffusion(device=dev)

    if args.train:
        xtr = load_split("train").to(dev)
        model = TinyUNet().to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"prior params {n_par/1e6:.2f} M   device {dev}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        n = len(xtr)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            tot, cnt = 0.0, 0
            for i in range(0, n, args.bs):
                j = perm[i:i + args.bs]
                x0 = xtr[j]
                t = torch.randint(0, diff.T, (len(j),), device=dev)
                noise = torch.randn_like(x0)
                x_t = diff.q_sample(x0, t, noise)
                pred = model(x_t, t)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(j)
                cnt += len(j)
            sched.step()
            print(f"  ep{ep:03d} eps_mse={tot/cnt:.4f}", flush=True)
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save({"model": model.state_dict(), "t_steps": diff.T, "epochs": ep + 1},
                       args.ckpt)
        print("saved", args.ckpt)

    if args.check:
        ck = torch.load(args.ckpt, map_location="cpu")
        model = TinyUNet().to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        xva = load_split("val").to(dev)
        rep = {"real": stats_of(xva[:200])}
        # 1) 各噪声水平的去噪误差
        errs = {}
        with torch.no_grad():
            for snr_db in (0, 10, 20):
                sigma = 10 ** (-snr_db / 20.0)
                for t_frac in (0.2, 0.5):
                    t = torch.full((200,), int(t_frac * diff.T), dtype=torch.long, device=dev)
                    noise = torch.randn_like(xva[:200])
                    x_t = diff.q_sample(xva[:200], t, noise)
                    eps_hat = model(x_t, t)
                    x0_hat = (x_t - diff.sq_1mab[t][:, None, None, None] * eps_hat) / \
                             diff.sq_ab[t][:, None, None, None]
                    nmse = 10 * torch.log10(((x0_hat - xva[:200]).pow(2).sum((1, 2, 3)) /
                                             xva[:200].pow(2).sum((1, 2, 3)))).mean()
                    errs[f"t{t_frac}"] = float(nmse)
        rep["denoise_nmse"] = errs
        # 2) 无条件生成样本的统计量
        gen = sample_prior(model, diff, 100, dev)
        rep["gen"] = stats_of(gen)
        with open("prior_check.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
