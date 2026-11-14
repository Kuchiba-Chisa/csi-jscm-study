# -*- coding: utf-8 -*-
"""E6：扩散先验 + DPS 后验采样接收端，与学习型解码器对比。

前向算子与生产链路完全一致（analog 编码器 + 逐样本功率归一化）：
    z = enc(H);  F(H) = z / sqrt(mean_i z_i^2 + eps);  y = F(H) + n,  n ~ N(0, σ²)
σ 由冻结的功率标定确定（analog 的 P_tx 恒为 1，故 σ = 10^(-SNR/20)）。

用法：
  python dps_rx.py --n 100 --k 4 --steps 150 --device cuda
产物：dps_results.json
"""
import argparse
import json
import os

import torch

from diff_prior import Diffusion, TinyUNet
from model_jscm import Link, nmse_db
from run_experiments import load_split, SNR_GRID
from eval_v2 import sgcs_sq

ZETA = 0.5          # 预先定下：guidance 步长 = ZETA / ||y - F(x0_hat)||


class Forward:
    """F(H) = normalize(enc(H))，对输入可导（DPS 要对输入求梯度）。"""

    def __init__(self, ckpt="ckpt_v2/analog_s0.pt", device="cpu"):
        net = Link("analog", 64, 16)
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.to(device).eval()
        self.enc = net.enc
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.dec = net.dec                     # analog 自带解码器（基线）
        for p in self.dec.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        z = self.enc(x)
        return z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)


@torch.no_grad()
def sample_prior(model, diff, n, device, steps=150, seed=7):
    """无条件 ancestral 采样（150 步），用于"没有任何观测"的基线。"""
    torch.manual_seed(seed)
    x = torch.randn(n, 2, 32, 32, device=device)
    for i in reversed(range(steps)):
        tval = int(i * diff.T / steps)
        t = torch.full((n,), tval, dtype=torch.long, device=device)
        eps = model(x, t)
        sab, s1m = diff.sq_ab[t][:, None, None, None], diff.sq_1mab[t][:, None, None, None]
        x0 = ((x - s1m * eps) / sab).clamp(-6, 6)
        if i > 0:
            tp = torch.full((n,), int((i - 1) * diff.T / steps), dtype=torch.long, device=device)
            mean, var = diff.posterior(x, x0, t, tp)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0
    return x


def dps(model, diff, fwd, y, n_chains, steps, device, sigma_used=None, zeta=ZETA):
    """对同一批观测跑 n_chains 条链，返回 (n_chains, n, 2,32,32)。

    引导项：g = ∂‖y − F(x̂0)‖²/∂x_t，按 (ζ · s / ‖r‖) 缩放，
    其中 s = clamp(1/(2σ̂²), 0.5, 4) 是后验精度权重（σ̂ 由接收端自己估计，
    失配实验里故意估错）。这里的 s 在跑结果之前就写定，不按结果调。
    """
    n = y.shape[0]
    sig = float(sigma_used) if sigma_used is not None else 1.0
    prec = min(max(1.0 / (2.0 * sig * sig), 0.5), 4.0)     # 后验精度权重，带上下限
    x = torch.randn(n * n_chains, 2, 32, 32, device=device)
    yb = y.repeat(n_chains, 1)                              # y: (n, Ns) → (k*n, Ns)
    for i in reversed(range(steps)):
        tval = int(i * diff.T / steps)
        t = torch.full((x.shape[0],), tval, dtype=torch.long, device=device)
        ab, sab, s1m = (diff.ab[t][:, None, None, None], diff.sq_ab[t][:, None, None, None],
                        diff.sq_1mab[t][:, None, None, None])
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            eps = model(xg, t)
            x0 = ((xg - s1m * eps) / sab).clamp(-6, 6)
            r = yb - fwd(x0)
            d = r.pow(2).flatten(1).sum(1)                  # 每个样本一个标量
            grad = torch.autograd.grad(d.sum(), xg)[0]
        eps = eps.detach()
        rn = d.sqrt().detach().view(-1, 1, 1, 1) + 1e-8
        if i > 0:
            tp = torch.full((x.shape[0],), int((i - 1) * diff.T / steps), dtype=torch.long, device=device)
            mean, var = diff.posterior(x, x0.detach(), t, tp)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0.detach()
        # 观测一致性引导：步长按残差归一化，再乘后验精度权重
        x = x - (zeta * prec / rn) * grad
    return x.view(n_chains, n, 2, 32, 32)


def metrics(pred, x):
    return {"nmse": round(float(nmse_db(pred, x)), 4),
            "sgcs2": round(float(sgcs_sq(pred, x)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=4, help="每个观测的采样链数（取平均）")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--zeta", type=float, default=ZETA)
    ap.add_argument("--snrs", default="0,10,20")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prior", default="ckpt_diff/prior.pt")
    ap.add_argument("--out", default="dps_results.json")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    diff = Diffusion(device=dev)
    ck = torch.load(args.prior, map_location="cpu")
    model = TinyUNet().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fwd = Forward(device=dev)

    xte = load_split(args.split)[:args.n].to(dev)
    snrs = [float(s) for s in args.snrs.split(",")]

    rep = {"n": args.n, "k": args.k, "steps": args.steps, "zeta": args.zeta, "split": args.split,
           "note": "E6：DPS 后验采样 vs 学习型解码器；同一 analog 编码器与同一观测；引导含后验精度权重", "results": {}}
    # 基线：无观测（先验样本）
    torch.manual_seed(7)
    prior_samp = sample_prior(model, diff, len(xte), dev, steps=args.steps)
    rep["results"]["prior_only"] = {
        "nmse": round(float(nmse_db(prior_samp, xte)), 4),
        "sgcs2": round(float(sgcs_sq(prior_samp, xte)), 4)}
    # 基线：analog 共享解码器（同一编码器、同一观测）
    with torch.no_grad():
        for s in snrs:
            torch.manual_seed(20260101 + int(s))
            z = fwd(xte)
            y = z + 10 ** (-s / 20.0) * torch.randn_like(z)
            rep["results"].setdefault("analog_shared", {})[str(int(s))] = \
                metrics(fwd.dec(y), xte)

    for s in snrs:
        rel = 10 ** (-s / 20.0)
        torch.manual_seed(20260101 + int(s))
        z = fwd(xte)
        y = z + rel * torch.randn_like(z)
        for k in (1, args.k):
            if k > args.k:
                continue
            print(f"[DPS] SNR={s:.0f} k={k} steps={args.steps} n={len(xte)}", flush=True)
            chains = dps(model, diff, fwd, y, k, args.steps, dev, zeta=args.zeta)
            est = chains.mean(0)
            rep["results"].setdefault(f"dps_k{k}", {})[str(int(s))] = {
                "nmse": round(float(nmse_db(est, xte)), 4),
                "sgcs2": round(float(sgcs_sq(est, xte)), 4),
                "nmse_per_chain": [round(float(nmse_db(chains[i], xte)), 4) for i in range(k)]}
            print("   ", rep["results"][f"dps_k{k}"][str(int(s))]["nmse"], flush=True)
        # SNR 失配：让接收端以为 SNR 偏 +3 dB（σ 估小）
        if s in (0.0, 10.0):
            wrong = 10 ** (-(s + 3) / 20.0)
            print(f"[DPS-mismatch] true SNR={s:.0f}, receiver assumes +3 dB", flush=True)
            est = dps(model, diff, fwd, y, args.k, args.steps, dev, sigma_used=wrong, zeta=args.zeta).mean(0)
            rep["results"].setdefault("dps_mismatch", {})[str(int(s))] = {
                "nmse": round(float(nmse_db(est, xte)), 4),
                "sgcs2": round(float(sgcs_sq(est, xte)), 4)}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print("saved", args.out)
    print(json.dumps(rep["results"], ensure_ascii=False, indent=1)[:1500])


if __name__ == "__main__":
    main()
