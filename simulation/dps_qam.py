# -*- coding: utf-8 -*-
"""E6b：把 DPS 的前向算子换成「编码器 + 16-PAM 硬映射」，即真正的有限星座 JSCM 链路。

与 dps_rx.py（analog 连续符号）的唯一区别是 F：
    z  = enc(H) / sqrt(mean_i z_i^2 + eps)        # UE 侧逐样本功率归一（生产链路同款）
    tx = STE_hard(z)                              # 16-PAM 硬映射：前向取最近电平，反向恒等
    y  = tx + n,   n ~ N(0, σ² I),  σ = 10^(-SNR/20) * sqrt(p_cal)
p_cal 是这条链路上硬符号的实际平均功率（learned_soft 的 calibration 值），冻结。

用法（本地 CPU 可跑）：
  python dps_qam.py --n 32 --ks 1,4,8 --steps 150 --snrs 10 --out dps_qam.json
"""
import argparse
import json
import time

import torch

from diff_prior import Diffusion, TinyUNet
from model_jscm import Link, calibrate, nmse_db
from run_experiments import load_split
from eval_v2 import sgcs_sq

ZETA = 0.5          # 引导步长 = ZETA / ‖r‖，预先定下，不按结果调


def load_net(ckpt, device):
    net = Link("learned_soft", 64, 16)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class FwdQAM:
    """F(H) = STE_hard(normalize(enc(H)))，对输入可导。"""

    def __init__(self, net):
        self.net = net
        self.enc = net.enc

    def __call__(self, x):
        z = self.enc(x)
        z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
        tx, _ = self.net.modulate(z)      # 前向 = 硬符号，反向 = 恒等（STE）
        return tx


@torch.no_grad()
def sample_prior(model, diff, n, device, steps=150, seed=7):
    torch.manual_seed(seed)
    x = torch.randn(n, 2, 32, 32, device=device)
    for i in reversed(range(steps)):
        tval = int(i * diff.T / steps)
        t = torch.full((n,), tval, dtype=torch.long, device=device)
        eps = model(x, t)
        s1m = diff.sq_1mab[t][:, None, None, None]
        sab = diff.sq_ab[t][:, None, None, None]
        x0 = ((x - s1m * eps) / sab).clamp(-6, 6)
        if i > 0:
            tp = torch.full((n,), int((i - 1) * diff.T / steps), dtype=torch.long, device=device)
            mean, var = diff.posterior(x, x0, t, tp)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0
    return x


def dps(model, diff, fwd, y, n_chains, steps, device, sigma, zeta=ZETA):
    n = y.shape[0]
    prec = min(max(1.0 / (2.0 * sigma * sigma), 0.5), 4.0)
    x = torch.randn(n * n_chains, 2, 32, 32, device=device)
    yb = y.repeat(n_chains, 1)
    for i in reversed(range(steps)):
        tval = int(i * diff.T / steps)
        t = torch.full((x.shape[0],), tval, dtype=torch.long, device=device)
        sab, s1m = diff.sq_ab[t][:, None, None, None], diff.sq_1mab[t][:, None, None, None]
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            eps = model(xg, t)
            x0 = ((xg - s1m * eps) / sab).clamp(-6, 6)
            r = yb - fwd(x0)
            d = r.pow(2).flatten(1).sum(1)
            grad = torch.autograd.grad(d.sum(), xg)[0]
        rn = d.sqrt().detach().view(-1, 1, 1, 1) + 1e-8
        if i > 0:
            tp = torch.full((x.shape[0],), int((i - 1) * diff.T / steps), dtype=torch.long, device=device)
            mean, var = diff.posterior(x, x0.detach(), t, tp)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0.detach()
        x = x - (zeta * prec / rn) * grad
    return x.view(n_chains, n, 2, 32, 32)


def metrics(pred, x):
    return {"nmse": round(float(nmse_db(pred, x)), 4),
            "sgcs2": round(float(sgcs_sq(pred, x)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ks", default="1,4,8")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--snrs", default="10")
    ap.add_argument("--zeta", type=float, default=ZETA)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default="ckpt_v2/learned_soft_s0.pt")
    ap.add_argument("--prior", default="ckpt_diff/prior.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="dps_qam.json")
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

    net = load_net(args.ckpt, dev)
    fwd = FwdQAM(net)
    xte = load_split(args.split)[:args.n].to(dev)
    # p_tx 必须在训练集上标定（与生产链路同一口径）：Link 里 p_cal 默认为 None
    xtr = load_split("train")[:2000].to(dev)
    with torch.no_grad():
        calibrate(net, xtr)
    p_tx = float(net.p_cal)

    rep = {"n": args.n, "steps": args.steps, "zeta": args.zeta, "split": args.split,
           "p_tx": p_tx, "ckpt": args.ckpt,
           "note": "E6b：有限星座 JSCM 前向（编码器 + 16-PAM STE 硬映射）+ DPS 接收；对照同一观测的解码器",
           "results": {}}

    with torch.no_grad():
        z_clean = fwd(xte)                                   # 无噪硬符号
        rep["results"]["tx_power"] = round(float(z_clean.pow(2).mean()), 4)
        pts = net.const.get_points()
        obs = {}
        for s in [float(v) for v in args.snrs.split(",")]:
            sigma = 10 ** (-s / 20.0) * (p_tx ** 0.5)
            g = torch.Generator(device="cpu").manual_seed(20260101 + int(s))
            y = z_clean + sigma * torch.randn(z_clean.shape, generator=g).to(dev)
            obs[int(s)] = (y, sigma)
            # 基线 1：soft 解码器（生产链路的 JSCM 接收端）
            rep["results"].setdefault("decoder_soft", {})[str(int(s))] = metrics(net.dec(y), xte)
            # 基线 2：硬判决（在接收端对 y 做最近电平判决）+ 同一解码器
            idx_y = (y.unsqueeze(-1) - pts).abs().argmin(-1)
            rep["results"].setdefault("decoder_hard", {})[str(int(s))] = metrics(
                net.dec(pts[idx_y]), xte)
        torch.manual_seed(7)
        prior_samp = sample_prior(model, diff, len(xte), dev, steps=args.steps)
        rep["results"]["prior_only"] = metrics(prior_samp, xte)

    for s in [float(v) for v in args.snrs.split(",")]:
        y, sigma = obs[int(s)]
        for k in [int(v) for v in args.ks.split(",")]:
            t0 = time.time()
            chains = dps(model, diff, fwd, y, k, args.steps, dev, sigma, args.zeta)
            r = metrics(chains.mean(0), xte)
            r.update({"k": k, "sec": round(time.time() - t0, 1)})
            rep["results"].setdefault("dps", {}).setdefault(str(int(s)), {})[str(k)] = r
            print(f"  SNR {s:>4.0f} dB  k={k:<2d}  NMSE {r['nmse']:+.3f}  SGCS2 {r['sgcs2']:.4f}  {r['sec']}s",
                  flush=True)
            torch.save({"chains": chains.detach().cpu()}, f"dps_qam_k{k}_snr{int(s)}.pt")


    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
