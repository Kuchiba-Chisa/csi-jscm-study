# -*- coding: utf-8 -*-
"""E5（修正版 Q2）：冻结同一编码器与星座，只重训解码器。

三个接收接口：硬判决 / 带噪观测（软信息）/ 带噪观测 + 噪声水平。
发送端完全相同，所以三者的差别只来自"接收端丢不丢软信息"。

用法：
  .venv/Scripts/python.exe rx_ablation.py --epochs 40 --seeds 0,1,2
输出：results_rx.json（val/test 逐点曲线，含平方 SGCS）
"""
import argparse
import json
import os

import numpy as np
import torch

from model_jscm import (HARD_RX, Decoder, LearnedConstellation, Link, calibrate,
                        cos_flat, nmse_db, sgcs)
from run_experiments import load_split, SNR_GRID
from eval_v2 import sgcs_sq


def build(rx, base_ckpt, ns=64, m=16):
    """新建链路，只从 base_ckpt 拷编码器与星座；解码器保持随机初始化。"""
    net = Link(rx, ns, m, power_ref=True)
    sd = torch.load(base_ckpt, map_location="cpu")
    own = net.state_dict()
    keep = {k: v for k, v in sd.items()
            if (k.startswith("enc.") or k.startswith("const.")) and k in own}
    assert any(k.startswith("enc.") for k in keep), f"{base_ckpt} 里没有编码器权重"
    own.update(keep)
    net.load_state_dict(own)
    for p in net.enc.parameters():
        p.requires_grad_(False)
    if isinstance(net.const, LearnedConstellation):
        for p in net.const.parameters():
            p.requires_grad_(False)
    return net


@torch.no_grad()
def tx_of(net, x):
    z = net.enc(x)
    z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
    tx, idx = net.modulate(z)
    return tx, idx


def train_dec(net, xtr, p_cal, epochs, seed, log=print, bs=128):
    torch.manual_seed(seed)
    tx, idx = tx_of(net, xtr)
    pts = net.const.get_points().detach()
    opt = torch.optim.Adam([p for p in net.dec.parameters() if p.requires_grad], lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xtr)
    for ep in range(epochs):
        # 关键：解码器必须处于 train 模式，否则 BatchNorm 只用初始化统计量、
        # running stats 永远不更新（v2 的 E5 就是踩了这个坑）
        net.train()
        net.enc.eval()
        if hasattr(net.const, "eval"):
            net.const.eval()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            snr = torch.empty(1).uniform_(0, 20).item()
            rel = 10 ** (-snr / 20.0)
            sigma = rel * p_cal ** 0.5
            y = tx[j] + sigma * torch.randn_like(tx[j])
            if net.scheme in HARD_RX:
                d = (y.unsqueeze(-1) - pts.view(1, 1, -1)).abs()
                y = pts[d.argmin(-1)]
            if net.scheme == "learned_soft_snr":
                s_t = torch.full((y.shape[0], 1), float(rel), dtype=y.dtype)
                y = torch.cat([y, s_t], dim=1)
            pred = net.dec(y)
            loss = (pred - xtr[j]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sched.step()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"    ep{ep:02d} train_mse={tot / n:.4f}")
    return net


@torch.no_grad()
def curve(net, x, snrs=SNR_GRID, noise_seed=20260101):
    net.eval()
    out = {"nmse": [], "sgcs2": [], "sgcs1": [], "cos_flat": []}
    for s in snrs:
        torch.manual_seed(noise_seed + int(round(s)))
        pred = net(x, float(s))
        out["nmse"].append(round(float(nmse_db(pred, x)), 4))
        out["sgcs2"].append(round(float(sgcs_sq(pred, x)), 4))
        out["sgcs1"].append(round(float(sgcs(pred, x)), 4))
        out["cos_flat"].append(round(float(cos_flat(pred, x)), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="learned_soft")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rx", default="learned_hard,learned_soft,learned_soft_snr")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results_rx.json")
    args = ap.parse_args()

    xtr, xva, xte = load_split("train"), load_split("val"), load_split("test")
    ck = f"ckpt_v2/{args.base}_s{args.base_seed}.pt"
    assert os.path.exists(ck), ck
    out = {"snr": SNR_GRID.tolist(), "base_ckpt": ck, "epochs": args.epochs,
           "note": "E5：冻结编码器+星座，只重训解码器", "val": {}, "test": {}}
    if os.path.exists(args.out):
        out = json.load(open(args.out, encoding="utf-8"))

    for rx in [s for s in args.rx.split(",") if s]:
        for seed in [int(s) for s in args.seeds.split(",") if s]:
            if str(seed) in out["val"].get(rx, {}):
                print(f"skip {rx} s{seed}")
                continue
            print(f"[{rx} s{seed}] 冻结编码器，只训解码器 {args.epochs} epoch")
            torch.manual_seed(seed)
            net = build(rx, ck)
            p_cal = calibrate(net, xtr)
            net.p_cal = p_cal
            net = train_dec(net, xtr, p_cal, args.epochs, seed)
            net.eval()
            out["val"].setdefault(rx, {})[str(seed)] = curve(net, xva)
            out["test"].setdefault(rx, {})[str(seed)] = curve(net, xte)
            m = out["test"][rx][str(seed)]["nmse"][5]
            print(f"  -> p_cal={p_cal:.3f}  test@10dB {m:+.2f} dB")
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
