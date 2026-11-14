# -*- coding: utf-8 -*-
"""BN 统计量失配诊断（只评估，不训练）。

假设：训练时每个 batch 只抽一个 SNR，解码器 BN 的 running stats 是跨 SNR 的
混合统计量；测试时在固定 SNR 下用这套混合统计量，低 SNR 处偏离最远。

做法：固定权重，把解码器 BN 的 running stats 重置后，用**该 SNR** 的训练数据
重新累积，再评估同一 SNR。对比"重标定前 / 后"。

    python bn_recal.py                      # 5 个方案 × 3 种子 × 11 个 SNR
    python bn_recal.py --schemes learned_soft learned_soft_snr --seeds 0
"""
import argparse
import json
import re

import numpy as np
import torch
from torch import nn

from model_jscm import Link, nmse_db, sgcs, calibrate
from run_experiments import load_split

SCHEMES = ["analog", "fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr"]
SNR_GRID = list(range(0, 21, 2))


def dec_bns(net):
    """解码器里的所有 BatchNorm2d。"""
    return [m for n, m in net.named_modules() if n.startswith("dec.") and isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def recalibrate(net, xtr, snr, bs=128, n_batch=40):
    """用该 SNR 的训练数据重新累积解码器 BN 统计量（不改变任何权重）。"""
    bns = dec_bns(net)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # 累积平均
        m.train()
    net.eval()                     # 其余模块保持 eval，只让 BN 累积
    for m in bns:
        m.train()
    net.p_cal = net.p_cal          # 保持冻结标定
    for i in range(0, min(len(xtr), bs * n_batch), bs):
        net(xtr[i:i + bs], float(snr))
    for m in bns:
        m.eval()
    return net


@torch.no_grad()
def evaluate(net, x, snr, noise_seed=1234):
    torch.manual_seed(noise_seed)
    pred = net(x, float(snr))
    return float(nmse_db(pred, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--snrs", nargs="+", type=int, default=SNR_GRID)
    ap.add_argument("--ckpt", default="ckpt_v2")
    ap.add_argument("--out", default="bn_recal.json")
    a = ap.parse_args()

    xtr, xva, xte = load_split("train"), load_split("val"), load_split("test")
    res = {}
    for scheme in a.schemes:
        base = re.sub(r"_f\d+$", "", scheme).removesuffix("_ps")   # learned_soft_f0 / _ps -> learned_soft
        for seed in a.seeds:
            path = f"{a.ckpt}/{scheme}_s{seed}.pt"
            try:
                net = Link(base, 64, 16)
                net.load_state_dict(torch.load(path, map_location="cpu"))
            except FileNotFoundError:
                print(f"[skip] {path}")
                continue
            p = calibrate(net, xtr)
            net.p_cal = p
            rec = {"p_cal": round(float(p), 4), "before": {}, "after": {}, "val_before": {}, "val_after": {}}
            for snr in a.snrs:
                rec["before"][snr] = round(evaluate(net, xte, snr), 3)
                rec["val_before"][snr] = round(evaluate(net, xva, snr), 3)
                net2 = Link(base, 64, 16)
                net2.load_state_dict(torch.load(path, map_location="cpu"))
                net2.p_cal = p
                recalibrate(net2, xtr, snr)
                rec["after"][snr] = round(evaluate(net2, xte, snr), 3)
                rec["val_after"][snr] = round(evaluate(net2, xva, snr), 3)
                print(f"{scheme:18s} s{seed} {snr:2d} dB: "
                      f"test {rec['before'][snr]:7.3f} -> {rec['after'][snr]:7.3f}"
                      f"   val {rec['val_before'][snr]:7.3f} -> {rec['val_after'][snr]:7.3f}", flush=True)
            res[f"{scheme}_s{seed}"] = rec
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)

    # 汇总：0 dB 处改善，逐种子
    print("\n=== 0 dB 改善（val，重标定后 − 前，负=变好）===")
    for scheme in a.schemes:
        d = []
        for seed in a.seeds:
            k = f"{scheme}_s{seed}"
            if k in res:
                d.append(res[k]["val_after"][0] - res[k]["val_before"][0])
        if d:
            d = np.array(d)
            sd = d.std(ddof=1) if len(d) > 1 else 0.0
            print(f"{scheme:18s} {np.round(d, 3)}  均值 {d.mean():+.3f}  std {sd:.3f}  "
                  f"{'全为负' if (d < 0).all() else '有正'}")


if __name__ == "__main__":
    main()
