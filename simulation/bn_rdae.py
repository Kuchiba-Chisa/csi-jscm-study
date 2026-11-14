# -*- coding: utf-8 -*-
"""RD-JSCC AE 上的 (SNR × 反馈长度) 二维 BN 重标定对照。

三种模式，只动接收端 BatchNorm 的 running stats，不重训任何权重：
  S0  原始（训练累积的 running stats，即"混合条件"统计量）
  S1  在训练集上按**混合条件**重统计（SNR~U(0,20)、Ns~{16,32,64} 逐样本随机）
  S2  在训练集上按**目标条件 (SNR, Ns)** 重统计

用法：
  python bn_rdae.py --seeds 0 1 2 --snrs 0 10 20 --ns 16 32 64 --n 400 --out results_bn_rdae.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from model_rdae import LinkRDAE
from quality_v5 import load_split, nmse_db_per

CKDIR = "ckpt_rdae"
TAG = "rdae_sample_s{}"


def dec_bns(net):
    return [m for n, m in net.dec.named_modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def acc_stats(net, xtr, mode, snr=10.0, ns=64, ns_pool=(16, 32, 64), rounds=2, bs=128, seed=0):
    """用训练数据重统计解码器 BN 的 running stats（momentum=None → 累积平均）。"""
    net.eval()
    bns = dec_bns(net)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    g = torch.Generator().manual_seed(seed)
    N = xtr.shape[0]
    for _ in range(rounds):
      perm = torch.randperm(N, generator=g)
      for i0 in range(0, N, bs):
        xb = xtr[perm[i0:i0 + bs]]
        bs_ = xb.shape[0]
        if mode == "mixed":
            snr_b = (torch.rand(bs_, generator=g) * 20.0)
            pool = torch.tensor(list(ns_pool))
            ns_b = pool[torch.randint(0, len(ns_pool), (bs_,), generator=g)]
        else:
            snr_b = torch.full((bs_,), float(snr))
            ns_b = torch.full((bs_,), int(ns), dtype=torch.long)
        net(xb, snr_b, ns_b)
    for m in bns:
        m.momentum = 0.1
        m.eval()
    net.eval()


@torch.no_grad()
def nmse_at(net, x, snr, ns):
    out = net(x, torch.full((x.shape[0],), float(snr)), torch.full((x.shape[0],), int(ns), dtype=torch.long))
    return float(nmse_db_per(out, x).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="results_bn_rdae.json")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    xtr = load_split("train") / 3.0   # 与 train_rdae.load01 同口径
    xte = load_split("test")[: a.n] / 3.0
    xtr_cal = xtr[:2000]

    res = {}
    for seed in a.seeds:
        net = LinkRDAE(cond=False)
        net.load_state_dict(torch.load(f"{CKDIR}/{TAG.format(seed)}.pt", map_location="cpu"))
        net.eval()
        net.calibrate(xtr_cal, ns_list=tuple(a.ns), snrs=tuple(a.snrs))
        row = {"p_cal_mode": net.p_cal_mode, "cells": {}}
        for snr in a.snrs:
            for ns in a.ns:
                row["cells"][f"{int(snr)}dB_ns{int(ns)}"] = {
                    "S0": nmse_at(net, xte, snr, ns)}
        print(f"seed{seed} S0 done", flush=True)

        acc_stats(net, xtr, "mixed", ns_pool=tuple(a.ns), rounds=a.rounds, seed=seed)
        row["seen_S1"] = a.rounds * xtr.shape[0]
        for snr in a.snrs:
            for ns in a.ns:
                row["cells"][f"{int(snr)}dB_ns{int(ns)}"]["S1"] = nmse_at(net, xte, snr, ns)
        print(f"seed{seed} S1 done ({row['seen_S1']} samples)", flush=True)

        for snr in a.snrs:
            for ns in a.ns:
                net2 = LinkRDAE(cond=False)
                net2.load_state_dict(torch.load(f"{CKDIR}/{TAG.format(seed)}.pt", map_location="cpu"))
                net2.eval()
                net2.calibrate(xtr_cal, ns_list=tuple(a.ns), snrs=tuple(a.snrs))
                acc_stats(net2, xtr, "target", snr=snr, ns=ns, rounds=a.rounds, seed=seed)
                c = row["cells"][f"{int(snr)}dB_ns{int(ns)}"]
                c["S2"] = nmse_at(net2, xte, snr, ns)
                c["gain_S1"] = c["S0"] - c["S1"]
                c["gain_S2"] = c["S0"] - c["S2"]
                print(f"  {int(snr):>3}dB ns{int(ns):>3}  S0 {c['S0']:+.3f}  S1 {c['S1']:+.3f}  S2 {c['S2']:+.3f}"
                      f"   d1 {c['gain_S1']:+.3f}  d2 {c['gain_S2']:+.3f}", flush=True)
        res[f"seed{seed}"] = row

    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("saved", a.out)

    print("\n=== 汇总（三种子均值，Δ = S0 − 该模式，正 = 重标定更好）===")
    keys = sorted({k for r in res.values() for k in r["cells"]},
                  key=lambda s: (int(s.split("dB")[0]), int(s.split("ns")[1])))
    for k in keys:
        g1 = [r["cells"][k]["gain_S1"] for r in res.values()]
        g2 = [r["cells"][k]["gain_S2"] for r in res.values()]
        print(f"{k:>12}  ΔS1 {np.mean(g1):+.3f} ± {np.std(g1, ddof=1) if len(g1) > 1 else 0:.3f}"
              f"   ΔS2 {np.mean(g2):+.3f} ± {np.std(g2, ddof=1) if len(g2) > 1 else 0:.3f}")


if __name__ == "__main__":
    main()
