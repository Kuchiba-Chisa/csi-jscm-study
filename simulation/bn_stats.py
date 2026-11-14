# -*- coding: utf-8 -*-
"""B1：把解码器 BN 统计量按 SNR 存成小文件，评估时按当前 SNR 取用。

目的：证明「共享模型在低 SNR 的劣势」有一部分不是训练不足，而是
BN running stats 是跨 SNR 的混合量；改用「每个 SNR 一组统计量」即可拿回。

产物：
    bn_stats/{scheme}_s{seed}_bn.pt     每组 224 个数（decoder 3 层 BN 的 mean+var）
    results_v3_b1.json

    python bn_stats.py --schemes learned_soft learned_soft_f0 --seeds 0 1 2
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from torch import nn

from model_jscm import Link, nmse_db, calibrate
from run_experiments import load_split

SNR_GRID = list(range(0, 21, 2))


def dec_bns(net):
    return [m for n, m in net.named_modules() if n.startswith("dec.") and isinstance(m, nn.BatchNorm2d)]


def build(scheme, seed):
    base = re.sub(r"_f\d+$", "", scheme).removesuffix("_ps")
    net = Link(base, 64, 16)
    net.load_state_dict(torch.load(f"ckpt_v2/{scheme}_s{seed}.pt", map_location="cpu"))
    return net


@torch.no_grad()
def snapshot_at(net, xtr, snr, bs=128, n_batch=40):
    """在该 SNR 下累积解码器 BN 统计量，返回 {层名: (mean, var)}。"""
    bns = dec_bns(net)
    names = [n for n, m in net.named_modules() if n.startswith("dec.") and isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    net.eval()
    for m in bns:
        m.train()
    for i in range(0, min(len(xtr), bs * n_batch), bs):
        net(xtr[i:i + bs], float(snr))
    snap = {}
    for n, m in zip(names, bns):
        snap[n + ".mean"] = m.running_mean.detach().clone()
        snap[n + ".var"] = m.running_var.detach().clone()
        m.eval()
    return snap


@torch.no_grad()
def apply_stats(net, snap):
    names = [n for n, m in net.named_modules() if n.startswith("dec.") and isinstance(m, nn.BatchNorm2d)]
    for n in names:
        m = dict(net.named_modules())[n]
        m.running_mean.copy_(snap[n + ".mean"])
        m.running_var.copy_(snap[n + ".var"])
        m.eval()


@torch.no_grad()
def evaluate(net, x, snr, noise_seed=1234):
    torch.manual_seed(noise_seed)
    return float(nmse_db(net(x, float(snr)), x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["learned_soft", "learned_soft_f0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results_v3_b1.json")
    a = ap.parse_args()

    xtr, xva, xte = load_split("train"), load_split("val"), load_split("test")
    os.makedirs("bn_stats", exist_ok=True)
    res, size_bytes = {}, None

    for scheme in a.schemes:
        for seed in a.seeds:
            net = build(scheme, seed)
            net.p_cal = calibrate(net, xtr)
            rec = {"p_cal": round(float(net.p_cal), 4), "base": {}, "b1": {}}
            stats = {}
            for snr in SNR_GRID:
                net_base = build(scheme, seed); net_base.p_cal = net.p_cal
                net_base.eval()          # 必须：默认是 train 模式，会用 batch 统计量，不是 running stats
                rec["base"][snr] = {"val": round(evaluate(net_base, xva, snr), 3),
                                    "test": round(evaluate(net_base, xte, snr), 3)}
                snap_net = build(scheme, seed)
                snap_net.p_cal = net.p_cal      # v4 §4.3.2：统计量的副本也必须冻结同一 P_tx
                snap = snapshot_at(snap_net, xtr, snr)
                stats[snr] = snap
                net_b1 = build(scheme, seed); net_b1.p_cal = net.p_cal
                net_b1.eval()
                apply_stats(net_b1, snap)
                rec["b1"][snr] = {"val": round(evaluate(net_b1, xva, snr), 3),
                                  "test": round(evaluate(net_b1, xte, snr), 3)}
                print(f"{scheme:16s} s{seed} {snr:2d} dB  val {rec['base'][snr]['val']:7.3f} -> "
                      f"{rec['b1'][snr]['val']:7.3f}   test {rec['base'][snr]['test']:7.3f} -> "
                      f"{rec['b1'][snr]['test']:7.3f}", flush=True)
            path = f"bn_stats/{scheme}_s{seed}_bn.pt"
            torch.save(stats, path)
            size_bytes = os.path.getsize(path)
            res[f"{scheme}_s{seed}"] = rec
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    nfl = sum(v.numel() for v in stats[SNR_GRID[0]].values())
    print(f"\n每组统计量 {nfl} 个数；{len(SNR_GRID)} 组共 {size_bytes/1024:.1f} KB "
          f"（单模型 5.3 MB 的 {size_bytes/1024/1024/5.3*100:.2f}%）")
    print("saved", a.out)


if __name__ == "__main__":
    main()
