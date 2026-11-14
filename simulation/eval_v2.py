# -*- coding: utf-8 -*-
"""E0 重评：对 ckpt_v2/ 下所有 checkpoint 做
  (1) 训练集标定全局 P_tx 并冻结 sigma（等 SNR 口径）
  (2) 验证集 + 测试集双评估
  (3) 平方 SGCS（SGCS^2 = mean |h^H h_hat|^2 / (|h|^2 |h_hat|^2)）
结果写 results_v2_eval.json，按 (scheme, seed) 存逐点曲线，便于配对求差。

判断点只读 val；图表和最终表格只读 test。
用法：.venv/Scripts/python.exe eval_v2.py [--only a,b] [--out results_v2_eval.json]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

import os
from model_jscm import Link, nmse_db, cos_flat, sgcs, calibrate
from run_experiments import load_split, SNR_GRID

CKPT_RE = re.compile(r"^(?P<scheme>[a-z0-9_]+?)_s(?P<seed>\d+)\.pt$")


def sgcs_sq(pred, target):
    """平方广义余弦相似度：|h^H h_hat|^2 / (|h|^2 |h_hat|^2)，逐子载波后平均。"""
    hp = torch.complex(pred[:, 0], pred[:, 1])
    ht = torch.complex(target[:, 0], target[:, 1])
    num = (ht.conj() * hp).sum(1).abs().pow(2)
    den = ht.abs().pow(2).sum(1) * hp.abs().pow(2).sum(1) + 1e-12
    return (num / den).mean()


@torch.no_grad()
def run_curve(net, x, snrs=SNR_GRID, noise_seed=20260101):
    """逐 SNR 评估。每个 SNR 用同一噪声种子，所有方案看到同一组噪声实现。"""
    net.eval()
    out = {"nmse": [], "sgcs2": [], "sgcs1": [], "cos_flat": [], "p_tx": []}
    for s in snrs:
        torch.manual_seed(noise_seed + int(round(s)))
        pred = net(x, float(s))
        out["nmse"].append(float(nmse_db(pred, x)))
        out["sgcs2"].append(float(sgcs_sq(pred, x)))
        out["sgcs1"].append(float(sgcs(pred, x)))
        out["cos_flat"].append(float(cos_flat(pred, x)))
        out["p_tx"].append(float(net.last_tx_power))
    return {k: [round(v, 4) for v in vals] for k, vals in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2")
    ap.add_argument("--out", default="results_v2_eval.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--ns", type=int, default=64)
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()

    xtr = load_split("train")
    xva = load_split("val")
    xte = load_split("test")

    files = sorted(glob.glob(os.path.join(args.ckpt, "*.pt")))
    want = [w for w in args.only.split(",") if w]
    out = {"snr": SNR_GRID.tolist(), "ns": args.ns, "m": args.m,
           "p_cal": {}, "val": {}, "test": {}, "note": "E0 重评：冻结 P_tx，SGCS 带平方"}
    if os.path.exists(args.out):
        out = json.load(open(args.out, encoding="utf-8"))

    for f in files:
        mm = CKPT_RE.match(os.path.basename(f))
        if not mm:
            print("skip", f)
            continue
        scheme, seed = mm.group("scheme"), int(mm.group("seed"))
        if want and scheme not in want:
            continue
        # learned_soft_f0 这类只表示「该模型训在固定 0 dB」，网络结构仍是 learned_soft
        base = re.sub(r"_f\d+$", "", scheme).removesuffix("_ps")   # B2 的 _ps 与基础架构相同
        net = Link(base, args.ns, args.m, power_ref=True)
        net.load_state_dict(torch.load(f, map_location="cpu"))
        p = calibrate(net, xtr)
        net.p_cal = p
        rv = run_curve(net, xva)
        rt = run_curve(net, xte)
        if p == 0:
            raise RuntimeError("标定失败")
        out["p_cal"].setdefault(scheme, []).append(round(p, 4))
        out["val"].setdefault(scheme, {})[str(seed)] = rv
        out["test"].setdefault(scheme, {})[str(seed)] = rt
        print(f"{scheme:18s} s{seed}  P_tx={p:.3f}  "
              f"test@10dB nmse={rt['nmse'][5]:+.2f} dB  sgcs2={rt['sgcs2'][5]:.3f}")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

    print("saved", args.out)


if __name__ == "__main__":
    main()
