# -*- coding: utf-8 -*-
"""loop_v5.py —— 「若按可靠性分数决定是否重传，能赚多少」

背景：v5 的 Q2/Q3 只证明「能排序」。答辩必然被问「排完序你拿它干什么」。
最直白的用法是重传一次并把两次观测平均（噪声降 3 dB）。

本脚本不训练任何东西，只回答一个上界问题：
在固定码字（同一 B=256 bit 的编码器）下，重传一次的收益由**噪声项占比**决定。
如果 10/20 dB 的逐样本波动几乎全来自信源，重传就是白花比特。

对照组（同等平均比特，1+p）：
  - selective : 按分数挑最可疑的 p 比例重传
  - random    : 随机挑 p 比例重传（同样花比特，但没有信息）
  - oracle    : 按真实 NMSE 挑（作弊，给上界）
  - replay-all: 全部重传（比特 2×，给「不看分数、一律重传」的参照）
"""
import argparse
import json

import numpy as np
import torch

from quality_v5 import (build, decode, load_split, nmse_db_per, q2_reencode,
                        sigma_of, spearman, transmit)

SNRS = [0.0, 10.0, 20.0]


def run(scheme, seed, snr, n, noise_seeds=(1234, 1235), split="test"):
    x = load_split(split)[:n]
    net = build(scheme, seed)
    sig = sigma_of(net, snr)
    _, y1 = transmit(net, x, snr, seed=noise_seeds[0])
    _, y2 = transmit(net, x, snr, seed=noise_seeds[1])

    h1 = decode(net, y1, snr)
    h2 = decode(net, (y1 + y2) / 2.0, snr)     # 两次观测平均 = 噪声降 3 dB
    nm1 = nmse_db_per(h1, x).numpy()
    nm2 = nmse_db_per(h2, x).numpy()

    score2 = q2_reencode(net, y1, h1).numpy()   # 免训练分数，越大越可疑

    out = {"snr": snr, "n": int(n), "sigma": float(sig),
           "nmse_single": float(nm1.mean()), "nmse_replay_all": float(nm2.mean()),
           "score_rho_self": None, "policies": {}}

    out["score_rho_self"] = spearman(score2, nm1)
    out["score_rho_gain"] = spearman(score2, nm1 - nm2)   # 分数与「重传收益」是否对齐

    for p in (0.2, 0.3):
        k = max(1, int(round(p * len(nm1))))
        res = {}
        for name, order in (
            ("selective", np.argsort(-score2)),           # 分数高 = 可疑
            ("gain_oracle", np.argsort(-(nm1 - nm2))),    # 按「重传真实收益」挑，这才是上界
            ("worst_oracle", np.argsort(-nm1)),           # 按「真实最差」挑（≠ 收益最大）
            ("random", np.random.default_rng(0).permutation(len(nm1))),
        ):
            m = nm1.copy()
            m[order[:k]] = nm2[order[:k]]
            res[name] = float(m.mean())
        res["bits"] = 1.0 + p
        out["policies"][f"p={p}"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="learned_soft")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="独立噪声重复次数")
    ap.add_argument("--out", default="results_v5_loop.json")
    a = ap.parse_args()

    res = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        for snr in SNRS:
            key = f"{a.scheme}_s{seed}_snr{int(snr)}"
            reps = [run(a.scheme, seed, snr, a.n,
                        noise_seeds=(1234 + 2 * i, 1235 + 2 * i))
                    for i in range(a.repeats)]
            pol = {}
            for p in reps[0]["policies"]:
                pol[p] = {k: float(np.mean([rp["policies"][p][k] for rp in reps]))
                          for k in reps[0]["policies"][p]}
                pol[p]["bits"] = reps[0]["policies"][p]["bits"]
            r = {
                "snr": snr, "n": int(a.n), "repeats": a.repeats,
                "nmse_single": float(np.mean([rp["nmse_single"] for rp in reps])),
                "nmse_replay_all": float(np.mean([rp["nmse_replay_all"] for rp in reps])),
                "score_rho_self": float(np.mean([rp["score_rho_self"] for rp in reps])),
                "score_rho_gain": float(np.mean([rp["score_rho_gain"] for rp in reps])),
                "std_single": float(np.std([rp["nmse_single"] for rp in reps])),
                "policies": pol,
            }
            res[key] = r
            print(f"[{key}] single={r['nmse_single']:+.3f} "
                  f"replay_all={r['nmse_replay_all']:+.3f} "
                  f"rho(score,nmse)={r['score_rho_self']:+.3f}", flush=True)
            for p, v in r["policies"].items():
                print(f"    {p}: sel={v['selective']:+.3f} gain_or={v['gain_oracle']:+.3f} "
                      f"worst_or={v['worst_oracle']:+.3f} rand={v['random']:+.3f} "
                      f"(single={r['nmse_single']:+.3f})", flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
