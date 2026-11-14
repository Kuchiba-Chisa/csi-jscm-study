# -*- coding: utf-8 -*-
"""把 v2 的结果汇总成一张可贴进文档/PPT 的表（只读 test 集，val 只看判断点）。"""
import json

EVAL = "results_v2_eval.json"
DIST = "results_distill.json"

d = json.load(open(EVAL, encoding="utf-8"))
snr = d["snr"]
CORE = ["analog", "fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr"]
SPEC = ["learned_soft_f0", "learned_soft_f10", "learned_soft_f20"]


def curve(sc, split, seed=None, metric="nmse"):
    seeds = sorted(d[split][sc], key=int) if sc in d[split] else []
    if not seeds:
        return None
    if seed is None:
        return [sum(d[split][sc][s][metric][i] for s in seeds) / len(seeds)
                for i in range(len(snr))]
    return d[split][sc][str(seed)][metric]


print("SNR      " + " ".join(f"{s:>7d}" for s in snr))
for sc in CORE + ["learned_soft_f0", "learned_soft_f10", "learned_soft_f20"]:
    c = curve(sc, "test")
    if c is None:
        continue
    print(f"{sc:22s}" + " ".join(f"{x:7.2f}" for x in c))

print("\n--- 蒸馏/重加权（E3b / E4），test 集 ---")
try:
    dd = json.load(open(DIST, encoding="utf-8"))
    base = curve("learned_soft", "test")
    spec0 = curve("learned_soft_f0", "test")
    i0 = snr.index(0)
    print(f"{'shared'}:   @0dB {base[i0]:.2f}")
    print(f"{'specialist_f0'}: @0dB {spec0[i0]:.2f}")
    for tag, seeds in dd["test"].items():
        cur = [sum(seeds[s]["nmse"][i] for s in seeds) / len(seeds) for i in range(len(snr))]
        print(f"{tag:22s}@0dB {cur[i0]:.2f}   @10dB {cur[snr.index(10)]:.2f}   "
              f"@20dB {cur[snr.index(20)]:.2f}")
except FileNotFoundError:
    print("（还没有 results_distill.json）")
