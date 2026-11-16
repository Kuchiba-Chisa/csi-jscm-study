"""Fit global output scale/offset on validation data for each model and SNR.

The sign and size of the fitted coefficient are empirical. In the recorded CNN
pilot, a<1 at low SNR reduces output amplitude; it does not repair underscaling.
Fit minimizes elementwise linear MSE; report averages per-H NMSE in dB.
The test-fit column is a diagnostic, not a deployable estimator or a rigorous
upper bound for the mean-dB metric. Replay and scale-only controls are in
experiments/output_calibration_20261115/audit_affine.py.
"""
import json
import os

import torch

from model_jscm import nmse_db, calibrate
from quality_v5 import build, load_split, transmit, decode, nmse_db_per

SNRS = tuple(range(0, 21, 2))
SEEDS = (0, 1, 2)
SCHEME = "learned_soft"
BASE_NSEED = 1234


def fit_affine(pred, target):
    """最小二乘拟合标量 a、b（对所有元素统一）。返回 (a, b)。"""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    pm, tm = p.mean(), t.mean()
    dp = p - pm
    var = (dp * dp).mean()
    cov = (dp * (t - tm)).mean()
    a = float(cov / (var + 1e-12))
    b = float(tm - a * pm)
    return a, b


@torch.no_grad()
def run_seed(seed):
    net = build(SCHEME, seed, ckpt_dir="ckpt_v2")
    tst = load_split("test")
    val = load_split("val")
    rows = []
    for snr in SNRS:
        yv = transmit(net, val, snr, seed=BASE_NSEED + snr)[1]
        hv = decode(net, yv, snr)
        a, b = fit_affine(hv, val)

        yt = transmit(net, tst, snr, seed=BASE_NSEED + snr + 7)[1]
        ht = decode(net, yt, snr)
        base = float(nmse_db_per(ht, tst).mean())
        cal = float(nmse_db_per(a * ht + b, tst).mean())

        a_o, b_o = fit_affine(ht, tst)
        orc = float(nmse_db_per(a_o * ht + b_o, tst).mean())

        rows.append(dict(snr=snr, base=base, affine=cal, delta=base - cal,
                         oracle=orc, oracle_delta=base - orc, a=a, b=b))
    return rows


def main():
    out = {}
    for seed in SEEDS:
        rows = run_seed(seed)
        out[f"s{seed}"] = rows
        print(f"--- seed {seed} ---")
        for r in rows:
            print(f"SNR {r['snr']:2d}: base {r['base']:+.3f}  affine {r['affine']:+.3f} "
                  f"(Δ {r['delta']:+.3f})  oracleΔ {r['oracle_delta']:+.3f}  a={r['a']:.3f} b={r['b']:+.4f}")
    with open("results_affine.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n=== 三种子均值（Δ = base - affine，正 = 校准更好）===")
    for i, snr in enumerate(SNRS):
        d = [out[f"s{s}"][i]["delta"] for s in SEEDS]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        a = [out[f"s{s}"][i]["a"] for s in SEEDS]
        print(f"SNR {snr:2d}: Δ {m:+.3f} ± {sd:.3f}   a≈{sum(a)/3:.3f}")


if __name__ == "__main__":
    main()
