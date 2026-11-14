"""把 results_v5_*.json 汇总成 markdown 表（供 02_方案/结果_v5.md 使用）。

用法：python summary_v5.py
"""
import json

import numpy as np

SNRS = ("0", "10", "20")
METRICS = ("Q0", "Q1", "Q2", "O1")


def load(fn):
    try:
        return json.load(open(fn, encoding="utf-8"))
    except FileNotFoundError:
        return None


def ms(vals, fmt="{:+.3f}"):
    vals = [v for v in vals if v == v]
    if not vals:
        return "—"
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{fmt.format(m)} ± {s:.3f}"


def main():
    main_j = load("results_v5_main.json")
    qh = load("results_v5_qhead.json")
    bn = load("results_v5_bn.json")

    if bn:
        print("### BN 对照（S0 / S1 / S2a，test，3 种子）\n")
        print("| SNR | S0 | S1（混合 SNR 重统计） | S2a（按 SNR 重统计） | S2a − S0 |")
        print("|---|---|---|---|---|")
        for snr in SNRS:
            row = {v: [] for v in ("S0", "S1", "S2a")}
            for key, d in bn.items():
                for v in row:
                    row[v].append(d[snr][v]["nmse_mean"])
            d0 = np.mean(row["S0"])
            print(f"| {snr} dB | {ms(row['S0'])} | {ms(row['S1'])} | {ms(row['S2a'])} "
                  f"| {np.mean(row['S2a']) - d0:+.3f} |")
        print()

    if main_j:
        for var in ("S0", "S2a"):
            print(f"### 逐样本指标（learned_soft，{var}，test，3 种子 × 3 噪声）\n")
            print("| SNR | NMSE(dB) | 指标 | AUROC | AUPRC | Spearman | 风险@100% | @80% | @70% | pAURC70-100 |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for snr in SNRS:
                keys = [k for k in main_j if k.endswith("_" + var)]
                for mname in METRICS:
                    au, ap, rho, r100, r80, r70, pa = ([] for _ in range(7))
                    nms = []
                    for k in keys:
                        for run in main_j[k][snr]:
                            d = run["metrics"][mname]
                            au.append(d["auroc"])
                            ap.append(d["auprc"])
                            rho.append(d["spearman"])
                            r = d["risk"]
                            r100.append(r["1.00"])
                            r80.append(r["0.80"])
                            r70.append(r["0.70"])
                            pa.append(d["aurc70_100"])
                            nms.append(run["nmse_mean"])
                    print(f"| {snr} dB | {ms(nms) if mname == 'Q0' else ''} | {mname} "
                          f"| {ms(au, '{:.3f}')} | {ms(ap, '{:.3f}')} | {ms(rho)} "
                          f"| {ms(r100)} | {ms(r80)} | {ms(r70)} | {ms(pa)} |")
            print()

    if qh:
        print("### 质量头 Q3（test，3 种子）\n")
        print("| SNR | AUROC | Spearman | MAE (dB) | 校准偏差 |")
        print("|---|---|---|---|---|")
        for var in ("S0", "S2a"):
            for snr in SNRS:
                au, rho, mae, ece = ([] for _ in range(4))
                for k, d in qh.items():
                    if not k.endswith("_" + var):
                        continue
                    s = d[snr]
                    au.append(s["auroc"])
                    rho.append(s["spearman"])
                    mae.append(s["mae"])
                    ece.append(s["calibration"])
                print(f"| {var} {snr} dB | {ms(au, '{:.3f}')} | {ms(rho)} | "
                      f"{np.mean(mae):.2f} | {np.mean(ece):.2f} |")
        print()


if __name__ == "__main__":
    main()
