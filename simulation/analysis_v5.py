"""v5 分析（D5）：风险—覆盖率细节、选择性质量、残差三项分解、Q1×Q2 关系。"""
import glob
import json

import numpy as np

import quality_v5 as Q

SNRS = ("0", "10", "20")


def load_npz(fn="results_v5_arrays.npz"):
    return np.load(fn)


def keys_by(z, var):
    ks = sorted({k.rsplit("|", 2)[0] for k in z.files})
    return [k for k in ks if k.endswith("_" + var)]


def main():
    z = load_npz()
    var = json.load(open("results_v5_arrays_var.json", encoding="utf-8"))
    qh = json.load(open("results_v5_qhead.json", encoding="utf-8"))

    print("## 1 残差三项分解（与 NMSE_dB 的 Spearman）\n")
    print("| 变体 | SNR | ρ(n²) | ρ(‖tx−F(Ĥ)‖²) | ρ(cross) | 均值 n² | 均值 rec² | 均值 cross |")
    print("|---|---|---|---|---|---|---|---|")
    for v in ("S0", "S2a"):
        for snr in SNRS:
            rows = []
            for k in keys_by(z, v):
                nm = z[f"{k}|{snr}|nmse"]
                rows.append([Q.spearman(z[f"{k}|{snr}|{m}"], nm)
                             for m in ("n2", "rec2", "cross")])
                rows[-1] += [float(z[f"{k}|{snr}|{m}"].mean()) for m in ("n2", "rec2", "cross")]
            arr = np.array(rows)          # (n_seed, 6): rho_n2, rho_rec, rho_cross, m_n2, m_rec, m_cross
            m = arr.mean(0)
            print(f"| {v} | {snr} dB | {m[0]:+.3f} | {m[1]:+.3f} | {m[2]:+.3f} "
                  f"| {m[3]:.2f} | {m[4]:.2f} | {m[5]:+.2f} |")
    print()

    print("## 2 风险—覆盖率（Q2，均值±std，含理想与随机参照）\n")
    print("| 变体 | SNR | 全部 | Q2@95% | @90% | @80% | @70% | 理想@70% | 随机@70% |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in ("S0", "S2a"):
        for snr in SNRS:
            cur = {c: [] for c in (1.0, 0.95, 0.9, 0.8, 0.7)}
            ideal70, rand70 = [], []
            for k in keys_by(z, v):
                nm = z[f"{k}|{snr}|nmse"]
                s = z[f"{k}|{snr}|Q2"]
                rc = Q.risk_curve(s, nm)
                for c in cur:
                    cur[c].append(rc[c])
                ideal70.append(Q.ideal_curve(nm)[0.7])
                rand70.append(float(nm.mean()))
            f = lambda a: f"{np.mean(a):+.3f} ± {np.std(a, ddof=1):.3f}"
            print(f"| {v} | {snr} dB | {f(cur[1.0])} | {f(cur[0.95])} | {f(cur[0.9])} "
                  f"| {f(cur[0.8])} | {f(cur[0.7])} | {np.mean(ideal70):+.3f} | {np.mean(rand70):+.3f} |")
    print()

    print("## 3 选择性质量（按 Q2 剔除最可疑的 10% / 20%，剩余集合的平均平方 SGCS）\n")
    print("| 变体 | SNR | 全部 | 剔除 10% | 剔除 20% | 提升@20% |")
    print("|---|---|---|---|---|---|")
    for v in ("S0", "S2a"):
        for snr in SNRS:
            base, k10, k20 = [], [], []
            for k in keys_by(z, v):
                sg = z[f"{k}|{snr}|sgcs"]
                s = z[f"{k}|{snr}|Q2"]
                order = np.argsort(s)
                base.append(float(sg.mean()))
                n = len(sg)
                k10.append(float(sg[order[:int(n * 0.9)]].mean()))
                k20.append(float(sg[order[:int(n * 0.8)]].mean()))
            print(f"| {v} | {snr} dB | {np.mean(base):.4f} | {np.mean(k10):.4f} "
                  f"| {np.mean(k20):.4f} | {np.mean(k20) - np.mean(base):+.4f} |")
    print()

    print("## 4 方差分解（test 前 200 个 H × 20 次噪声，3 种子）\n")
    print("| SNR | Ŵ (dB²) | B̂ (dB²) | 噪声占比 W/(W+B̂) | bootstrap 95% |")
    print("|---|---|---|---|---|")
    for snr in SNRS:
        rows = [v for k, v in var.items() if k.endswith(f"|{snr}")]
        print(f"| {snr} dB | {np.mean([r['W'] for r in rows]):.3f} "
              f"| {np.mean([r['B_hat'] for r in rows]):.3f} "
              f"| {np.mean([r['frac_within'] for r in rows]):.3f} "
              f"| [{np.mean([r['ci_lo'] for r in rows]):.3f}, "
              f"{np.mean([r['ci_hi'] for r in rows]):.3f}] |")
    print()

    print("## 5 Q1 × Q2：BN 修正前后各指标 AUROC\n")
    print("| SNR | Q2 (S0) | Q2 (S2a) | Q3 (S0) | Q3 (S2a) |")
    print("|---|---|---|---|---|")
    for snr in SNRS:
        def au(var_name, metric):
            vals = []
            for k in keys_by(z, var_name):
                nm = z[f"{k}|{snr}|nmse"]
                vals.append(Q.auroc(z[f"{k}|{snr}|{metric}"], Q.bad_label(nm)))
            return np.mean(vals)
        def au_qh(var_name):
            vals = [v[snr]["auroc"] for k, v in qh.items() if k.endswith("_" + var_name)]
            return np.mean(vals)
        print(f"| {snr} dB | {au('S0', 'Q2'):.3f} | {au('S2a', 'Q2'):.3f} "
              f"| {au_qh('S0'):.3f} | {au_qh('S2a'):.3f} |")


if __name__ == "__main__":
    main()
