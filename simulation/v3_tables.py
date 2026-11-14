# -*- coding: utf-8 -*-
"""v3 结果表：一处算清、直接粘进 结果_v3.md。

来源：
  results_v2_eval.json   E1/E2 主表（test/val，3 种子，噪声种子 20260101）
  bn_recal.json          P0/B1：默认统计量 vs 按该 SNR 重统计（含专用模型）
  results_v3_ps_eval.json  逐样本抽 SNR 重训（_ps）
  results_v3_rx.json     E5 修正版
  results_rx.json        E5 v2 缺陷版（对照）
  bn_stats/              B1 统计量文件大小
"""
import json
import os
import statistics as st

SNR = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def pick(d, scheme):
    """只取 scheme 自己的种子键（避免 learned_soft 命中 learned_soft_snr / _ps）。"""
    return [v for k, v in d.items() if k.startswith(scheme + "_s") and k[len(scheme) + 2].isdigit()]
ANCH = [0, 10, 20]


def load(fn):
    with open(fn, encoding="utf-8") as f:
        return json.load(f)


def curve(d, split, scheme, key="nmse"):
    """返回 (3 种子均值曲线, 0 dB 处种子 std)。"""
    by = d[split][scheme]
    seeds = [by[k][key] for k in sorted(by, key=int)]
    m = [st.mean([s[i] for s in seeds]) for i in range(len(SNR))]
    sd = st.stdev([s[0] for s in seeds]) if len(seeds) > 1 else 0.0
    return m, sd, len(seeds)


def fmt(row, idx=ANCH):
    return " / ".join(f"{row[SNR.index(s)]:+.2f}" for s in idx)


def main():
    ev2 = load("results_v2_eval.json")
    rec = load("bn_recal.json")
    rec_s = load("bn_recal_spec.json")
    ps = load("results_v3_ps_eval.json")
    rx3 = load("results_v3_rx.json")
    rx2 = load("results_rx.json")

    print("### A. E1 主表（test，3 种子均值）")
    for s in ["analog", "fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr"]:
        m, sd, n = curve(ev2, "test", s)
        print(f"{s:20s} {fmt(m)}   std@0dB={sd:.3f} n={n}")

    print("\n### B. E2 专用 / 共享（test）")
    for s in ["learned_soft_f0", "learned_soft_f10", "learned_soft_f20"]:
        m, sd, n = curve(ev2, "test", s)
        print(f"{s:20s} {fmt(m)}   std@0dB={sd:.3f} n={n}")

    print("\n### C. P0/B1：0 dB 处默认统计量 → 按该 SNR 重统计（test）")
    for s in ["fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr",
              "learned_soft_f0", "learned_soft_f10", "learned_soft_f20"]:
        vs = pick(rec, s) or pick(rec_s, s)
        b = [v["before"]["0"] for v in vs]
        a = [v["after"]["0"] for v in vs]
        print(f"{s:20s} {st.mean(b):+.3f} → {st.mean(a):+.3f}   收回 {st.mean(b)-st.mean(a):+.3f} dB  n={len(b)}")

    print("\n### D. 共享 vs 专用 f0：0 dB 差距（test）")
    sh_b = st.mean([v["before"]["0"] for v in pick(rec, "learned_soft")])
    sh_a = st.mean([v["after"]["0"] for v in pick(rec, "learned_soft")])
    sp = st.mean([v["before"]["0"] for v in pick(rec_s, "learned_soft_f0")])
    print(f"共享默认 {sh_b:+.3f} / 共享重统计 {sh_a:+.3f} / 专用 {sp:+.3f}")
    print(f"差距 {sh_b - sp:.3f} → {sh_a - sp:.3f} dB")

    print("\n### E. 逐样本抽 SNR 重训（test）")
    for s in ["learned_hard_ps", "learned_soft_ps", "learned_soft_snr_ps"]:
        m, sd, n = curve(ps, "test", s)
        print(f"{s:22s} {fmt(m)}   std@0dB={sd:.3f} n={n}")

    print("\n### F. 叠加 BN 重统计的收益（ps 模型，val 0 dB）")
    for s in ["learned_hard_ps", "learned_soft_ps", "learned_soft_snr_ps"]:
        d = load("bn_recal_ps.json")
        g = [v["val_before"]["0"] - v["val_after"]["0"] for k, v in d.items() if k.startswith(s + "_s")]
        if g:
            print(f"{s:22s} 收回 {st.mean(g):+.3f} dB  n={len(g)}")

    print("\n### G. E5 修正版（test，3 种子）")
    for s in ["learned_hard", "learned_soft", "learned_soft_snr"]:
        m, sd, n = curve(rx3, "test", s)
        print(f"{s:18s} {fmt(m)}   std@0dB={sd:.3f} n={n}")
    print("--- v2 缺陷版（对照）")
    for s in ["learned_hard", "learned_soft", "learned_soft_snr"]:
        m, sd, n = curve(rx2, "test", s)
        print(f"{s:18s} {fmt(m)}   n={n}")

    print("\n### H. 端到端参考（E1 里的 learned_soft，test）")
    m, sd, n = curve(ev2, "test", "learned_soft")
    print(f"learned_soft         {fmt(m)}   n={n}")

    print("\n### I. B1 统计量文件")
    for f in sorted(os.listdir("bn_stats"))[:3]:
        print(f"{f}  {os.path.getsize(os.path.join('bn_stats', f))} B")
    n = len([f for f in os.listdir("bn_stats") if "_bn.pt" in f])
    tot = sum(os.path.getsize(os.path.join("bn_stats", f)) for f in os.listdir("bn_stats"))
    print(f"共 {n} 个文件，总计 {tot/1024:.0f} KB")


if __name__ == "__main__":
    main()
