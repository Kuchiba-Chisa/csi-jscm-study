# -*- coding: utf-8 -*-
"""v2 统计：E1 公平消融 / E2 共享 vs 专用的配对差值 / 判断点。

约定：NMSE 越小越好（负 dB）。Δ_k = NMSE_共享 − NMSE_专用；
Δ > 0 表示专用模型更好。判断点规则见 00_总方案_v2.md 第 3 节。
判断点只读 val；最终表格读 test。
"""
import json
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "results_v2_eval.json"
ANCHORS = (0, 10, 20)
SHARED = ("learned_soft", "learned_soft_snr")
SPEC = {"learned_soft_f0": 0, "learned_soft_f10": 10, "learned_soft_f20": 20}
CORE = ["analog", "fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr"]

d = json.load(open(SRC, encoding="utf-8"))
snr = np.array(d["snr"], dtype=float)


def seeds_of(scheme, split):
    return sorted(d[split].get(scheme, {}).keys(), key=int)


def curve(scheme, split, metric="nmse"):
    """返回 (n_seeds, n_snr)；没有该方案时返回 None。"""
    ss = seeds_of(scheme, split)
    if not ss:
        return None
    return np.array([d[split][scheme][s][metric] for s in ss], dtype=float)


def mean_std(a, axis=0):
    if a is None:
        return None, None
    m = a.mean(axis=0)
    sdd = a.std(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(m)
    return m, sdd


def idx_of(anchor):
    return int(np.argmin(np.abs(snr - anchor)))


def paired(a, b, split, metric="nmse"):
    """按共同种子配对求 A - B，返回 (均值曲线, 共同种子列表)。"""
    sa, sb = set(seeds_of(a, split)), set(seeds_of(b, split))
    common = sorted(sa & sb, key=int)
    if not common:
        return None, []
    A = np.array([d[split][a][s][metric] for s in common], dtype=float)
    B = np.array([d[split][b][s][metric] for s in common], dtype=float)
    return A - B, common


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def e1(split):
    sec(f"E1 公平消融（{split} 集，均值 ± 样本标准差 ddof=1）")
    print(f"{'scheme':18s}" + "".join(f"{int(s):>8d}" for s in snr))
    for sc in CORE:
        m, sdd = mean_std(curve(sc, split))
        if m is None:
            continue
        print(f"{sc:18s}" + "".join(f"{x:8.2f}" for x in m))
        print(f"{'  ± std':18s}" + "".join(f"{x:8.2f}" for x in sdd))
    print("\n相邻方案的增益（正 = 后者更好；逐 SNR 的配对均值）")
    for a, b, name in [("fixed_hard", "learned_hard", "学习星座"),
                       ("learned_hard", "learned_soft", "软信息直通"),
                       ("learned_soft", "learned_soft_snr", "SNR 条件化")]:
        g, common = paired(a, b, split)
        if g is None:
            print(f"  {name}: 缺方案或缺共同种子")
            continue
        per = g.mean(axis=0)
        print(f"  {name:12s} 均值 {g.mean():+.3f} dB  逐点 " +
              " ".join(f"{x:+.2f}" for x in per) +
              f"  最差 {per.min():+.2f} / 最好 {per.max():+.2f}"
              f"  正号点 {int((per > 0).sum())}/{len(per)}  种子 {','.join(common)}")


def e2(split, judge=False):
    tag = "验证集（判断点）" if judge else "测试集（最终报告）"
    sec(f"E2 共享 vs 专用（{tag}）")
    sh = curve("learned_soft", split)
    if sh is None or not SPEC:
        print("缺方案")
        return
    j = {a: idx_of(a) for a in ANCHORS}
    print(f"{'方案':18s}" + "".join(f"@{a}dB".rjust(9) for a in ANCHORS))
    for sc in SPEC:
        c = curve(sc, split)
        print(f"{sc:18s}" + "".join(
            f"{c.mean(axis=0)[j[a]]:9.2f}" if c is not None else "      n/a" for a in ANCHORS))
    m_sh = sh.mean(axis=0)
    print(f"{'learned_soft':18s}" + "".join(f"{m_sh[j[a]]:9.2f}" for a in ANCHORS))
    c_snr = curve("learned_soft_snr", split)
    if c_snr is not None:
        m_c = c_snr.mean(axis=0)
        print(f"{'learned_soft_snr':18s}" + "".join(f"{m_c[j[a]]:9.2f}" for a in ANCHORS))

    print("\n配对差值 Δ_k = NMSE_共享 − NMSE_专用（正 = 专用更好）")
    for sc, a in SPEC.items():
        c = curve(sc, split)
        if c is None:
            print(f"  {sc}: 缺")
            continue
        k = j[a]
        dk = sh[:, k] - c[:, k]
        mean, sdd = dk.mean(), (dk.std(ddof=1) if len(dk) > 1 else 0.0)
        ok = bool((dk > 0).all() and mean > max(2 * sdd, 1e-9))
        print(f"  @{a:2d} dB  逐种子 {np.array2string(dk, precision=2)}  "
              f"均值 {mean:+.2f}  标准差 {sdd:.2f}  → "
              f"{'专用模型有明确优势' if ok else '未观察到明确优势'}")

    # 模型库口径：每个 SNR 用最近的专用模型
    keys = list(SPEC)
    anchors = [SPEC[k] for k in keys]
    if all(curve(k, split) is not None for k in keys):
        lib = np.stack([curve(k, split).mean(axis=0) for k in keys])
        row_of = {a: i for i, a in enumerate(anchors)}
        lib_curve = np.array([lib[row_of[min(anchors, key=lambda a: abs(a - s))], i]
                              for i, s in enumerate(snr)])
        gap = m_sh - lib_curve                        # 正 = 模型库更好
        print("\n模型库（每个 SNR 取最近的专用模型）相对共享模型的差距"
              "（正 = 模型库更好）")
        print("  " + " ".join(f"{int(s):>5d}" for s in snr))
        print("  " + " ".join(f"{x:+5.2f}" for x in gap))
        print(f"  平均 {gap.mean():+.2f} dB；0 dB {gap[idx_of(0)]:+.2f}；"
              f"10 dB {gap[idx_of(10)]:+.2f}；20 dB {gap[idx_of(20)]:+.2f}")


def storage():
    sec("部署成本")
    n = 1323698
    print(f"  单模型参数 1,323,698（共享 1 个） vs 3 个专用合计 ~3.97 M")
    print(f"  存储比：3 个专用是共享的 {3 * n / n:.1f}×")
    print(f"  fp32 权重：单模型 {n * 4 / 1e6:.1f} MB；三个专用 {3 * n * 4 / 1e6:.1f} MB")


if __name__ == "__main__":
    e1("val")
    e1("test")
    e2("val", judge=True)
    e2("test")
    storage()
