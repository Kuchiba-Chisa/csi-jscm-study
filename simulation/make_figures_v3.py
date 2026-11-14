# -*- coding: utf-8 -*-
"""v3 图：BN 统计量分解 + E5 修正前后。

产物：
  fig_bn_decomp.png  0 dB 处「默认统计量 → 按该 SNR 重统计」的分组柱，叠专用模型水平线
  fig_e5_v3.png      E5 三接口（修正版）曲线 + v2 缺陷版本对照 + 端到端参考
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, TEAL, RED, AMBER, GRAY = "#1F2328", "#6B7280", "#00798C", "#D1495B", "#C98A1E", "#9AA5B1"


def curves(fn):
    d = json.load(open(fn, encoding="utf-8"))
    snr = d["snr"]
    out = {}
    for split in ("val", "test"):
        out[split] = {}
        for n, byseed in d[split].items():
            seeds = [byseed[k]["nmse"] for k in sorted(byseed, key=int)]
            out[split][n] = np.array([st.mean([s[i] for s in seeds]) for i in range(len(snr))])
    return np.array(snr), out


def recal_gain(fn, scheme, snr=0):
    d = json.load(open(fn, encoding="utf-8"))
    g = []
    for k, v in d.items():
        if k.startswith(scheme + "_s"):
            g.append(v["val_before"][str(snr)] - v["val_after"][str(snr)])  # before - after
    return float(np.mean(g)) if g else float("nan"), len(g)


def draw_bn(snr_v2, ev2, path="fig_bn_decomp.png"):
    snr, i0 = snr_v2, int(np.where(snr_v2 == 0)[0][0])
    schemes = [("fixed_hard", "固定 16-PAM", GRAY), ("learned_hard", "可学星座", AMBER),
               ("learned_soft", "可学星座+观测直入", TEAL)]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    xs, labels = [], []
    for j, (name, lab, col) in enumerate(schemes):
        base = ev2["test"][name][i0]
        g, n = recal_gain("bn_recal.json", name)
        ax.bar(j * 2.0, base, 0.8, color=col, alpha=0.35, edgecolor="white")
        ax.bar(j * 2.0 + 0.85, base - g, 0.8, color=col, edgecolor="white")
        ax.text(j * 2.0, base - 0.06, f"{base:.2f}", ha="center", va="top", fontsize=8, color=INK)
        ax.text(j * 2.0 + 0.85, base - g - 0.06, f"{base-g:.2f}", ha="center", va="top",
                fontsize=8, color="white")
        ax.text(j * 2.0 + 0.4, 0.04, f"{lab}\n重统计 {g:+.2f} dB", ha="center", fontsize=8.5, color=MUTED)
        xs += [j * 2.0, j * 2.0 + 0.85]
        labels += ["默认", "重统计"]

    f0 = ev2["test"]["learned_soft_f0"][i0]
    ax.axhline(f0, color=RED, ls="--", lw=1.4)
    ax.text(4.1, f0 + 0.03, f"专用 f0（只在 0 dB 训练）{f0:.2f}", color=RED, fontsize=8.5, ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NMSE @ 0 dB（dB，越低越好）")
    ax.set_title("0 dB 的差距有一半只是 BN 统计量没对齐（test，3 种子均值）", fontsize=11.5)
    ax.set_ylim(-2.15, 0)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def draw_e5(path="fig_e5_v3.png"):
    snr_a, a = curves("results_v2_eval.json")
    snr_b, b = curves("results_v3_rx.json")
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(snr_a, a["test"]["learned_soft"], "k:", lw=1.8, label="端到端联合训练（参考）")
    for name, lab, col in (("learned_hard", "硬判决接收端", AMBER),
                           ("learned_soft", "软信息直通", TEAL),
                           ("learned_soft_snr", "软信息+噪声水平", "#7B3FA0")):
        ax.plot(snr_b, b["test"][name], "o-", color=col, lw=1.9, ms=3.5, label=f"{lab}（修正后）")
    # v2 缺陷版本：只有 soft+snr 这一条一眼可见异常
    v2 = json.load(open("results_rx.json", encoding="utf-8"))
    seeds = [v2["test"]["learned_soft_snr"][k]["nmse"] for k in sorted(v2["test"]["learned_soft_snr"], key=int)]
    ax.plot(v2["snr"], np.mean(seeds, axis=0), "s--", color=RED, lw=1.4, ms=3,
            label="软信息+噪声水平（v2，BN 缺陷）")

    ax.annotate("v2 的 0 dB 异常：\n只重训接收反而好 0.41 dB", xy=(0.05, -1.70), xytext=(0.3, -1.14),
                fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_xlabel("反馈信道 SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_ylim(-4.6, -1.0)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.grid(color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    snr_v2, ev2 = curves("results_v2_eval.json")
    draw_bn(snr_v2, ev2)
    draw_e5()
