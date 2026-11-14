"""v2 出图：功率口径修正 + 标准 SGCS + 3 个训练种子（均值 ± 样本标准差 ddof=1）。

输入：results_v2_eval.json（冻结 P_tx、平方 SGCS、val/test 双评估）
      results_distill.json（E3b / E4）
      results_rx.json（E5：冻结发送端只换接收）
输出：
    fig_curve_v2.png       四条主线的 NMSE / SGCS（带 = seed 间 std）
    fig_q3.png             E2：专用 vs 共享 vs 条件共享 + 模型库缺口
    fig_budget.png         E2/E4：0 dB 差距与蒸馏折中
    fig_rx_v2.png          E5：同一发送端，三种接收接口
    fig_constellation_v2.png / fig_recon_v2.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_experiments import load_split
from model_jscm import Link, nmse_db
from eval_v2 import sgcs_sq, run_curve

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CORE = ["analog", "fixed_hard", "learned_hard", "learned_soft"]
C = {"analog": "#8c8c8c", "fixed_hard": "#d1495b", "learned_hard": "#edae49",
     "learned_soft": "#00798c", "learned_soft_snr": "#7b3fa0"}
SHORT = {"analog": "无星座约束（参考上界）", "fixed_hard": "固定 16-PAM + 硬判决",
         "learned_hard": "可学习 16-PAM + 硬判决", "learned_soft": "可学习 PAM + 观测直入",
         "learned_soft_snr": "同上 + 输入噪声水平"}
FIX = {"learned_soft_f0": 0.0, "learned_soft_f10": 10.0, "learned_soft_f20": 20.0}
FIX_C = {"learned_soft_f0": "#c0392b", "learned_soft_f10": "#d1495b",
         "learned_soft_f20": "#edae49"}
EVAL = "results_v2_eval.json"


def load(path=EVAL, split="test"):
    d = json.load(open(path, encoding="utf-8"))
    snr = np.array(d["snr"], dtype=float)
    st = {}
    for k, seeds in d[split].items():
        nm = np.array([v["nmse"] for v in seeds.values()], dtype=float)
        sg = np.array([v["sgcs2"] for v in seeds.values()], dtype=float)
        st[k] = {"nmse": nm.mean(0),
                 "nmse_std": nm.std(0, ddof=1) if len(nm) > 1 else np.zeros_like(nm[0]),
                 "sgcs": sg.mean(0),
                 "sgcs_std": sg.std(0, ddof=1) if len(sg) > 1 else np.zeros_like(sg[0]),
                 "n": len(nm)}
    return snr, st


def draw_curve(snr, st, path="fig_curve_v2.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for k in CORE:
        if k not in st:
            continue
        s = st[k]
        for ax, key, std, lab in ((axes[0], "nmse", "nmse_std", "NMSE (dB)"),
                                  (axes[1], "sgcs", "sgcs_std", "SGCS²（逐子载波）")):
            ax.plot(snr, s[key], "o-", ms=4, lw=2, color=C[k],
                    label=SHORT[k] if ax is axes[0] else None)
            ax.fill_between(snr, s[key] - s[std], s[key] + s[std], color=C[k], alpha=.15, lw=0)
    axes[0].set_ylabel("NMSE (dB)"); axes[1].set_ylabel("SGCS²（逐子载波）")
    axes[0].set_title("(a) 重构误差：越低越好"); axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_title("(b) 波束方向一致性：越高越好"); axes[1].grid(alpha=.3)
    for ax in axes:
        ax.set_xlabel("反馈信道 SNR (dB)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("saved", path)


def draw_q3(snr, st, path="fig_q3.png"):
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    for k, s0 in FIX.items():
        if k not in st:
            continue
        s = st[k]
        ax.plot(snr, s["nmse"], "o--", ms=4, lw=1.8, color=FIX_C[k],
                label=f"专用模型（只在 {s0:.0f} dB 训练）")
        ax.axvline(s0, color=FIX_C[k], ls=":", lw=1, alpha=.5)
    for k, ls in (("learned_soft", "-"), ("learned_soft_snr", "-.")):
        if k not in st:
            continue
        s = st[k]
        ax.plot(snr, s["nmse"], "o" + ls, ms=5, lw=2.6, color=C[k], label=SHORT[k])
        ax.fill_between(snr, s["nmse"] - s["nmse_std"], s["nmse"] + s["nmse_std"],
                        color=C[k], alpha=.15, lw=0)
    ax.set_xlabel("反馈信道 SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.set_title("E2：一个共享模型 vs 三个专用模型（同架构，只改训练时的 SNR 分布）")
    ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("saved", path)


def draw_budget(snr, st, path="fig_budget.png"):
    """左：模型库相对共享模型的缺口（逐 SNR）；右：E4 蒸馏的折中。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax = axes[0]
    if all(k in st for k in FIX) and "learned_soft" in st:
        anchors = [FIX[k] for k in FIX]
        lib = np.stack([st[k]["nmse"] for k in FIX])
        row = {a: i for i, a in enumerate(anchors)}
        libc = np.array([lib[row[min(anchors, key=lambda a: abs(a - s))], i]
                         for i, s in enumerate(snr)])
        gap = st["learned_soft"]["nmse"] - libc
        cols = ["#c0392b" if g > 0.15 else "#9aa7b0" for g in gap]
        ax.bar(snr, gap, width=1.4, color=cols)
        ax.axhline(0, color="k", lw=.8)
        for i, g in enumerate(gap):
            if abs(g) > 0.15:
                ax.text(snr[i], g + .03, f"{g:+.2f}", ha="center", fontsize=8.5)
        ax.set_xlabel("反馈信道 SNR (dB)")
        ax.set_ylabel("模型库 − 共享（dB，正 = 模型库更好）")
        ax.set_title("(a) 3 倍存储换来多少：缺口只集中在低 SNR")
        ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    try:
        dd = json.load(open("results_distill.json", encoding="utf-8"))["test"]
        base = st["learned_soft"]["nmse"]
        i0 = int(np.argmin(np.abs(snr - 0)))
        i10 = int(np.argmin(np.abs(snr - 10)))
        xs, ys, labs = [], [], []
        for tag, seeds in dd.items():
            cur = np.array([v["nmse"] for v in seeds.values()]).mean(0)
            xs.append(cur[i10] - base[i10]); ys.append(base[i0] - cur[i0])
            labs.append(tag.replace("_lam", " λ=").replace("e3b_w3,1,1", "重加权 3:1:1"))
        ax.scatter(xs, ys, s=70, color="#00798c", zorder=3)
        for x, y, t in zip(xs, ys, labs):
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(8, -10), fontsize=8.5)
        if "learned_soft_f0" in st:
            spec = st["learned_soft"]["nmse"][i0] - st["learned_soft_f0"]["nmse"][i0]
            ax.axhline(spec, color="#c0392b", ls="--", lw=1.2)
            ax.text(0.02, spec + 0.02, f"专用 f0 在 0 dB 的优势 {spec:+.2f} dB",
                    color="#c0392b", fontsize=8.5)
            ax.set_ylim(-0.03, spec + 0.10)
        ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("10 dB 处损失（dB，向右变差）")
        ax.set_ylabel("0 dB 处收益（dB，向上更好）")
        ax.set_title("(b) 蒸馏/重加权：0 dB 处换回多少，10 dB 处赔掉多少")
        ax.grid(alpha=.25)
    except FileNotFoundError:
        ax.text(.5, .5, "还没有 results_distill.json", ha="center", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("saved", path)


def draw_rx(snr, path="fig_rx_v2.png"):
    """E5：冻结发送端（编码器 + 星座），只重训接收端。"""
    if not os.path.exists("results_rx.json"):
        print("skip fig_rx_v2：没有 results_rx.json")
        return
    d = json.load(open("results_rx.json", encoding="utf-8"))
    st = {}
    for k, seeds in d["test"].items():
        nm = np.array([v["nmse"] for v in seeds.values()])
        st[k] = (nm.mean(0), nm.std(0, ddof=1) if len(nm) > 1 else np.zeros_like(nm[0]))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    style = {"learned_hard": ("#d1495b", "o--", "硬判决接收端"),
             "learned_soft": ("#00798c", "o-", "软信息直通接收端"),
             "learned_soft_snr": ("#7b3fa0", "s-", "软信息 + 输入噪声水平")}
    for k, (c, ls, lab) in style.items():
        if k not in st:
            continue
        m, sd = st[k]
        ax.plot(snr, m, ls, ms=4, lw=2, color=c, label=lab)
        ax.fill_between(snr, m - sd, m + sd, color=c, alpha=.15, lw=0)
    ev = json.load(open(EVAL, encoding="utf-8"))["test"]["learned_soft"]
    full = np.array([v["nmse"] for v in ev.values()]).mean(0)
    ax.plot(snr, full, "k:", lw=2, label="端到端一起训练（参考）")
    ax.set_xlabel("反馈信道 SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.set_title("E5：同一发送端、同一观测，只换接收接口")
    ax.grid(alpha=.3); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("saved", path)


def _load_net(name, seed=0):
    net = Link(name, 64, 16, power_ref=True)
    net.load_state_dict(torch.load(os.path.join("ckpt_v2", f"{name}_s{seed}.pt"),
                                   map_location="cpu"))
    net.eval()
    return net


def draw_constellation(path="fig_constellation_v2.png"):
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for yi, (k, c) in enumerate([("fixed_hard", "#d1495b"), ("learned_hard", "#00798c")]):
        net = _load_net(k)
        pts = net.const.get_points() if hasattr(net.const, "get_points") else net.const
        pts = np.sort(pts.detach().numpy())
        ax.vlines(pts, yi - .18, yi + .18, color=c, lw=6, alpha=.85)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["固定 16-PAM", "学习到的 16-PAM"])
    ax.set_xlabel("符号幅度（已归一化到单位功率）")
    ax.set_xlim(-2.3, 2.3); ax.grid(alpha=.25, axis="x")
    ax.set_title("星座几何：学习到的点略有聚散，但收益只有 ~0.03 dB（见 E1）")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("saved", path)


def draw_recon(snr_db=10.0, path="fig_recon_v2.png"):
    xte = load_split("test")[:8]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0))
    h = torch.sqrt(xte[0, 0] ** 2 + xte[0, 1] ** 2).numpy()
    vmax = np.percentile(h, 99)
    axes[0].imshow(h, cmap="viridis", vmax=vmax)
    axes[0].set_title("真实 |H|", fontsize=10)
    for ax, k in zip(axes[1:], CORE):
        net = _load_net(k)
        with torch.no_grad():
            p = net(xte, snr_db)
        hp = torch.sqrt(p[0, 0] ** 2 + p[0, 1] ** 2).numpy()
        nm = float(nmse_db(p, xte))
        ax.imshow(hp, cmap="viridis", vmax=vmax)
        ax.set_title(f"{SHORT[k].split('（')[0]}\n{nm:.1f} dB", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if not os.path.exists(EVAL):
        raise SystemExit(f"{EVAL} 还没有，先跑 eval_v2.py")
    snr, st = load()
    draw_curve(snr, st)
    draw_q3(snr, st)
    draw_budget(snr, st)
    draw_rx(snr)
    draw_constellation()
    draw_recon()
