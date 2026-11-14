"""v5 / Q2：逐样本可靠性估计的评估口径。

只做「给定模型 + 一批 H + 固定噪声 → 各指标分数 + 指标评价」，
不负责批量跑实验（那是 run_v5.py 的事）。

口径要点（与 00_后续执行_v5.md 一致）：
- 所有分数「越大越可疑」；Spearman = corr(score, NMSE_dB)。
- 风险 = mean(NMSE_dB)（先取 dB 再平均）。
- 同一 SNR 下，各指标 / 各模型共享同一批 H 与同一组标准高斯噪声。
- 重编码算子 F = 编码器 + 逐样本功率归一化 + 星座映射，必须与链路发送端逐元素相等。
"""
import json
import math
import os

import numpy as np
import torch

from model_jscm import Link, LearnedConstellation, calibrate, nmse_db, sgcs

SNRS = (0, 10, 20)


# ------------------------------------------------------------------ 数据 / 模型
def load_split(name, root="data"):
    """与 run_experiments.load_split 严格同口径（逐样本幂均归一化）。"""
    d = np.load(os.path.join(root, f"{name}.npz"))["x"].astype(np.float32)
    d = d / np.sqrt((d ** 2).mean(axis=(1, 2, 3), keepdims=True))
    return torch.from_numpy(d)


def build(scheme, seed, ns=64, m=16, ckpt_dir="ckpt_v2", xtr=None):
    """载入一个已训练模型，切 eval、标定 p_cal（训练集），返回 net。"""
    net = Link(scheme, ns, m)
    sd = torch.load(os.path.join(ckpt_dir, f"{scheme}_s{seed}.pt"), map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    if xtr is None:
        xtr = load_split("train")
    calibrate(net, xtr)
    net.eval()
    return net


# ------------------------------------------------------------------ 链路前向
def forward_op(net, x):
    """F(H)：编码器 + 逐样本功率归一化 + 星座映射（不加噪）。"""
    z = net.enc(x)
    z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
    tx, _ = net.modulate(z)
    return tx


def sigma_of(net, snr):
    """与 Link.forward 完全一致的噪声标准差。"""
    rel = 10 ** (-float(snr) / 20.0)
    p_ref = net.p_cal if net.p_cal is not None else net.last_tx_power
    return rel * float(p_ref) ** 0.5


def transmit(net, x, snr, noise=None, seed=1234):
    """发送：返回 (tx, y)。noise 为标准高斯（同形），便于各指标配对共享。"""
    tx = forward_op(net, x)
    sig = sigma_of(net, snr)
    if noise is None:
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(tx.shape, generator=g)
    y = tx + sig * noise
    return tx, y


def receive_op(net, y, snr):
    """链路接收端：先做硬判决（若该方案如此），再按需拼 SNR，最后解码。"""
    y = net.receive(y, None)
    if net.scheme == "learned_soft_snr":
        rel = 10 ** (-float(snr) / 20.0)
        s_t = torch.full((y.shape[0], 1), rel, dtype=y.dtype)
        y = torch.cat([y, s_t], dim=1)
    return y


@torch.no_grad()
def decode(net, y, snr):
    return net.dec(receive_op(net, y, snr))


def nmse_db_per(pred, target):
    """逐样本 NMSE（dB）。"""
    num = (pred - target).pow(2).sum((1, 2, 3))
    den = target.pow(2).sum((1, 2, 3))
    return 10 * torch.log10(num / den + 1e-12)


def sgcs_per(pred, target):
    """逐样本平方 SGCS：逐子载波 |h^H ĥ| / (‖h‖‖ĥ‖)，平方后对子载波取平均。

    与 model_jscm.sgcs 同一定义（那边再对样本取均值）。
    """
    hp = torch.complex(pred[:, 0], pred[:, 1])      # (B, Nt, Nc)
    ht = torch.complex(target[:, 0], target[:, 1])
    num = (ht.conj() * hp).sum(1).abs()             # (B, Nc)
    den = ht.abs().pow(2).sum(1).sqrt() * hp.abs().pow(2).sum(1).sqrt() + 1e-12
    return (num / den).pow(2).mean(1)


# ------------------------------------------------------------------ 指标（分数）
@torch.no_grad()
def q0_snr(y, snr):
    """Q0：只看 SNR —— 固定 SNR 下是常数，AUROC 应为 0.5。"""
    return torch.full((y.shape[0],), float(snr))


def _pts(net):
    c = net.const
    return c.get_points().detach() if isinstance(c, LearnedConstellation) else c.detach()


@torch.no_grad()
def q1_const_dist(net, y):
    """Q1：到最近星座点的平均平方距离（越大越可疑）。"""
    pts = _pts(net).view(1, 1, -1)
    d = (y.unsqueeze(-1) - pts).pow(2).min(-1).values
    return d.mean(1)


@torch.no_grad()
def q2_reencode(net, y, xhat):
    """Q2：重编码一致性 ‖y − F(Ĥ)‖² / Ns（越大越可疑）。"""
    tx_hat = forward_op(net, xhat)
    return (y - tx_hat).pow(2).mean(1)


# ------------------------------------------------------------------ 指标评价
def spearman(a, b):
    """秩相关；无并列时的精确 Spearman。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = _rank(a)
    rb = _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _rank(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
    # 并列取平均秩
    for val in np.unique(v):
        idx = np.where(v == val)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def auroc(score, label):
    """score 越大越可疑；label 为 1 表示坏样本。用 Mann-Whitney U。"""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank(score)
    u = r[label].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auprc(score, label):
    """平均精度（AP）：按分数降序，逐点计算 precision 的加权平均。"""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(np.float64)
    order = np.argsort(-score, kind="mergesort")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / max(y.sum(), 1e-12)
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float((prec * drec).sum())


def bad_label(nmse, frac=0.2):
    """每个 SNR 下 NMSE 最差的 frac 为坏样本。"""
    nmse = np.asarray(nmse, dtype=np.float64)
    k = max(1, int(round(len(nmse) * frac)))
    thr = np.sort(nmse)[-k]
    lab = nmse >= thr
    lab[np.argsort(nmse)[-k:]] = True          # 处理并列
    return lab


COVERS = (1.0, 0.95, 0.90, 0.80, 0.70)


def risk_curve(score, loss, covers=COVERS):
    """按分数从可信（小）到可疑（大）排序，保留前 c 的比例，报告其平均损失。"""
    score = np.asarray(score, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    out = {}
    for c in covers:
        k = max(1, int(round(len(score) * c)))
        out[c] = float(loss[order[:k]].mean())
    return out


def ideal_curve(loss, covers=COVERS):
    """理想选择：按真实损失从好到坏保留（需要额外信息，只作参照）。

    注意方向：risk_curve 保留「分数最小」的样本，所以这里必须把 loss 本身当分数，
    即保留损失最小的。早先误传 -loss，实际保留的是最差样本。
    """
    return risk_curve(np.asarray(loss, dtype=np.float64), loss, covers)


def random_curve(loss, covers=COVERS):
    loss = np.asarray(loss, dtype=np.float64)
    return {c: float(loss.mean()) for c in covers}


def partial_aurc(curve, lo=0.70, hi=1.00):
    """覆盖率 [lo, hi] 上的部分 AURC（梯形积分 / 区间长度）。"""
    cs = sorted([c for c in curve if lo - 1e-9 <= c <= hi + 1e-9])
    xs = np.array(cs, dtype=np.float64)
    ys = np.array([curve[c] for c in cs], dtype=np.float64)
    area = float(np.trapezoid(ys, xs) if hasattr(np, 'trapezoid') else np.trapz(ys, xs))
    return area / (hi - lo)


def mean_abs_calibration(pred, actual, bins=10):
    """质量头校准：按预测值分箱，|均值(预测) − 均值(实际)| 的箱内加权平均。"""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    edges = np.quantile(pred, np.linspace(0, 1, bins + 1))
    tot, n = 0.0, 0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pred >= lo) & (pred <= hi) if i == bins - 1 else (pred >= lo) & (pred < hi)
        if m.sum() == 0:
            continue
        tot += abs(pred[m].mean() - actual[m].mean()) * int(m.sum())
        n += int(m.sum())
    return float(tot / max(n, 1))


# ------------------------------------------------------------------ 小工具
def summarize(score, nmse, name=""):
    """一个 SNR 下的全部数字。"""
    lab = bad_label(nmse)
    cur = risk_curve(score, nmse)
    return {
        "name": name,
        "auroc": auroc(score, lab),
        "auprc": auprc(score, lab),
        "spearman": spearman(score, nmse),
        "risk": {f"{c:.2f}": v for c, v in cur.items()},
        "aurc70_100": partial_aurc(cur),
    }


if __name__ == "__main__":
    # 极简冒烟：一个模型、一个 SNR、200 个样本，打印各指标与计时
    import time

    t0 = time.time()
    xtr, xte = load_split("train"), load_split("test")[:200]
    net = build("learned_soft", 0, xtr=xtr)
    print(f"[smoke] p_cal={net.p_cal:.4f}  load={time.time() - t0:.1f}s")

    for snr in SNRS:
        t1 = time.time()
        with torch.no_grad():
            tx, y = transmit(net, xte, snr, seed=1234)
            xhat = decode(net, y, snr)
        nm = nmse_db_per(xhat, xte).numpy()
        scores = {
            "Q0_snr": q0_snr(y, snr).numpy(),
            "Q1_const": q1_const_dist(net, y).numpy(),
            "Q2_reenc": q2_reencode(net, y, xhat).numpy(),
        }
        print(f"\n[smoke] SNR={snr} dB  NMSE={nm.mean():+.2f}±{nm.std():.2f} dB  "
              f"({time.time() - t1:.1f}s)")
        for k, v in scores.items():
            s = summarize(v, nm, k)
            print(f"  {k:9s} AUROC={s['auroc']:.3f} AUPRC={s['auprc']:.3f} "
                  f"rho={s['spearman']:+.3f} AURC70-100={s['aurc70_100']:+.3f}")
