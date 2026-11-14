# -*- coding: utf-8 -*-
"""v4 数据层：COST2100 官方 .mat 读取 + 合成兜底，统一输出 .npz（复数 32×32）。

约定（与 model_v4.py 一致）
---------------------------
* 数据形状：(N, 32, 32) 复数，第一维=天线维（angular），第二维=时延维（delay）。
* 归一化：逐样本复数 RMS 归一（NMSE 对全局缩放不变，可绕开 [0,1] 之类的口径分歧）。
* `--offset`：读原始 .mat 后先减去的常数。CsiNet 发布版把复系数整体平移到 [0,1] 附近，
  常见做法是减 0.5；这里默认 0.5，但读入后会打印原始统计量，可据此改正。
* 场景：文件名含 `in` / `out` 分别视为室内 / 室外；两套都不存在时报错而不是猜。

用法
----
    # 只看文件里有什么、统计量如何
    python data_v4.py --src assets/cost2100 --report

    # 真正导出（室内）
    python data_v4.py --src assets/cost2100 --scenario in --out data_v4

    # 没有真实数据时：合成 6 万样本（v4 §2 允许，但必须写进文档）
    python data_v4.py --src synthetic --n 60000 --out data_v4_sync
"""
import argparse
import glob
import os
import sys

import numpy as np

FILES = {
    "in": ["DATA_Htrainin.mat", "DATA_Hvalidin.mat", "DATA_Htestin.mat"],
    "out": ["DATA_Htrainout.mat", "DATA_Hvalidout.mat", "DATA_Htestout.mat"],
}
SPLITS = ["train", "val", "test"]


# --------------------------------------------------------------------- 读 .mat
def _loadmat(path):
    from scipy.io import loadmat
    m = loadmat(path)
    cand = [k for k in m if not k.startswith("__")]
    if "HT" in m:
        arr = m["HT"]
    elif len(cand) == 1:
        arr = m[cand[0]]
    else:
        raise KeyError(f"{path}: 找不到 HT，候选键 {cand}")
    return np.asarray(arr)


def to_complex(arr):
    """(N,2,Na,Nd) 或 (N,Na,Nd) -> (N,Na,Nd) complex128。"""
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        return a.astype(np.complex128)
    if a.ndim == 4 and a.shape[1] == 2:
        return a[:, 0].astype(np.float64) + 1j * a[:, 1].astype(np.float64)
    if a.ndim == 3:
        return a.astype(np.complex128)
    raise ValueError(f"形状不认识：{a.shape}")


def rms_normalize(h, eps=1e-12):
    """逐样本复数 RMS 归一。"""
    p = np.sqrt((np.abs(h) ** 2).mean(axis=(1, 2), keepdims=True))
    return (h / (p + eps)).astype(np.complex64)


# ------------------------------------------------------------------ 找 & 分组
def find_scenario(src, scenario=None):
    """返回 {split: path}。只认 FILES 里列出的名字，避免误读别的 .mat。"""
    got = {}
    for sc in (["in", "out"] if scenario is None else [scenario]):
        d = {}
        for i, split in enumerate(SPLITS):
            p = os.path.join(src, FILES[sc][i])
            if os.path.isfile(p):
                d[split] = p
        if len(d) == 3:
            got[sc] = d
    if not got:
        allm = sorted(glob.glob(os.path.join(src, "**", "*.mat"), recursive=True))
        head = "\n  ".join(allm[:12]) if allm else "(目录里没有 .mat)"
        raise FileNotFoundError(
            f"{src} 下没找到完整的 train/val/test 三件套（{FILES['in']} 或 out 版）。\n"
            f"目录里的 .mat：\n  {head}"
        )
    return got


def report(src):
    got = find_scenario(src, None) if _has_any(src) else {}
    if not got:
        print("没有按标准命名找到三件套，逐个列出统计量：")
        for p in sorted(glob.glob(os.path.join(src, "**", "*.mat"), recursive=True))[:12]:
            try:
                a = _loadmat(p)
            except Exception as e:                                   # noqa: BLE001
                print(f"  {os.path.basename(p)}: 读失败 {e}")
                continue
            print(f"  {os.path.basename(p)}: dtype={a.dtype} shape={a.shape} "
                  f"min={a.real.min():.3f} max={a.real.max():.3f} "
                  f"mean={a.real.mean():.3f} rms={np.sqrt((a.real ** 2).mean()):.3f}")
        return
    for sc, d in got.items():
        print(f"[{sc}]")
        for split, p in d.items():
            a = _loadmat(p)
            h = to_complex(a)
            print(f"  {split:5s} {os.path.basename(p):24s} raw{a.shape}  "
                  f"min={a.real.min():+.3f} max={a.real.max():+.3f} "
                  f"mean={a.real.mean():+.3f}  |H|rms={np.sqrt((np.abs(h) ** 2).mean()):.4f}")


def _has_any(src):
    return bool(glob.glob(os.path.join(src, "**", "*.mat"), recursive=True))


# --------------------------------------------------------------------- 合成兜底
def make_synthetic(n, seed=0, n_path=6, nt=32, nd=32):
    """角-时延域稀疏合成信道：每样本若干多径，角度与时延各占一格，复高斯系数。

    只保证「结构像 CSI（角度-时延稀疏、复高斯、逐样本功率有起伏）」，
    不承诺与 COST2100 的统计一致 —— 文档里必须这么写。
    """
    rng = np.random.default_rng(seed)
    h = np.zeros((n, nt, nd), dtype=np.complex128)
    for i in range(n):
        k = rng.integers(1, n_path + 1)
        aa = rng.integers(0, nt, size=k)
        dd = rng.integers(0, nd, size=k)
        g = (rng.normal(size=k) + 1j * rng.normal(size=k)) / np.sqrt(2)
        g = g * (rng.random(k) ** 1.5 + 0.1)          # 多径幅度有起伏
        h[i, aa, dd] += g
    # 轻微角度扩散：沿天线维做 3 点卷积，避免过于理想
    ker = np.array([0.25, 1.0, 0.25])
    h = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, h)
    return rms_normalize(h)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/cost2100",
                    help="COST2100 .mat 所在目录，或字面量 synthetic")
    ap.add_argument("--scenario", default=None, choices=["in", "out"])
    ap.add_argument("--offset", type=float, default=0.5, help="读 raw 后先减去的常数")
    ap.add_argument("--n", type=int, default=60000, help="合成样本总数")
    ap.add_argument("--out", default="data_v4")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.report:
        report(a.src)
        return

    os.makedirs(a.out, exist_ok=True)
    if a.src == "synthetic":
        n_tr, n_va, n_te = int(a.n * 0.8), int(a.n * 0.1), int(a.n * 0.1)
        H = make_synthetic(a.n, seed=a.seed)
        parts = {"train": H[:n_tr], "val": H[n_tr:n_tr + n_va], "test": H[n_tr + n_va:]}
        note = f"synthetic n={a.n} seed={a.seed}"
    else:
        got = find_scenario(a.src, a.scenario)
        sc = list(got)[0]
        parts = {}
        for split, p in got[sc].items():
            raw = _loadmat(p)
            h = to_complex(raw)
            h = h - a.offset
            parts[split] = rms_normalize(h)
            print(f"{sc}/{split}: raw{raw.shape} -> {parts[split].shape} complex64")
        note = f"cost2100-{sc}"
        if len(got) > 1:
            print(f"注意：同时存在 {list(got)}，本次只导出 {sc}（用 --scenario 指定另一个）")

    for split in SPLITS:
        h = parts[split]
        f = os.path.join(a.out, f"{split}.npz")
        np.savez_compressed(f, H=h, note=np.array(note))
        mb = os.path.getsize(f) / 1e6
        print(f"saved {f}  shape={h.shape} {h.dtype} {mb:.1f} MB  ({note})")

    # 一致性自检：三种 split 的样本数、功率
    for split in SPLITS:
        h = parts[split]
        p = (np.abs(h) ** 2).mean(axis=(1, 2))
        print(f"  {split:5s} n={len(h):6d}  E|H|^2={p.mean():.4f}  "
              f"per-sample min/max={p.min():.3f}/{p.max():.3f}")
    if parts["train"].shape[1:] != (32, 32):
        print(f"警告：形状不是 32×32 而是 {parts['train'].shape[1:]}，"
              f"若来自真实数据请确认天线维/时延维顺序", file=sys.stderr)


if __name__ == "__main__":
    main()
