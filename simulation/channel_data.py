"""合成大规模 MIMO 信道数据（天线-时延多径后沿时延维 DFT；不保证 COST2100 统计）。

为什么不用 COST2100 原始数据：一周周期内下载/预处理成本高，且本项目的结论
依赖的是"角-时延域稀疏 + 频域相关"这一结构，而不是具体数据集。自造数据可以
完全复现，也便于答辩时说明假设。

信道模型:
    H[n_t, n_c] = sum_k  g_k * a(theta_k) * exp(-j 2 pi n_c tau_k / Nc)
    a(theta)   = exp(-j pi n sin(theta)) / sqrt(Nt)   (ULA 阵列响应)

用法:
    python channel_data.py            # 生成并缓存到 data/*.npz
"""
import os
import numpy as np

NT, NC = 32, 32           # 基站天线数、子带数
N_TRAIN, N_VAL, N_TEST = 6000, 1000, 1000


def _array_response(theta, nt):
    n = np.arange(nt)
    return np.exp(-1j * np.pi * n * np.sin(theta)) / np.sqrt(nt)


def gen_one(rng, nt=NT, nc=NC, n_cluster=4, n_ray=4, decay=1.5):
    """单样本：生成天线-时延域矩阵，再沿时延维变换到空-频域。"""
    h_ad = np.zeros((nt, nc), dtype=np.complex128)
    for _ in range(n_cluster):
        theta_c = rng.uniform(-np.pi / 3, np.pi / 3)      # 簇中心角
        d_c = rng.integers(0, 4)                          # 簇中心时延抽头
        for _ in range(n_ray):
            theta = theta_c + rng.laplace(0.0, 0.03)      # 簇内角度扩展
            a = _array_response(theta, nt)
            tau = int(d_c + rng.integers(0, 2)) % nc
            g = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            h_ad[:, tau] += g * a * np.exp(-tau / decay)
    # 噪声底：保证不是严格稀疏，贴近真实测量
    h_ad += 0.01 * (rng.standard_normal(h_ad.shape) + 1j * rng.standard_normal(h_ad.shape))
    # 时延域 -> 频域（DFT 矩阵）
    w = np.exp(-2j * np.pi * np.outer(np.arange(nc), np.arange(nc)) / nc)
    h = h_ad @ w
    h = h / np.mean(np.abs(h_ad))                          # CsiNet 式归一化
    return np.stack([h.real, h.imag], axis=0).astype(np.float32)


def gen_dataset(n, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2, NT, NC), dtype=np.float16)
    for i in range(n):
        out[i] = gen_one(rng)
    return out


def main():
    os.makedirs("data", exist_ok=True)
    specs = [("train", N_TRAIN, 2024), ("val", N_VAL, 7), ("test", N_TEST, 11)]
    for name, n, seed in specs:
        path = os.path.join("data", f"{name}.npz")
        if os.path.exists(path):
            print(f"skip {path}")
            continue
        arr = gen_dataset(n, seed)
        np.savez_compressed(path, x=arr)
        print(f"{path}: {arr.shape} {arr.nbytes/1e6:.0f} MB(float16)")


if __name__ == "__main__":
    main()
