# -*- coding: utf-8 -*-
"""线性逆问题基线：直接把 y ∈ R^64 线性映射回 H ∈ R^2048。

如果"CSI 压缩+恢复是一个线性逆问题"成立，那么最优线性估计
（岭回归 / Wiener，闭式解）应当接近神经网络解码器。
本脚本只做这个对照，不训练任何网络。
"""
import json

import numpy as np
import torch

import quality_v5 as Q

SNRS = [0, 10, 20]
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]


def features(net, x):
    """analog 方案的前向：编码器 + 逐样本功率归一（无量化、无噪声）。"""
    with torch.no_grad():
        z = net.enc(x)
        z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
    return z


def flatten(h):
    return h.reshape(h.shape[0], -1).numpy()


def main():
    net = Q.build("analog", 0)
    xtr, xte = Q.load_split("train"), Q.load_split("test")
    Htr, Hte = flatten(xtr), flatten(xte)

    out = {}
    for snr in SNRS:
        # 训练集上取一次干净特征，再按该 SNR 加噪（配对）
        ztr = features(net, xtr)
        zte = features(net, xte)
        sigma = Q.sigma_of(net, snr)
        g = torch.Generator().manual_seed(1234)
        ytr = (ztr + sigma * torch.randn(ztr.shape, generator=g)).numpy()
        yte = (zte + sigma * torch.randn(zte.shape, generator=g)).numpy()

        # 标准化，岭回归闭式解
        mu, sd = ytr.mean(0), ytr.std(0) + 1e-8
        Ytr = (ytr - mu) / sd
        Yte = (yte - mu) / sd
        G = Ytr.T @ Ytr
        b = Ytr.T @ Htr

        best = None
        for lam in LAMBDAS:
            W = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
            pred = Yte @ W
            mse = ((pred - Hte) ** 2).sum(1) / (Hte ** 2).sum(1)
            nmse = float(np.mean(10 * np.log10(mse + 1e-12)))
            if best is None or nmse < best[1]:
                best = (lam, nmse)
        out[snr] = {"lam": best[0], "nmse_db": round(best[1], 3)}

    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open("results_linear_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
