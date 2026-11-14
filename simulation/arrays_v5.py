"""v5 逐样本数组与方差分解（供 analysis_v5.py 使用）。

A) 主数组：test 1000 个 H，每个 SNR 一组固定噪声，保存
   nmse, sgcs, Q1, Q2, O1, 以及重编码残差的三项分解（n², ‖tx−F(Ĥ)‖², 2⟨n,…⟩）
B) 方差分解：test 前 200 个 H，每个 SNR 采 20 次噪声，保存 L_ij（NMSE_dB）矩阵

用法：python arrays_v5.py --schemes learned_soft --seeds 0 1 2 --variants S0 S2a --out results_v5_arrays.npz
"""
import argparse
import json

import numpy as np
import torch

import quality_v5 as Q
from bn_v5 import recalibrate
from run_v5 import decision_error, transmit3

SNRS = (0, 10, 20)


@torch.no_grad()
def one(net, x, snr, noise_seed):
    tx, y, idx = transmit3(net, x, snr, noise_seed=noise_seed)
    sig = Q.sigma_of(net, snr)
    g = torch.Generator().manual_seed(noise_seed)
    n = sig * torch.randn(tx.shape, generator=g)
    xhat = Q.decode(net, y, snr)
    tx_hat = Q.forward_op(net, xhat)
    diff = tx - tx_hat
    return {
        "nmse": Q.nmse_db_per(xhat, x).numpy(),
        "sgcs": Q.sgcs_per(xhat, x).numpy(),
        "Q1": Q.q1_const_dist(net, y).numpy(),
        "Q2": Q.q2_reencode(net, y, xhat).numpy(),
        "O1": decision_error(net, y, idx).numpy(),
        "n2": (n ** 2).sum(1).numpy(),
        "rec2": (diff ** 2).sum(1).numpy(),
        "cross": (2 * (n * diff).sum(1)).numpy(),
    }


@torch.no_grad()
def var_matrix(net, x, snr, n_noise=20, base_seed=9000):
    """返回 (n_H, n_noise) 的 NMSE_dB 矩阵。"""
    cols = []
    for k in range(n_noise):
        _, y, _ = transmit3(net, x, snr, noise_seed=base_seed + k)
        xhat = Q.decode(net, y, snr)
        cols.append(Q.nmse_db_per(xhat, x).numpy())
    return np.stack(cols, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["learned_soft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["S0", "S2a"])
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--n-var", type=int, default=200)
    ap.add_argument("--n-noise", type=int, default=20)
    ap.add_argument("--out", default="results_v5_arrays.npz")
    a = ap.parse_args()

    xtr, xte = Q.load_split("train"), Q.load_split("test")
    store, varmat = {}, {}
    for scheme in a.schemes:
        for seed in a.seeds:
            for var in a.variants:
                key = f"{scheme}_s{seed}_{var}"
                for snr in SNRS:
                    net = Q.build(scheme, seed, xtr=xtr)
                    if var == "S2a":
                        recalibrate(net, xtr, snr=snr, mixed=False)
                    r = one(net, xte, snr, a.noise_seed)
                    for k, v in r.items():
                        store[f"{key}|{snr}|{k}"] = v.astype(np.float32)
                    if var == "S0":
                        varmat[f"{key}|{snr}"] = var_matrix(
                            net, xte[:a.n_var], snr, a.n_noise).astype(np.float32)
                    print(f"  {key} SNR={snr:2d} nmse={r['nmse'].mean():+.3f}", flush=True)

    # 方差分解（每个 SNR × 种子）
    dec = {}
    for k, L in varmat.items():
        w = L.var(axis=1, ddof=1).mean()          # Ŵ
        vm = L.mean(axis=1).var(ddof=1)           # Var_H(mean) = B + W/M
        b = vm - w / L.shape[1]
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, L.shape[0], L.shape[0])
            Lb = L[idx]
            wb = Lb.var(axis=1, ddof=1).mean()
            bb = Lb.mean(axis=1).var(ddof=1) - wb / L.shape[1]
            boots.append(wb / (wb + max(bb, -1e9)))
        dec[k] = {"W": float(w), "B_hat": float(b), "frac_within": float(w / (w + b))
                  if (w + b) > 0 else float("nan"),
                  "ci_lo": float(np.percentile(boots, 2.5)),
                  "ci_hi": float(np.percentile(boots, 97.5))}
        print(f"  var {k}: W={w:.3f} B̂={b:.3f} W/(W+B̂)={dec[k]['frac_within']:.3f} "
              f"[{dec[k]['ci_lo']:.3f}, {dec[k]['ci_hi']:.3f}]", flush=True)

    np.savez_compressed(a.out, **store)
    with open(a.out.replace(".npz", "_var.json"), "w", encoding="utf-8") as f:
        json.dump(dec, f, ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
