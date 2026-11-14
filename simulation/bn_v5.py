"""v5 §4.1(B1 修正)：BatchNorm 统计量对照 S0 / S1 / S2a。

S0  = 训练得到的原始统计量（不再重统计）
S1  = 混合 SNR 重新统计（40 个 batch，每个 batch 随机 SNR ~ U(0,20)）
S2a = 按该 SNR 重新统计（40 个 batch，全部是该 SNR）

三者校准用的 batch 数与样本数完全一致（40 × 128 = 5120 个样本），
差别只在「统计时喂什么 SNR 的噪声」。S2a 就是主实验的"修正后共享版本"。

用法：
  python bn_v5.py --schemes learned_soft --seeds 0 1 2 --out results_v5_bn.json
"""
import argparse
import json

import numpy as np
import torch

import quality_v5 as Q

SNRS = (0, 10, 20)


def dec_bns(net):
    return [m for m in net.dec.modules()
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))]


@torch.no_grad()
def recalibrate(net, xtr, snr=None, n_batch=40, bs=128, seed=17, mixed=False):
    """只更新解码器 BN 的 running stats；其余模块保持 eval，权重不动。"""
    bns = dec_bns(net)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None            # 累积平均，而不是滑动平均
        m.train()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(xtr), generator=g)
    for i in range(n_batch):
        xb = xtr[perm[i * bs:(i + 1) * bs]]
        s = float(torch.empty(1).uniform_(0, 20).item()) if mixed else float(snr)
        net(xb, s)                   # 只为了让 BN 看到激活
    for m in bns:
        m.eval()
    net.eval()
    return net


@torch.no_grad()
def evaluate(net, x, snr, noise_seed=1234):
    _, y = Q.transmit(net, x, snr, seed=noise_seed)
    xhat = Q.decode(net, y, snr)
    nm = Q.nmse_db_per(xhat, x)
    sg = Q.sgcs_per(xhat, x)
    return {"nmse_mean": float(nm.mean()), "nmse_std": float(nm.std()),
            "sgcs_mean": float(sg.mean()),
            "nmse": nm.numpy().tolist(), "sgcs": sg.numpy().tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["learned_soft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="ckpt_v2")
    ap.add_argument("--out", default="results_v5_bn.json")
    a = ap.parse_args()

    xtr, x = Q.load_split("train"), Q.load_split(a.split)
    res = {}
    for scheme in a.schemes:
        for seed in a.seeds:
            key = f"{scheme}_s{seed}"
            res[key] = {}
            for snr in SNRS:
                row = {}
                net = Q.build(scheme, seed, ckpt_dir=a.ckpt, xtr=xtr)
                row["S0"] = evaluate(net, x, snr)

                net = Q.build(scheme, seed, ckpt_dir=a.ckpt, xtr=xtr)
                recalibrate(net, xtr, mixed=True)
                row["S1"] = evaluate(net, x, snr)

                net = Q.build(scheme, seed, ckpt_dir=a.ckpt, xtr=xtr)
                recalibrate(net, xtr, snr=snr, mixed=False)
                row["S2a"] = evaluate(net, x, snr)
                res[key][str(snr)] = row
                print(f"{key} SNR={snr:2d} dB  "
                      f"S0={row['S0']['nmse_mean']:+.3f}  "
                      f"S1={row['S1']['nmse_mean']:+.3f}  "
                      f"S2a={row['S2a']['nmse_mean']:+.3f}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved", a.out)

    # 汇总：按 SNR 平均三个种子
    print("\n均值（3 种子）:")
    for snr in SNRS:
        vals = {v: np.mean([res[k][str(snr)][v]["nmse_mean"] for k in res])
                for v in ("S0", "S1", "S2a")}
        print(f"  SNR={snr:2d} dB  S0={vals['S0']:+.3f}  S1={vals['S1']:+.3f}  "
              f"S2a={vals['S2a']:+.3f}  （S2a−S0 = {vals['S2a'] - vals['S0']:+.3f} dB）")


if __name__ == "__main__":
    main()
