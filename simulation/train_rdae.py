"""训练 / 评估 RD-JSCC AE（`model_rdae.LinkRDAE`）—— v6 §2 的外部效度对照。

口径与 v5 一致（同一 data/*.npz 划分、同一 SNR/Ns 定义、同一指标），
唯一差别：编解码器换成 RD-JSCC 的 AFCsiNetPlus（0.55 M 参数，带 AF 注意力与 5 个残差块）。

  python train_rdae.py --smoke                                  # 1 epoch 验形状
  python train_rdae.py --seeds 0 1 2 --epochs 40                # 主实验（逐样本抽 SNR）
  python train_rdae.py --seeds 0 --epochs 40 --snr-mode batch   # 每 batch 一个 SNR
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model_rdae import LinkRDAE
from model_v4 import from_symbols, nmse_db, normalize_power, pair_mask, qam_map_ste, sgcs, to_symbols
from quality_v5 import auroc, bad_label, spearman

SNR_GRID = list(range(0, 21, 2))
NS_LIST = (16, 32, 64)
Q2_SNRS = (0, 10, 20)


SCALE = 3.0          # 逐样本 RMS 归一化后 ÷3，落在 ±1 附近，便于 tanh 输出


def load01(split: str, root: str = "data"):
    """与 v5 完全同口径：`quality_v5.load_split`（逐样本幂均归一化），再除以固定常数 3。

    固定常数不影响 NMSE（分子分母同比缩放），只把取值范围挪进解码器的 tanh 量程。
    """
    from quality_v5 import load_split
    return load_split(split, root=root) / SCALE


def train(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(args.device)
    xtr = load01("train").to(dev)
    net = LinkRDAE(cond=False).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    lossf = nn.MSELoss()

    n, steps = xtr.shape[0], max(1, xtr.shape[0] // args.bs)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for it in range(steps):
            xb = xtr[perm[it * args.bs:(it + 1) * args.bs]]
            b = xb.shape[0]
            ns = torch.full((b,), int(np.random.choice(NS_LIST)), device=dev)
            if args.snr_mode == "batch":
                snr = torch.full((b,), float(np.random.uniform(0, 20)), device=dev)
            else:
                snr = torch.empty(b, device=dev).uniform_(0, 20)
            loss = lossf(net(xb, snr, ns), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        if args.smoke or ep % max(1, args.epochs // 5) == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:02d} mse={tot / steps:.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.calibrate(xtr, NS_LIST)
    return net


@torch.no_grad()
def evaluate(net, x, ns):
    dev = next(net.parameters()).device
    x = x.to(dev)
    nm, sc = [], []
    for snr in SNR_GRID:
        out = net(x, float(snr), ns)
        nm.append(float(nmse_db(out, x)))
        sc.append(float(sgcs(out, x)))
    return nm, sc


@torch.no_grad()
def q2_auroc(net, x, ns, snr, seed=1234, n=400):
    """Q2：**委托** model_rdae.q2_scores（同一条发送路径：normalize_power → qam_map_ste → add_noise → pair_mask）。

    早先这里自己拼了一段：`sigma * randn(tx.shape)` —— 每实维都用 σ（漏 ÷√2，等效 SNR 高 3 dB），
    且未屏蔽未发送位置。已删除，统一到 q2_scores，避免两个入口口径不一致。
    """
    from model_rdae import q2_scores
    return q2_scores(net, x, ns, snr, seed=seed, n=n)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seeds", type=int, nargs="+", default=[0])
    pa.add_argument("--epochs", type=int, default=40)
    pa.add_argument("--bs", type=int, default=128)
    pa.add_argument("--lr", type=float, default=2e-3)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--snr-mode", choices=["sample", "batch"], default="sample")
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--out", default="results_rdae.json")
    a = pa.parse_args()
    if a.smoke:
        a.epochs = 1
        a.seeds = a.seeds[:1]

    res = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    xte = load01("test")
    for seed in a.seeds:
        tag = f"rdae_{a.snr_mode}_s{seed}"
        print(f"[{tag}] train", flush=True)
        net = train(a, seed)
        net.eval()
        rec = {"snr": SNR_GRID, "nmse": {}, "sgcs": {}, "q2": {}}
        for ns in NS_LIST:
            nm, sc = evaluate(net, xte, ns)
            rec["nmse"][str(ns)], rec["sgcs"][str(ns)] = nm, sc
            print(f"  ns={ns:2d} " + " ".join(f"{v:+.2f}" for v in nm), flush=True)
        if not a.smoke:
            for snr in Q2_SNRS:
                rec["q2"][str(snr)] = q2_auroc(net, xte, 64, snr)
            print("  q2(AUROC, rho, nmse):", {k: [round(v, 3) for v in val] for k, val in rec["q2"].items()},
                  flush=True)
        res[tag] = rec
        os.makedirs("ckpt_rdae", exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()},
                   f"ckpt_rdae/{tag}.pt")
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
