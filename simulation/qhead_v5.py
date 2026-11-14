"""v5 Q3：学习型质量头（对照 3DC-SC 一类"接收端预测重建质量"的思路）。

输入 [y, SNR]，输出预测的 NMSE_dB；2 隐藏层 ×128 ReLU；Adam 1e-3；MSE；
训练集 H × SNR{0,10,20} × 每 H 4 次噪声；验证集早停；测试集只报告。

用法：
  python qhead_v5.py --schemes learned_soft --seeds 0 1 2 --variants S0 S2a --out results_v5_qhead.json
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import quality_v5 as Q
from bn_v5 import recalibrate
from run_v5 import transmit3

SNRS = (0, 10, 20)


class Head(nn.Module):
    def __init__(self, ns=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ns + 1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, y, snr):
        s = torch.full((y.shape[0], 1), float(snr)) if not torch.is_tensor(snr) \
            else snr.view(-1, 1).float()
        return self.net(torch.cat([y, s], 1)).squeeze(-1)


@torch.no_grad()
def build_cache(net, x, snrs=SNRS, n_noise=4, base_seed=5000, bs=512):
    """返回 (Y, S, T)：y 向量、SNR、真实 NMSE_dB。"""
    Ys, Ss, Ts = [], [], []
    for si, snr in enumerate(snrs):
        for k in range(n_noise):
            for i in range(0, len(x), bs):
                xb = x[i:i + bs]
                _, y, _ = transmit3(net, xb, snr, noise_seed=base_seed + 100 * si + k)
                xhat = Q.decode(net, y, snr)
                Ys.append(y)
                Ss.append(torch.full((len(xb),), float(snr)))
                Ts.append(Q.nmse_db_per(xhat, xb))
    return torch.cat(Ys), torch.cat(Ss), torch.cat(Ts)


def train_head(Y, S, T, Yv, Sv, Tv, epochs=60, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Y.mean(0, keepdim=True), Y.std(0, keepdim=True) + 1e-6
    head = Head(Y.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    n = len(Y)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        head.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = head((Y[idx] - mu) / sd, S[idx])
            loss = ((pred - T[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vp = head((Yv - mu) / sd, Sv)
            vl = float(((vp - Tv) ** 2).mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, (mu, sd), best


@torch.no_grad()
def predict(head, ms, Y, S):
    mu, sd = ms
    return head((Y - mu) / sd, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["learned_soft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["S0", "S2a"])
    ap.add_argument("--out", default="results_v5_qhead.json")
    a = ap.parse_args()

    xtr, xva, xte = Q.load_split("train"), Q.load_split("val"), Q.load_split("test")
    res = {}
    for scheme in a.schemes:
        for seed in a.seeds:
            for var in a.variants:
                key = f"{scheme}_s{seed}_{var}"
                t0 = time.time()
                cache = {}
                for split, x in (("train", xtr), ("val", xva), ("test", xte)):
                    net = Q.build(scheme, seed, xtr=xtr)
                    if var == "S2a":
                        net = recalibrate(net, xtr, snr=SNRS[0], mixed=False)
                    cache[split] = (net, x)
                net, x = cache["train"]
                Y, S, T = build_cache(net, x, n_noise=4, base_seed=5000)
                net, x = cache["val"]
                Yv, Sv, Tv = build_cache(net, x, n_noise=2, base_seed=7000)
                head, ms, vloss = train_head(Y, S, T, Yv, Sv, Tv)

                row = {"val_mse": vloss, "dur_s": round(time.time() - t0, 1)}
                for snr in SNRS:
                    net, x = cache["test"]
                    # 测试集每个 SNR 固定一组噪声（与主实验不同种子，避免拟合到同一批噪声）
                    _, y, _ = transmit3(net, x, snr, noise_seed=1234)
                    xhat = Q.decode(net, y, snr)
                    nm = Q.nmse_db_per(xhat, x).numpy()
                    pred = predict(head, ms, y, torch.full((len(x),), float(snr))).numpy()
                    s = Q.summarize(pred, nm, "Q3")
                    s["mae"] = float(np.abs(pred - nm).mean())
                    s["calibration"] = Q.mean_abs_calibration(pred, nm)
                    row[str(snr)] = s
                    print(f"{key} SNR={snr:2d}  AUROC={s['auroc']:.3f} rho={s['spearman']:+.3f} "
                          f"MAE={s['mae']:.2f} dB  ECE={s['calibration']:.2f}  ({row['dur_s']}s)",
                          flush=True)
                res[key] = row
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
