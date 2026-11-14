# -*- coding: utf-8 -*-
"""UE 估计误差敏感性（v4：方差由 CNR 决定，并含"误差感知训练"对照）。

建模（按文献写法）：
    H_est = H_true + E,  E 每个元素 i.i.d. CN(0, σ_e²),  CNR = 10·log10(P_elem / σ_e²)
即 σ_e² = P_elem · 10^(−CNR/10)。注意 CNR 0 dB 意味着估计误差与信道同功率（最严苛的一档），
等价于 NMSE_est = −CNR = 0 dB。CNR = inf 表示理想 CSI。

扫描：CNR ∈ {inf, 20, 10, 0} dB × 反馈 SNR ∈ {0, 10, 20} dB，3 次重复（每次重抽 E 与链路噪声，
同一重复内各 CNR 共用底层高斯 -> 配对）。

用法：
  python ue_est_v5.py --n 300 --repeats 3 --ckpt ckpt_v2 --out results_ue_cnr.json
  python ue_est_v5.py --ckpt ckpt_ue_aug.pt --out results_ue_cnr_aug.json
"""
import argparse
import json

import numpy as np
import torch

import model_jscm as M
import quality_v5 as Q

BASE_E_SEED = 7000


def elem_power(x):
    """全集平均复数元素功率 E[|H_elem|²]（第 0/1 通道为实/虚部）。"""
    return float((x[:, 0].pow(2) + x[:, 1].pow(2)).mean())


def add_est_error(x, cnr_db, seed, p_elem):
    """元素级复高斯估计误差：σ_e² = P_elem·10^(−CNR/10)；实/虚部各 σ_e²/2。"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(x.shape, generator=g)
    sigma2 = p_elem * (10 ** (-cnr_db / 10.0))
    return x + z * (sigma2 / 2.0) ** 0.5


def db_abs(pred, target, den):
    num = (pred - target).pow(2).sum((1, 2, 3))
    return 10 * torch.log10(num / den + 1e-12)


def load_net(ckpt, xtr, ns=64, m=16):
    net = M.Link("learned_soft", ns=ns, m=m)
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd if not isinstance(sd, dict) or "state_dict" not in sd
                        else sd["state_dict"])
    net.eval()
    net.p_cal = M.calibrate(net, xtr)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2/learned_soft_s0.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--snrs", default="0,10,20")
    ap.add_argument("--cnr", default="inf,20,10,0")
    ap.add_argument("--out", default="results_ue_cnr.json")
    a = ap.parse_args()

    xtr = Q.load_split("train")
    xte = Q.load_split("test")[: a.n]
    net = load_net(a.ckpt, xtr)
    den_true = xte.pow(2).sum((1, 2, 3))
    p_elem = elem_power(xte)

    cnrs = []
    for s in a.cnr.split(","):
        s = s.strip()
        cnrs.append((None if s.lower() == "inf" else float(s)))
    snrs = [float(s) for s in a.snrs.split(",")]

    out = {"ckpt": a.ckpt, "n": a.n, "repeats": a.repeats, "p_elem": p_elem,
           "model": "E_elem ~ CN(0, P_elem*10^(-CNR/10))", "rows": []}

    for cnr in cnrs:
        for snr in snrs:
            fb, fb_true, fin_abs, au, sg = [], [], [], [], []
            for j in range(a.repeats):
                e_seed = BASE_E_SEED + j
                n_seed = 1234 + j
                xe = xte if cnr is None else add_est_error(xte, cnr, e_seed, p_elem)
                with torch.no_grad():
                    tx, y = Q.transmit(net, xe, snr, seed=n_seed)
                    xh = Q.decode(net, y, snr)
                    r = np.asarray(Q.q2_reencode(net, y, xh))
                fb.append(float(np.asarray(Q.nmse_db_per(xh, xe)).mean()))
                fb_true.append(float(np.asarray(db_abs(xh, xe, den_true)).mean()))
                fa = np.asarray(db_abs(xh, xte, den_true))
                fin_abs.append(float(fa.mean()))
                au.append(float(Q.auroc(torch.from_numpy(r), Q.bad_label(fa, frac=0.2))))
                sg.append(float(np.asarray(Q.sgcs_per(xh, xte)).mean()))
            row = dict(cnr=("inf" if cnr is None else cnr), snr=snr,
                       nmse_fb_db=round(float(np.mean(fb)), 3),
                       nmse_fb_true_den=round(float(np.mean(fb_true)), 3),
                       nmse_final_db=round(float(np.mean(fin_abs)), 3),
                       sgcs=round(float(np.mean(sg)), 4),
                       auroc_mean=round(float(np.mean(au)), 3),
                       auroc_std=round(float(np.std(au, ddof=1)) if len(au) > 1 else 0.0, 3))
            out["rows"].append(row)
            print("CNR %5s SNR %3d | final %7.3f dB | SGCS2 %.4f | AUROC %.3f" % (row["cnr"], int(snr), row["nmse_final_db"], row["sgcs"], row["auroc_mean"]), flush=True)

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
