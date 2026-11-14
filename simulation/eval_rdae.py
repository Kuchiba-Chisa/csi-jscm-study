"""RD-JSCC AE 的补充评估：曲线、BN 对照（S0 / S2a）、Q2 AUROC。

与 `bn_v5.py` 同一套 S0/S2a 定义：
- S0：直接用训练累积的 running stats；
- S2a：在**该 SNR** 的训练数据上重跑 40 个 batch、只更新解码器 BN 的 mean/var（不改权重）。
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from model_rdae import LinkRDAE, q2_scores
from model_v4 import nmse_db, sgcs
from quality_v5 import auroc, bad_label, spearman
from train_rdae import load01

SNR_GRID = list(range(0, 21, 2))


def dec_bns(net):
    return [m for m in net.dec.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def recalibrate(net, xtr, snr, ns=64, bs=128, n_batch=40):
    bns = dec_bns(net)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.eval()
    for m in bns:
        m.train()
    gen = torch.Generator(device=xtr.device).manual_seed(7)
    idx = torch.randperm(xtr.shape[0], generator=gen)[: n_batch * bs]
    for i in range(0, len(idx), bs):
        xb = xtr[idx[i:i + bs]]
        net(xb, float(snr), ns)
    net.eval()
    for m in bns:
        m.momentum = 0.1


@torch.no_grad()
def curve(net, x, ns):
    nm, sc = [], []
    for snr in SNR_GRID:
        out = net(x, float(snr), ns)
        nm.append(float(nmse_db(out, x)))
        sc.append(float(sgcs(out, x)))
    return nm, sc


@torch.no_grad()
def q2(net, x, ns, snr, seed=1234):
    return q2_scores(net, x, ns, snr, seed=seed)[:2]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", default="ckpt_rdae/rdae_sample_s0.pt")
    pa.add_argument("--ns", type=int, default=64)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--out", default="results_rdae_bn.json")
    a = pa.parse_args()

    dev = torch.device(a.device)
    net = LinkRDAE().to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    xtr = load01("train").to(dev)
    xte = load01("test").to(dev)
    net.calibrate(xtr, (16, 32, 64))
    net.eval()

    rec = {"ckpt": a.ckpt, "ns": a.ns, "snr": SNR_GRID}
    rec["curve"] = {}
    for ns in (16, 32, 64):
        nm, sc = curve(net, xte, ns)
        rec["curve"][str(ns)] = {"nmse": nm, "sgcs": sc}
        print(f"ns={ns:2d} " + " ".join(f"{v:+.2f}" for v in nm), flush=True)

    # 每个 SNR 都从**原始 ckpt**重新载入：早先是连续重标定，10 dB 的"before"其实是 0 dB 之后的状态。
    rec["bn"] = {}
    rec["q2_s0"] = {}
    rec["q2_s2a"] = {}
    for snr in (0, 10, 20):
        net.load_state_dict(torch.load(a.ckpt, map_location=dev))
        net.calibrate(xtr, (16, 32, 64))          # p_cal 与 ckpt 一起重置
        net.eval()
        before = float(nmse_db(net(xte, float(snr), a.ns), xte))

        # Q2 在 S0 下
        for ns in (16, 32, 64):
            au, rho, nm = q2_scores(net, xte, ns, snr)
            rec["q2_s0"][f"{ns}_{snr}"] = [au, rho, nm]
            print(f"q2-s0  ns={ns:2d} snr={snr:2d}  AUROC={au:.3f} rho={rho:+.3f} nmse={nm:+.3f}", flush=True)

        recalibrate(net, xtr, snr, a.ns)
        after = float(nmse_db(net(xte, float(snr), a.ns), xte))
        rec["bn"][str(snr)] = [before, after]
        print(f"bn  snr={snr:2d}  S0={before:+.3f}  S2a={after:+.3f}  Δ={before - after:+.3f}", flush=True)

        for ns in (16, 32, 64):
            au, rho, nm = q2_scores(net, xte, ns, snr)
            rec["q2_s2a"][f"{ns}_{snr}"] = [au, rho, nm]
            print(f"q2-s2a ns={ns:2d} snr={snr:2d}  AUROC={au:.3f} rho={rho:+.3f} nmse={nm:+.3f}", flush=True)

    json.dump(rec, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
