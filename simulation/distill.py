# -*- coding: utf-8 -*-
"""E3a / E3b / E4：同一 SNR 混合分布下的三种训练。

协议（00_总方案_v2.md 第 4 节）：
  - SNR 采样：一半 batch 从 {0,10,20} 抽（锚点 batch），另一半从 U(0,20) 抽；
  - 学生统一是条件共享模型 learned_soft_snr，种子/轮数/结构相同；
  - E3a：只做重建（匹配对照）
  - E3b：重建损失按锚点加权（--anchor_weights 3,1,1，权重在验证集上选）
  - E4 ：重建 + 教师蒸馏 L = ||Ĥs − H||² + λ||Ĥs − stopgrad(ĤT)||²，
         蒸馏项只加在锚点 batch 上，第 k 个种子用第 k 个教师

用法：
  python distill.py --mode e4 --lam 1.0 --seeds 0,1,2 --device cuda
输出：results_distill.json（val/test 逐点曲线）
"""
import argparse
import json
import os

import numpy as np
import torch

from model_jscm import Link, nmse_db, sgcs, calibrate
from run_experiments import load_split, SNR_GRID
from eval_v2 import sgcs_sq

ANCHORS = (0.0, 10.0, 20.0)


class MixSampler:
    """一半锚点 batch（SNR 精确等于 0/10/20），一半均匀 batch。"""

    def __init__(self, n, bs, anchors=ANCHORS):
        self.n, self.bs, self.anchors = n, bs, anchors
        self.n_batch = (n + bs - 1) // bs

    def __iter__(self):
        perm = torch.randperm(self.n)
        for i in range(0, self.n, self.bs):
            j = perm[i:i + self.bs]
            if torch.rand(1).item() < 0.5:
                k = int(torch.randint(0, len(self.anchors), (1,)).item())
                yield j, self.anchors[k], k
            else:
                yield j, float(torch.empty(1).uniform_(0, 20).item()), None


XTR = None


def pcal_from_cache(scheme, seed):
    """复用 E1/E2 重评时标定过的 P_tx（results_v3_eval.json），保证口径一致。"""
    try:
        d = json.load(open("results_v3_eval.json", encoding="utf-8"))["p_cal"]
        return float(d[scheme][seed])
    except Exception:
        return None


def load_teacher(anchor, seed, ns, m, device):
    p = f"ckpt_v2/learned_soft_f{int(anchor)}_s{seed}.pt"
    if not os.path.exists(p):
        return None
    t = Link("learned_soft", ns, m)
    t.load_state_dict(torch.load(p, map_location="cpu"))
    t.p_cal = pcal_from_cache("learned_soft_f%d" % int(anchor), seed) or calibrate(t, XTR)
    t.to(device).eval()
    for q in t.parameters():
        q.requires_grad_(False)
    return t


def train(mode, xtr, xva, xte, epochs, ns, m, seed, device, lam=1.0,
          anchor_w=(1.0, 1.0, 1.0), log=print):
    torch.manual_seed(seed)
    net = Link("learned_soft_snr", ns, m).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    teachers = {}
    if mode == "e4":
        for a in ANCHORS:
            t = load_teacher(a, seed, ns, m, device)
            if t is None:
                raise FileNotFoundError(f"缺教师 ckpt: learned_soft_f{int(a)}_s{seed}")
            teachers[a] = t

    for ep in range(epochs):
        net.train()
        tot, cnt = 0.0, 0
        for j, snr, aidx in MixSampler(len(xtr), 128):
            xb = xtr[j]
            pred = net(xb, snr)
            rec = (pred - xb).pow(2).mean()
            w = anchor_w[aidx] if (aidx is not None and mode == "e3b") else 1.0
            loss = w * rec
            if mode == "e4" and aidx is not None:
                with torch.no_grad():
                    y_t = teachers[ANCHORS[aidx]](xb, snr)
                loss = loss + lam * (pred - y_t).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
            cnt += len(j)
        sched.step()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"    ep{ep:02d} loss={tot / cnt:.4f}")
    return net


@torch.no_grad()
def curve(net, x, snrs=SNR_GRID, noise_seed=20260101):
    net.eval()
    if getattr(net, "p_cal", None) is None:
        dev = next(net.parameters()).device
        net.p_cal = calibrate(net, XTR.to(dev))
    out = {"nmse": [], "sgcs2": [], "sgcs1": []}
    for s in snrs:
        torch.manual_seed(noise_seed + int(round(s)))
        pred = net(x, float(s))
        out["nmse"].append(round(float(nmse_db(pred, x)), 4))
        out["sgcs2"].append(round(float(sgcs_sq(pred, x)), 4))
        out["sgcs1"].append(round(float(sgcs(pred, x)), 4))
    return out


def main():
    global XTR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="e4", choices=["e3a", "e3b", "e4"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--anchor_weights", default="1,1,1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ns", type=int, default=64)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results_distill.json")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    xtr, xva, xte = load_split("train"), load_split("val"), load_split("test")
    XTR = xtr
    xtr, xva, xte = xtr.to(dev), xva.to(dev), xte.to(dev)
    aw = tuple(float(v) for v in args.anchor_weights.split(","))
    tag = f"{args.mode}" + (f"_lam{args.lam}" if args.mode == "e4" else "") + \
          (f"_w{args.anchor_weights}" if args.mode == "e3b" else "")

    out = {"snr": SNR_GRID.tolist(), "tag": tag, "epochs": args.epochs,
           "val": {}, "test": {}}
    if os.path.exists(args.out):
        out = json.load(open(args.out, encoding="utf-8"))

    for seed in [int(s) for s in args.seeds.split(",") if s]:
        if str(seed) in out["val"].get(tag, {}):
            print(f"skip {tag} s{seed}")
            continue
        print(f"[{tag} s{seed}] training {args.epochs} epochs on {dev}")
        net = train(args.mode, xtr, xva, xte, args.epochs, args.ns, args.m,
                    seed, dev, lam=args.lam, anchor_w=aw)
        out["val"].setdefault(tag, {})[str(seed)] = curve(net, xva)
        out["test"].setdefault(tag, {})[str(seed)] = curve(net, xte)
        m10 = out["test"][tag][str(seed)]["nmse"][5]
        print(f"  -> test@10dB {m10:+.2f} dB")
        os.makedirs("ckpt_distill", exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()},
                   f"ckpt_distill/{tag}_s{seed}.pt")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
