"""BN -> GroupNorm 对照：同一 RD-JSCC AE 骨干，只换归一化层。

目的：检验归一化结构替换与 BN 统计量重标定在不同条件下的相对效果，
GroupNorm 不使用 BN 的 running stats，但相对性能、场景变化与其他模块仍需实测，
不预设 GN 一定变差，也不把没有 BN 统计量等同于系统免维护。

口径与 `train_rdae.py` **完全一致**（同数据、同 SNR/Ns 采样、同轮数、同评估），
唯一差别：构造 `LinkRDAE` 后把每个 `nn.BatchNorm2d(C)` 替换成 `nn.GroupNorm(g, C)`（eps 同为 1e-3）。

  # 冒烟（1 epoch，验证替换与形状）
  python train_rdae_gn.py --smoke
  # 主对照（建议 CPU 空闲时跑）
  python train_rdae_gn.py --seeds 0 1 2 --epochs 40

产物：`results_rdae_gn.json`、`ckpt_rdae_gn/rdae_gn_sample_s{seed}.pt`
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import torch
import torch.nn as nn

import train_rdae as T
from model_rdae import LinkRDAE as _LinkRDAE


def _groups_for(c: int, want: int = 4) -> int:
    """选一个能整除通道数的分组数；2 通道时退化成逐通道（等价 InstanceNorm）。"""
    for g in (want, 2, 1):
        if c % g == 0:
            return g
    return 1


def swap_bn_to_gn(mod: nn.Module, want: int = 4) -> int:
    """就地替换 BatchNorm2d -> GroupNorm，返回替换个数。"""
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            g = _groups_for(child.num_features, want)
            setattr(mod, name, nn.GroupNorm(g, child.num_features, eps=child.eps))
            n += 1
        else:
            n += swap_bn_to_gn(child, want)
    return n


class LinkRDAE_GN(_LinkRDAE):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_gn = swap_bn_to_gn(self)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in self.modules()), "仍有 BN 残留"


# 让 `train_rdae.train()` 构造 GN 版本（复用其训练循环，保证口径一致）
T.LinkRDAE = LinkRDAE_GN


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seeds", type=int, nargs="+", default=[0])
    pa.add_argument("--epochs", type=int, default=40)
    pa.add_argument("--bs", type=int, default=128)
    pa.add_argument("--lr", type=float, default=2e-3)
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--snr-mode", choices=["sample", "batch"], default="sample")
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--threads", type=int, default=0, help=">0 时限制 torch 线程")
    pa.add_argument("--out", default="results_rdae_gn.json")
    a = pa.parse_args()
    if a.smoke:
        a.epochs, a.seeds = 1, a.seeds[:1]
    if a.threads > 0:
        torch.set_num_threads(a.threads)

    res = json.load(io.open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    xte = T.load01("test")
    for seed in a.seeds:
        tag = f"rdae_gn_{a.snr_mode}_s{seed}"
        print(f"[{tag}] train (GN)", flush=True)
        net = T.train(a, seed)
        net.eval()
        n_gn = sum(1 for m in net.modules() if isinstance(m, nn.GroupNorm))
        has_stats = any(hasattr(m, "running_mean") and m.running_mean is not None for m in net.modules())
        print(f"  GroupNorm 层数={n_gn}  是否有冻结统计量={has_stats}", flush=True)
        rec = {"snr": T.SNR_GRID, "nmse": {}, "sgcs": {}, "q2": {}, "n_gn": n_gn, "has_running_stats": has_stats}
        for ns in T.NS_LIST:
            nm, sc = T.evaluate(net, xte, ns)
            rec["nmse"][str(ns)], rec["sgcs"][str(ns)] = nm, sc
            print(f"  ns={ns:2d} " + " ".join(f"{v:+.2f}" for v in nm), flush=True)
        if not a.smoke:
            for snr in T.Q2_SNRS:
                rec["q2"][str(snr)] = T.q2_auroc(net, xte, 64, snr)
        res[tag] = rec
        os.makedirs("ckpt_rdae_gn", exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, f"ckpt_rdae_gn/{tag}.pt")
        json.dump(res, io.open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
