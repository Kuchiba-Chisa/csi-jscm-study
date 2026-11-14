"""v5 主实验（D4）：逐样本可靠性指标的测试集结果。

跑的东西（每个 SNR ∈ {0,10,20}，同一批 H 与同一组噪声，保证配对）：
  Q0  只看 SNR（常数，基线）
  Q1  最近星座点距离
  Q2  重编码一致性 ‖y − F(Ĥ)‖²/Ns
  O1  真实判决错误率（特权诊断，不是上限）
另外报：逐样本 NMSE / 平方 SGCS、风险—覆盖率曲线、理想与随机参照。

用法：
  python run_v5.py --schemes learned_soft --seeds 0 1 2 --variants S0 S2a --out results_v5_main.json
"""
import argparse
import json

import numpy as np
import torch

import quality_v5 as Q
from bn_v5 import recalibrate

SNRS = (0, 10, 20)


@torch.no_grad()
def transmit3(net, x, snr, noise_seed=1234):
    """返回 (tx, y, idx)：idx 是发送端实际选中的星座点编号（for O1）。

    噪声在 tx 形状 (n, ns) 上生成，种子固定 → 各模型/各指标共享同一组噪声。
    """
    z = net.enc(x)
    z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
    tx, idx = net.modulate(z)
    sig = Q.sigma_of(net, snr)
    g = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(tx.shape, generator=g)
    return tx, tx + sig * noise, idx


@torch.no_grad()
def decision_error(net, y, idx):
    """O1：接收端硬判决与发送端实际符号不一致的比例（逐样本）。"""
    pts = Q._pts(net).view(1, 1, -1)
    hard = (y.unsqueeze(-1) - pts).abs().argmin(-1)
    return (hard != idx).float().mean(1)


def one_run(net, x, snr, noise_seed, kind):
    tx, y, idx = transmit3(net, x, snr, noise_seed=noise_seed)
    xhat = Q.decode(net, y, snr)
    nm = Q.nmse_db_per(xhat, x)
    sg = Q.sgcs_per(xhat, x)
    sc = {
        "Q0": Q.q0_snr(y, snr),
        "Q1": Q.q1_const_dist(net, y),
        "Q2": Q.q2_reencode(net, y, xhat),
        "O1": decision_error(net, y, idx),
    }
    out = {"nmse": nm.numpy(), "sgcs": sg.numpy(), "scores": {}}
    for k, v in sc.items():
        v = v.numpy().astype(np.float64)
        out["scores"][k] = {
            "score": v,
            **Q.summarize(v, nm.numpy(), k),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["learned_soft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["S0", "S2a"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="ckpt_v2")
    ap.add_argument("--noise-seeds", nargs="+", type=int, default=[1234])
    ap.add_argument("--out", default="results_v5_main.json")
    a = ap.parse_args()

    xtr, x = Q.load_split("train"), Q.load_split(a.split)
    res = {}
    for scheme in a.schemes:
        for seed in a.seeds:
            for var in a.variants:
                key = f"{scheme}_s{seed}_{var}"
                res[key] = {}
                for snr in SNRS:
                    per_noise = []
                    for nseed in a.noise_seeds:
                        net = Q.build(scheme, seed, ckpt_dir=a.ckpt, xtr=xtr)
                        if var == "S2a":
                            recalibrate(net, xtr, snr=snr, mixed=False)
                        elif var == "S1":
                            recalibrate(net, xtr, mixed=True)
                        r = one_run(net, x, snr, nseed, var)
                        per_noise.append({
                            "nmse_mean": float(np.mean(r["nmse"])),
                            "sgcs_mean": float(np.mean(r["sgcs"])),
                            "auroc_nmse": Q.auroc(r["scores"]["Q2"]["score"], Q.bad_label(r["nmse"])),
                            "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "score"}
                                        for k, v in r["scores"].items()},
                        })
                    res[key][str(snr)] = per_noise
                    m = per_noise[0]["metrics"]
                    print(f"{key} SNR={snr:2d}  NMSE={per_noise[0]['nmse_mean']:+.3f}  "
                          + "  ".join(f"{k}:{m[k]['auroc']:.3f}" for k in ("Q1", "Q2", "O1")),
                          flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
