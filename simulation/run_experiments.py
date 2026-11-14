"""跑四组对比实验并出图。

    python run_experiments.py --smoke     # 先跑通（小数据、2 epoch）
    python run_experiments.py             # 正式实验

产物: results.json / fig_snr.png / fig_constellation.png / fig_recon.png
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

from model_jscm import Link, nmse_db, sgcs, cos_flat, canon, BASES

SNR_GRID = np.arange(0, 21, 2)          # dB
SCHEMES = ["analog", "fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr"]
# Q3 对照：同一架构 learned_soft，专用（固定 SNR 训练）vs 共享（随机 SNR）vs 条件共享
Q3_SCHEMES = ["learned_soft_f0", "learned_soft_f10", "learned_soft_f20",
              "learned_soft", "learned_soft_snr"]
LABEL = {
    "analog": "无星座约束（连续符号，参考上限）",
    "fixed_hard": "固定均匀 16-PAM + 硬判决",
    "learned_hard": "可学习 16-PAM + 硬判决",
    "learned_soft": "可学习 16-PAM + 带噪观测直入解码器（共享模型）",
    "learned_soft_snr": "同上 + 解码器输入噪声水平（条件共享模型）",
    "learned_soft_f0": "专用模型：只在 SNR=0 dB 训练",
    "learned_soft_f10": "专用模型：只在 SNR=10 dB 训练",
    "learned_soft_f20": "专用模型：只在 SNR=20 dB 训练",
}

FIXED_RE = re.compile(r"^(.+)_f(\d+)$")


def parse_scheme(name):
    """'learned_soft_f10' -> ('learned_soft', 10.0)：同架构，只在 SNR=10 dB 训练。旧名同样可用。

    'learned_soft_ps' -> ('learned_soft', 'ps')：B2，每个样本独立抽 SNR。
    """
    if name.endswith("_ps") and canon(name[:-3]) in BASES:
        return canon(name[:-3]), "ps"
    mt = FIXED_RE.match(name)
    if mt and canon(mt.group(1)) in BASES:
        return canon(mt.group(1)), float(mt.group(2))
    return canon(name), None


def load_split(name):
    d = np.load(os.path.join("data", f"{name}.npz"))["x"].astype(np.float32)
    # 逐样本功率归一化：NMSE 本身是尺度不变的，归一化只改善优化条件
    d = d / np.sqrt((d ** 2).mean(axis=(1, 2, 3), keepdims=True))
    return torch.from_numpy(d)


def train_one(scheme, xtr, xva, epochs, ns, m, seed=0, log=print, fixed_snr=None,
              power_ref=True, device="cpu"):
    torch.manual_seed(seed)
    base, fs = parse_scheme(scheme)
    per_sample = (fs == "ps")
    if per_sample:
        fs = None
    if fixed_snr is None:
        fixed_snr = fs
    net = Link(base, ns=ns, m=m, power_ref=power_ref).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = xtr.shape[0]
    bs = 128
    best = 1e9
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = xtr[idx]
            if fixed_snr is not None:
                snr = float(fixed_snr)                       # 固定 SNR 训练（对照：专用模型）
            elif per_sample:
                snr = torch.empty(xb.shape[0], device=xb.device).uniform_(0, 20)   # B2：每样本独立 SNR
            else:
                snr = torch.empty(1).uniform_(0, 20).item()  # 训练时 SNR 随机 -> 单模型自适应
            pred = net(xb, snr)
            loss = torch.mean((pred - xb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
            cnt += xb.shape[0]
        sched.step()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"    ep{ep:02d} train_mse={tot/cnt:.4f}")
    return net


@torch.no_grad()
def evaluate(net, xte, snrs=SNR_GRID, noise_seed=1234):
    """固定测试噪声种子，所有方案看到同一组噪声实现。"""
    net.eval()
    out = {"nmse": [], "sgcs": [], "cos_flat": [], "tx_power": []}
    for s in snrs:
        torch.manual_seed(noise_seed + int(s))
        p = net(xte, float(s))
        out["nmse"].append(float(nmse_db(p, xte)))
        out["sgcs"].append(float(sgcs(p, xte)))
        out["cos_flat"].append(float(cos_flat(p, xte)))
        out["tx_power"].append(net.last_tx_power)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ns", type=int, default=64, help="实数符号数（bit=ns*log2M）")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--only", type=str, default="", help="只跑这些方案，逗号分隔")
    ap.add_argument("--q3", action="store_true", help="跑 Q3 对照（专用/共享/条件共享）")
    ap.add_argument("--seeds", type=str, default="0,1,2", help="训练种子，逗号分隔")
    ap.add_argument("--out", type=str, default="results_v2.json")
    ap.add_argument("--legacy_power", action="store_true",
                    help="按旧口径（z 功率）定噪声，仅用于复现旧结果")
    ap.add_argument("--device", type=str, default="cpu", help="cpu / cuda / auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"device=cuda  {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(os.cpu_count() or 4)
        print("device=cpu")
    xtr, xva, xte = load_split("train"), load_split("val"), load_split("test")
    xtr, xva, xte = xtr.to(dev), xva.to(dev), xte.to(dev)
    if args.smoke:
        xtr, xva, xte = xtr[:512], xva[:128], xte[:128]
        args.epochs = 2
    print(f"train {tuple(xtr.shape)}  test {tuple(xte.shape)}  "
          f"bits={args.ns*int(np.log2(args.m))}")

    if args.only:
        schemes = [x for x in args.only.split(",") if x]
    else:
        schemes = Q3_SCHEMES if args.q3 else SCHEMES
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    ckdir = os.environ.get("JSCM_CKPT", "ckpt_v2")
    os.makedirs(ckdir, exist_ok=True)

    # 断点续跑：已有结果的 (scheme, seed) 跳过
    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out, encoding="utf-8"))["results"]

    def dump():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"snr": SNR_GRID.tolist(), "results": results, "ns": args.ns,
                       "m": args.m, "power_ref": not args.legacy_power,
                       "device": args.device,
                       "constellation": f"real {args.m}-PAM", "epochs": args.epochs},
                      f, ensure_ascii=False, indent=2)

    for sc in schemes:
        results.setdefault(sc, {})
        for sd in seeds:
            if str(sd) in results[sc]:
                print(f"[{sc} seed{sd}] 已有结果，跳过")
                continue
            t0 = time.time()
            print(f"[{sc} seed{sd}] training {args.epochs} epochs ...", flush=True)
            net = train_one(sc, xtr, xva, args.epochs, args.ns, args.m, seed=sd,
                            power_ref=not args.legacy_power, device=dev)
            r = evaluate(net, xte)
            r["time_s"] = round(time.time() - t0, 1)
            r["params"] = sum(p.numel() for p in net.parameters())
            results[sc][str(sd)] = r
            print(f"  NMSE = {[round(v, 2) for v in r['nmse']]}  "
                  f"P_tx@10dB = {r['tx_power'][5]:.3f}  ({r['time_s']:.0f}s)", flush=True)
            cpu_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            torch.save(cpu_sd, os.path.join(ckdir, f"{sc}_s{sd}.pt"))
            dump()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
