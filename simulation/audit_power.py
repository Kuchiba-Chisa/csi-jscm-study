# -*- coding: utf-8 -*-
"""功率审计：各方案在测试集上的实际发送功率与「每单位功率」的 NMSE。

回答的质疑：sscc 与 lc_hard 的 +0.18 dB 是不是发送功率不同造成的？
做法：固定 SNR=10 dB，对每个方案统计 E[tx^2]（量化/星座之后的真实符号能量），
并把 NMSE 换算成「同等功率」下的值：NMSE_corr = NMSE * E[tx^2]。
"""
import json
import torch

from model_jscm import Link, nmse_db
from run_experiments import load_split, parse_scheme, SNR_GRID

torch.manual_seed(0)
xte = load_split("test")

rows = []
for name in ["analog", "sscc", "lc_hard", "jscm", "jscm_snr"]:
    base, _ = parse_scheme(name)
    net = Link(base, 64, 16)
    net.load_state_dict(torch.load(f"ckpt_{name}.pt", map_location="cpu"))
    net.eval()

    tx2, nm = [], []
    with torch.no_grad():
        for snr in [0.0, 10.0, 20.0]:
            for i in range(0, len(xte), 500):
                xb = xte[i:i + 500]
                z = net.enc(xb)
                z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
                if base == "analog":
                    tx = z
                else:
                    tx, _ = net.modulate(z)
                tx2.append(float(tx.pow(2).mean()))
                nm.append(float(nmse_db(net(xb, snr), xb)))
    p = sum(tx2) / len(tx2)
    rows.append((name, p, sum(nm) / len(nm)))

print(f"{'scheme':10s} {'E[tx^2]':>9s} {'NMSE(dB)':>10s} {'功率校正后':>11s}")
for name, p, nm in rows:
    corr = nm - 10 * torch.log10(torch.tensor(p))
    print(f"{name:10s} {p:9.3f} {nm:10.2f} {float(corr):11.2f}")

d = {n: (p, nm) for n, p, nm in rows}
gain = d["sscc"][1] - d["lc_hard"][1]
gcorr = float(d["sscc"][1] - 10 * torch.log10(torch.tensor(d["sscc"][0]))
              - (d["lc_hard"][1] - 10 * torch.log10(torch.tensor(d["lc_hard"][0]))))
print(f"\n星座增益（未校正） = {gain:+.3f} dB")
print(f"星座增益（功率校正）= {gcorr:+.3f} dB")

json.dump({n: {"tx_power": p, "nmse": nm} for n, p, nm in rows},
          open("power_audit.json", "w"), indent=1)
print("saved power_audit.json")
