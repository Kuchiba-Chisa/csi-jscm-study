# -*- coding: utf-8 -*-
"""功率敏感度：把发送符号乘 beta，看 NMSE 怎么变。

用途：判断「sscc 实测功率 0.849 vs lc_hard 0.936（差 0.4 dB）」能解释多少 NMSE 差异。
若 d(NMSE)/d(SNR) ≈ 1，则功率差会等量反映到 NMSE，必须校正；
若远小于 1（量化受限），则功率差是二阶小量。
"""
import torch

from model_jscm import Link, nmse_db
from run_experiments import load_split, parse_scheme

torch.manual_seed(0)
xte = load_split("test")[:1000]


def measure(name, beta, snr):
    """按 beta 缩放符号功率后重测 NMSE（缩放点在量化之后、加噪之前）。"""
    base, _ = parse_scheme(name)
    net = Link(base, 64, 16)
    net.load_state_dict(torch.load(f"ckpt_{name}.pt", map_location="cpu"))
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xte), 500):
            xb = xte[i:i + 500]
            z = net.enc(xb)
            z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
            tx, idx = net.modulate(z)
            sigma = 10 ** (-snr / 20.0)
            y = tx * beta + sigma * torch.randn_like(tx)
            y = net.receive(y, idx)
            out.append(float(nmse_db(net.dec(y), xb)))
    return sum(out) / len(out)


print(f"{'SNR':>5s} " + " ".join(f"b={b:<5.2f}" for b in (0.85, 1.0, 1.2, 1.4)))
for snr in (0, 10, 20):
    row = [measure("sscc", b, snr) for b in (0.85, 1.0, 1.2, 1.4)]
    # 功率每 +1 dB 对应的 NMSE 改善（斜率）
    dsnr = 10 * torch.log10(torch.tensor(1.4 / 1.0))
    slope = (row[-1] - row[1]) / float(dsnr)
    print(f"{snr:5d} " + " ".join(f"{v:7.2f}" for v in row) + f"   d NMSE/d SNR = {slope:+.2f}")
