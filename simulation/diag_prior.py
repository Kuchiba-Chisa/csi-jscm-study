# -*- coding: utf-8 -*-
"""诊断先验采样：打印各步的 |x| 与 x0 估计统计。"""
import torch
from diff_prior import TinyUNet, Diffusion, sample_prior, stats_of

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("ckpt_diff/prior.pt", map_location="cpu")
print("epochs trained:", ck.get("epochs"))
model = TinyUNet().to(dev)
model.load_state_dict(ck["model"])
model.eval()
diff = Diffusion(device=dev)

n = 8
x = torch.randn(n, 2, 32, 32, device=dev)
with torch.no_grad():
    for i in reversed(range(diff.T)):
        t = torch.full((n,), i, dtype=torch.long, device=dev)
        eps = model(x, t)
        x0 = ((x - diff.sq_1mab[t][:, None, None, None] * eps) /
              diff.sq_ab[t][:, None, None, None])
        if i % 50 == 0 or i < 5:
            print(f"i={i:3d} ab={float(diff.ab[t][0]):.5f} |x|={x.pow(2).mean().sqrt():8.2f} "
                  f"|x0|={x0.pow(2).mean().sqrt():8.2f} x0max={float(x0.abs().max()):7.2f}")
        x0 = x0.clamp(-6, 6)
        if i > 0:
            tp = torch.full((n,), i - 1, dtype=torch.long, device=dev)
            mean, var = diff.posterior(x, x0, t, tp)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0
print("final", stats_of(x))
