"""RD-JSCC 的 AE（AFCsiNetPlus）移植到 PyTorch，用于外部效度对照。

来源：`baseline_review/rd-jscc-main/modules/vq_compressor.py`
- `AFModule`（11–28 行）：x * sigmoid(W2 relu(W1 [GAP(x); snr] + b1) + b2)，逐通道广播
- `AFCsiNetPlusEncoderLayer`（31–51 行）：[Conv(2,7x7)+BN+LeakyReLU+AF] x2 → Flatten → Dense(tanh)
- `AFCsiNetPlusDecoderLayer`（89–143 行）：
  Dense(32*32*2) → Reshape(2,32,32) → AF → Conv(2,7x7)+BN+sigmoid → AF
  → 5 个残差块 {Conv(8,7x7)+BN+LReLU+AF, Conv(16,5x5)+BN+LReLU+AF, Conv(2,3x3)+BN+tanh+AF}，x = x + y

只移植 AE（不含扩散、不含 VQ）；发送口径与 `model_v4.LinkV4` 逐行对齐：
`normalize_power → to_symbols → qam_map_ste → add_noise → from_symbols*pair_mask`。
本改编实际输入为逐样本实数 RMS 归一后除以 3；不是 [0,1]，NMSE 仅对共同缩放不变。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v4 import (
    NS_MAX, add_noise, from_symbols, nmse_db, normalize_power, pair_mask,
    qam_map_ste, sgcs, to_symbols, tx_power,
)


class AFModule(nn.Module):
    """Attention Feature：用 (通道均值, σ/√P) 生成逐通道门控。"""

    def __init__(self, ch: int, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(ch + 1, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x: torch.Tensor, snr_feat: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(2, 3))                      # (B, C) — GlobalAveragePooling2D
        m = torch.cat([m, snr_feat], dim=1)         # (B, C+1)
        m = torch.sigmoid(self.fc2(F.relu(self.fc1(m))))
        return x * m[:, :, None, None]


class CBR(nn.Sequential):
    def __init__(self, c_in, c_out, k, act="lrelu"):
        layers = [nn.Conv2d(c_in, c_out, k, padding=k // 2), nn.BatchNorm2d(c_out, eps=1e-3)]
        layers.append({"lrelu": nn.LeakyReLU(0.3, inplace=True),
                       "sigmoid": nn.Sigmoid(),
                       "tanh": nn.Tanh()}[act])
        super().__init__(*layers)


class EncoderRDAE(nn.Module):
    def __init__(self, out_dim: int = 2 * NS_MAX):
        super().__init__()
        self.c1, self.af1 = CBR(2, 2, 7), AFModule(2)
        self.c2, self.af2 = CBR(2, 2, 7), AFModule(2)
        self.fc = nn.Linear(2 * 32 * 32, out_dim)

    def forward(self, x, snr_feat):
        x = self.af1(self.c1(x), snr_feat)
        x = self.af2(self.c2(x), snr_feat)
        return torch.tanh(self.fc(x.flatten(1)))


class _ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1, self.af1 = CBR(2, 8, 7), AFModule(8)
        self.b2, self.af2 = CBR(8, 16, 5), AFModule(16)
        self.b3, self.af3 = CBR(16, 2, 3, act="tanh"), AFModule(2)

    def forward(self, x, snr_feat):
        y = self.af1(self.b1(x), snr_feat)
        y = self.af2(self.b2(y), snr_feat)
        y = self.af3(self.b3(y), snr_feat)
        return x + y


class DecoderRDAE(nn.Module):
    def __init__(self, in_dim: int = 2 * NS_MAX, extra: int = 0, residual_num: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_dim + extra, 2 * 32 * 32)
        self.af0 = AFModule(2)
        self.c0, self.af0b = CBR(2, 2, 7, act="sigmoid"), AFModule(2)
        self.blocks = nn.ModuleList([_ResBlock() for _ in range(residual_num)])

    def forward(self, s, snr_feat):
        x = self.fc(s).view(-1, 2, 32, 32)
        x = self.af0(x, snr_feat)
        x = self.af0b(self.c0(x), snr_feat)
        for blk in self.blocks:
            x = blk(x, snr_feat)
        return x


class LinkRDAE(nn.Module):
    """RD-JSCC AE 版链路；`cond=True` 时把 [σ/√P, ns/64] 拼进解码器输入。"""

    def __init__(self, cond: bool = False, ns_max: int = NS_MAX):
        super().__init__()
        self.ns_max, self.cond = ns_max, cond
        self.enc = EncoderRDAE(2 * ns_max)
        self.dec = DecoderRDAE(2 * ns_max, extra=2 if cond else 0)
        self.p_cal = None

    def forward(self, x, snr_db, ns, return_parts=False):
        dev, B = x.device, x.shape[0]
        if not torch.is_tensor(ns):
            ns = torch.full((B,), int(ns), dtype=torch.long)
        ns = ns.to(dtype=torch.long, device=dev)
        if not torch.is_tensor(snr_db):
            snr_db = torch.full((B,), float(snr_db))
        snr_db = snr_db.to(dev)
        rel = 10 ** (-snr_db / 20.0)
        feat = rel.view(-1, 1)

        z = normalize_power(self.enc(x, feat), ns)
        sym = qam_map_ste(to_symbols(z))
        if self.p_cal is None:
            # 训练/未标定时：按当前 batch 的实测功率定噪声（与主网络 power_ref=True 同口径），
            # 否则训练用名义功率 1、评估用标定值 0.74，会凭空多出 ~1.3 dB 的信噪比。
            p_ref = tx_power(sym, ns).detach().clamp_min(1e-6)
        else:
            p_ref = torch.tensor([self.p_ref_of(v, s) for v, s in zip(ns.tolist(), snr_db.tolist())],
                                 device=dev)
        noisy, sigma, _ = add_noise(sym, ns, snr_db, p_ref)
        y = from_symbols(noisy) * pair_mask(ns, device=dev)
        if self.cond:
            y = torch.cat([y, feat, (ns.float() / self.ns_max).view(-1, 1)], 1)
        out = self.dec(y, feat)
        if return_parts:
            return out, {"sigma": sigma, "p_tx": tx_power(sym, ns), "ns": ns}
        return out

    @torch.no_grad()
    def calibrate(self, xtr, ns_list=(16, 32, 64), snrs=(0.0, 10.0, 20.0), bs=512):
        """在训练集上标定每个 (Ns, SNR) 的实际发送功率 P_tx。

        修了两处口径问题：
        1. 先 `eval()` 冻结 BN（原实现在 train 模式下会继续更新 running stats）；
        2. 条件特征按**该 SNR** 给（AF 门控依赖 σ/√P，feat=0 会得到与实际运行不同的 P_tx）。
        若同一 Ns 下不同 SNR 的 P_tx 差异 < 0.2%，则合并成一个值（p_cal 变体 `None`），否则按 (Ns, SNR) 存。
        """
        self.eval()
        dev = next(self.parameters()).device
        raw = {}
        for n in ns_list:
            for snr in snrs:
                feat = torch.full((1, 1), 10 ** (-float(snr) / 20.0), device=dev)
                tot, cnt, per_snr = 0.0, 0, []
                for i in range(0, xtr.shape[0], bs):
                    xb = xtr[i:i + bs].to(dev)
                    z = normalize_power(self.enc(xb, feat.expand(xb.shape[0], 1)), int(n))
                    sym = qam_map_ste(to_symbols(z))
                    p = tx_power(sym, torch.full((xb.shape[0],), int(n), device=dev))
                    per_snr.append(float(p.mean()))
                    tot += float(p.sum()); cnt += xb.shape[0]
                raw[(int(n), float(snr))] = (tot / max(cnt, 1), per_snr)

        spread = {}
        for n in ns_list:
            vals = [raw[(int(n), float(s))][0] for s in snrs]
            spread[int(n)] = (max(vals) - min(vals)) / max(min(vals), 1e-9)
        if all(v < 5e-3 for v in spread.values()):          # <0.5%：与 SNR 无关，合并成一个值
            self.p_cal = {int(n): raw[(int(n), float(snrs[0]))][0] for n in ns_list}
            self.p_cal_mode = "ns_only"
        else:
            self.p_cal = {(int(n), float(s)): raw[(int(n), float(s))][0]
                          for n in ns_list for s in snrs}
            self.p_cal_mode = "ns_snr"
        self.p_cal_spread = spread
        self.p_cal_snrs = tuple(float(s) for s in snrs)
        self.p_cal_seen = raw                               # 含逐 batch 明细，便于审计
        return self.p_cal

    def p_ref_of(self, ns, snr_db):
        """取该 (Ns, SNR) 的 P_tx 参考值；兼容两种标定模式。"""
        if self.p_cal is None:
            return None
        if getattr(self, "p_cal_mode", "ns_only") == "ns_snr":
            sni = min(self.p_cal_snrs, key=lambda s: abs(s - float(snr_db)))
            return self.p_cal[(int(ns), float(sni))]
        return self.p_cal[int(ns)]


@torch.no_grad()
def q2_scores(net, x, ns, snr, seed=1234, n=400):
    """Q2：只用 y 的免训练分数 ‖y − F(Ĥ)‖²/n，坏样本 = 该 SNR 最差 20%。

    **与训练链路完全同一条发送路径**：`normalize_power → qam_map_ste → add_noise → pair_mask`。
    （早先的评估版直接 `sigma*randn(tx.shape)`：既漏了 σ/√2，又给未发送位置加了噪声，
      等效 SNR 低 3 dB、且 ns<64 时额外失配——这条已修。）
    返回 (auroc, spearman, mean_nmse)。
    """
    from quality_v5 import auroc, bad_label, spearman

    x = x[:n]
    b, dev = x.shape[0], x.device
    ns_t = torch.full((b,), int(ns), dtype=torch.long, device=dev)
    snr_t = torch.full((b,), float(snr), device=dev)
    feat = (10 ** (-float(snr) / 20.0)) * torch.ones(b, 1, device=dev)

    z0 = normalize_power(net.enc(x, feat), ns_t)
    sym = qam_map_ste(to_symbols(z0))
    p_ref = net.p_ref_of(ns, snr)
    p_ref = torch.ones(b, device=dev) if p_ref is None else torch.full((b,), float(p_ref), device=dev)
    torch.manual_seed(seed)
    noisy, sigma, _ = add_noise(sym, ns_t, snr_t, p_ref)
    y = from_symbols(noisy) * pair_mask(ns_t, device=dev)

    xhat = net.dec(y, feat)
    z1 = normalize_power(net.enc(xhat, feat), ns_t)
    f = from_symbols(qam_map_ste(to_symbols(z1))) * pair_mask(ns_t, device=dev)

    score = ((y - f) ** 2).sum(1).cpu().numpy() / float(ns)
    mse = ((xhat - x) ** 2).flatten(1).mean(1)
    pwr = (x ** 2).flatten(1).mean(1).clamp_min(1e-12)
    nm = (10 * torch.log10(mse / pwr)).cpu().numpy()
    return float(auroc(score, bad_label(nm))), float(spearman(score, nm)), float(nm.mean())


def _smoke():
    torch.manual_seed(0)
    net = LinkRDAE()
    x = torch.rand(4, 2, 32, 32)
    for ns in (16, 32, 64):
        out, parts = net(x, 10.0, ns, True)
        print(f"ns={ns:2d} out={tuple(out.shape)} nmse={nmse_db(out, x):+.3f} dB "
              f"Ptx={float(parts['p_tx'].mean()):.3f} sgcs={sgcs(out, x):.3f} "
              f"sigma={float(parts['sigma'].mean()):.4f}")
    p = sum(q.numel() for q in net.parameters())
    e = sum(q.numel() for q in net.enc.parameters())
    print(f"params total={p:,} enc={e:,} dec={p - e:,}")
    print("calib", {k: round(v, 4) for k, v in net.calibrate(x, (16, 32, 64)).items()})
    out, parts = net(x, 10.0, 64, True)
    print(f"after calib  Ptx={float(parts['p_tx'].mean()):.3f} sigma={float(parts['sigma'].mean()):.4f}")


if __name__ == "__main__":
    _smoke()
