# -*- coding: utf-8 -*-
"""v4 链路：复数 16-QAM + 前缀截断的多码率 + 有限星座 JSCM。

与 v3 的区别（全部写进结果文档的"改动列表"）：
  * 星座：复数 16-QAM（Gray），固定不学；v3 是一维实 16-PAM（可学）。
  * 码率：前缀截断 Ns ∈ {16, 64} 个复符号（64 / 256 bit），未发送位置严格置 0。
  * 功率：只对有效符号做逐样本归一化，P_tx(Ns) 在训练集上标定并冻结。
  * 骨干：CsiNet 风格（RefineNet 带残差相加），32×32 → 128×8×8 → 2·Ns。

执行顺序严格按 v4 文档 §2.2：
  z(2*NS_MAX) → 取前缀 → 逐样本功率归一化 → 只用有效符号做 QAM+STE
  → 只对有效符号加 AWGN → 组装（未发送位置置 0）→ 解码器
复数 16-QAM 没有零星座点，所以绝不能"先补零再整段量化"。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NS_MAX = 64                      # 最多 64 个复符号 = 128 个实维度
M = 16
BITS = {16: 64, 64: 256}         # Ns -> 标签比特数（log2(16)=4）
LEVELS = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / math.sqrt(10.0)   # 平均功率 1
# Gray 映射：符号序号 k -> (I 的 2 bit, Q 的 2 bit)，相邻序号只差 1 bit
GRAY2 = [(0, 0), (0, 1), (1, 1), (1, 0)]
LEVEL_LABEL = [0, 1, 3, 2]        # 4 个电平的 2-bit Gray 标签（相邻只差 1 bit）


# ---------------------------------------------------------------- 星座与掩码
def qam16_points():
    """16 个复星座点（返回 (16,) complex）。"""
    pts = []
    for i in range(4):
        for q in range(4):
            pts.append(complex(LEVELS[i].item(), LEVELS[q].item()))
    return torch.tensor(pts, dtype=torch.complex64)


def qam16_gray_table():
    """k -> (level_index_I, level_index_Q)，Gray 编码。"""
    tab = []
    for k in range(16):
        hi, lo = k >> 2, k & 3
        tab.append((GRAY2[hi][0] * 2 + GRAY2[hi][1], GRAY2[lo][0] * 2 + GRAY2[lo][1]))
    return tab


def _nearest_level(v):
    """每个实数就近量化到 {-3,-1,1,3}/√10；返回量化值与索引。"""
    idx = (v.unsqueeze(-1) - LEVELS.to(v.device)).abs().argmin(-1)
    return LEVELS.to(v.device)[idx], idx


def qam_map_ste(z):
    """z: (B, ..., 2) 末维是 (re, im) 的未量化符号 → 硬映射 + 直通梯度。"""
    re, qi = _nearest_level(z[..., 0])
    im, qq = _nearest_level(z[..., 1])
    hard = torch.stack([re, im], dim=-1)
    return z + (hard - z).detach()          # 前向严格硬映射，反向恒等


def pair_mask(ns, ns_max=NS_MAX, device=None):
    """ns: int 或 (B,) LongTensor → (B, 2*ns_max) 的 0/1 掩码（re/im 成对）。"""
    if not torch.is_tensor(ns):
        ns = torch.full((1,), int(ns), dtype=torch.long)
    ns = ns.to(dtype=torch.long)
    pos = torch.arange(ns_max, device=ns.device).unsqueeze(0)          # (1, ns_max)
    m = (pos < ns.unsqueeze(1)).to(torch.float32)                      # (B, ns_max)
    return m.repeat_interleave(2, dim=1)                               # (B, 2*ns_max)


def to_symbols(v):
    """(B, 2*ns_max) → (B, ns_max, 2)。"""
    return v.view(v.shape[0], -1, 2)


def from_symbols(s):
    """(B, ns_max, 2) → (B, 2*ns_max)。"""
    return s.reshape(s.shape[0], -1)


def normalize_power(z, ns, eps=1e-8):
    """只对有效符号做逐样本功率归一化，使 E|s|² = 1。ns 可为 int。"""
    if not torch.is_tensor(ns):
        ns = torch.full((z.shape[0],), int(ns), dtype=torch.long, device=z.device)
    else:
        ns = ns.to(z.device)
    mask = pair_mask(ns, device=z.device)
    zv = z * mask
    e = (zv ** 2).sum(1, keepdim=True)               # 有效符号的总能量（实+虚）
    e = e.clamp_min(eps)
    scale = (ns.to(z.device).float().view(-1, 1) / e).sqrt()
    return z * scale


def tx_power(sym, ns):
    """有效符号的平均功率 E|s|²，返回 (B,)。ns 可为 int。"""
    if not torch.is_tensor(ns):
        ns = torch.full((sym.shape[0],), int(ns), dtype=torch.long, device=sym.device)
    else:
        ns = ns.to(sym.device)
    mask = pair_mask(ns, device=sym.device).view(sym.shape[0], -1, 2)
    e = (sym ** 2 * mask).sum((1, 2))
    return e / ns.to(sym.device).float()


def add_noise(sym, ns, snr_db, p_ref):
    """只在有效符号上加复高斯噪声，σ² = P_tx / SNR；returns (noisy, sigma, measured)。"""
    if not torch.is_tensor(ns):
        ns = torch.full((sym.shape[0],), int(ns), dtype=torch.long, device=sym.device)
    else:
        ns = ns.to(sym.device)
    if not torch.is_tensor(snr_db):
        snr_db = torch.full((sym.shape[0],), float(snr_db))
    mask = pair_mask(ns, device=sym.device).view(sym.shape[0], -1, 2)
    rel = 10 ** (-snr_db.to(sym.device).float() / 20.0)         # (B,) σ/√P
    sigma = rel * (p_ref ** 0.5)
    # 复噪声：E|n|² = σ²，实部/虚部各 σ²/2（v4 §3.1 第 4 条）
    n = torch.randn_like(sym) * (sigma / math.sqrt(2.0)).view(-1, 1, 1)
    return sym + n * mask, sigma, n * mask


# ---------------------------------------------------------------- 骨干
def _cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.3, inplace=True),
    )


class RefineNet(nn.Module):
    """CsiNet 的 RefineNet：Conv-ReLU-Conv 后**残差相加**（v3 那个别名版本漏了相加）。"""

    def __init__(self, ch=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


class EncoderV4(nn.Module):
    """2×32×32 → 128×8×8 → FC → 2·NS_MAX（未量化、未归一化）。"""

    def __init__(self, out_dim=2 * NS_MAX):
        super().__init__()
        self.body = nn.Sequential(
            _cbr(2, 16, 1),        # 32×32
            _cbr(16, 32, 2),       # 16×16
            _cbr(32, 64, 2),       # 8×8
            _cbr(64, 128, 1),      # 8×8
        )
        self.fc = nn.Linear(128 * 8 * 8, out_dim)

    def forward(self, x):
        h = self.body(x).flatten(1)
        return self.fc(h)


class DecoderV4(nn.Module):
    """(2·NS_MAX [+2]) → 128×8×8 → 转置卷积到 32×32 → 2 个 RefineNet → 2 通道。"""

    def __init__(self, in_dim=2 * NS_MAX, cond=0):
        super().__init__()
        self.fc = nn.Linear(in_dim + cond, 128 * 8 * 8)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.3, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.3, True),
        )
        self.ref1, self.ref2 = RefineNet(32), RefineNet(32)
        self.out = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, y):
        h = self.fc(y).view(-1, 128, 8, 8)
        h = self.ref2(self.ref1(self.up(h)))
        return self.out(h)


class LinkV4(nn.Module):
    """完整链路。ns 可为 int（全 batch 同一码率）或 (B,) 长张量（逐样本码率）。"""

    def __init__(self, cond=False, ns_max=NS_MAX):
        super().__init__()
        self.ns_max = ns_max
        self.enc = EncoderV4(2 * ns_max)
        self.dec = DecoderV4(2 * ns_max, cond=2 if cond else 0)
        self.cond = cond
        self.p_cal = None          # {Ns: float}，训练集标定后冻结

    # ---- 前向 ----------------------------------------------------
    def forward(self, x, snr_db, ns, return_parts=False):
        dev = x.device
        B = x.shape[0]
        if not torch.is_tensor(ns):
            ns = torch.full((B,), int(ns), dtype=torch.long)
        ns = ns.to(dtype=torch.long, device=dev)
        if not torch.is_tensor(snr_db):
            snr_db = torch.full((B,), float(snr_db))
        snr_db = snr_db.to(dev)

        z = self.enc(x)                                  # (B,2*ns_max)
        z = normalize_power(z, ns)                       # 只用有效符号
        sym = qam_map_ste(to_symbols(z))                 # 硬映射 + STE
        p_ref = self._p_ref(ns, dev)                     # (B,)
        noisy, sigma, _ = add_noise(sym, ns, snr_db, p_ref)
        y = from_symbols(noisy) * pair_mask(ns, device=dev)     # 未发送位置严格 0
        if self.cond:
            yc = torch.stack([10 ** (-snr_db / 20.0), ns.float() / self.ns_max], 1)
            y = torch.cat([y, yc], 1)
        out = self.dec(y)
        if return_parts:
            return out, {"sigma": sigma, "p_tx": tx_power(sym, ns), "ns": ns}
        return out

    def _p_ref(self, ns, dev):
        if self.p_cal is None:
            return torch.ones(ns.shape[0], device=dev)   # 训练早期按 1 处理；标定后覆盖
        return torch.tensor([self.p_cal[int(v)] for v in ns.tolist()], device=dev)

    @torch.no_grad()
    def calibrate(self, xtr, ns_list=(16, 64), snr_db=60.0, bs=512):
        """在训练集上标定每个 Ns 的 P_tx（无噪声），写入 self.p_cal 并冻结。"""
        tot = {int(n): 0.0 for n in ns_list}
        cnt = {int(n): 0 for n in ns_list}
        for n in ns_list:
            for i in range(0, xtr.shape[0], bs):
                xb = xtr[i:i + bs].to(next(self.parameters()).device)
                z = normalize_power(self.enc(xb), int(n))
                sym = qam_map_ste(to_symbols(z))
                p = tx_power(sym, torch.full((xb.shape[0],), int(n)))
                tot[int(n)] += float(p.sum()); cnt[int(n)] += xb.shape[0]
        self.p_cal = {k: tot[k] / max(cnt[k], 1) for k in tot}
        return self.p_cal


# ---------------------------------------------------------------- 指标
def complex_from_pairs(x):
    """(B,2,32,32) 的 angle-delay 表示 → 复数 (B,32,32)。"""
    return torch.complex(x[:, 0], x[:, 1])


def to_freq(h, nc=32):
    """angle-delay 复数 (B,Nt,Nl) → 频域 (B,Nc,Nt)，沿时延维做 FFT。"""
    return torch.fft.fft(h, n=nc, dim=-1).transpose(1, 2)


def nmse_db(pred, x):
    """逐样本 NMSE，再取 batch 平均（dB）。"""
    num = ((pred - x) ** 2).sum((1, 2, 3))
    den = (x ** 2).sum((1, 2, 3)).clamp_min(1e-12)
    return float((10 * torch.log10(num / den + 1e-12)).mean())


def sgcs(pred, x, nc=32):
    """逐子载波平方余弦（1 = 方向完全一致）。"""
    hf = to_freq(complex_from_pairs(pred), nc)      # (B,Nc,Nt)
    ht = to_freq(complex_from_pairs(x), nc)
    num = (hf.conj() * ht).sum(-1).abs() ** 2
    den = (hf.abs() ** 2).sum(-1) * (ht.abs() ** 2).sum(-1)
    return float((num / den.clamp_min(1e-12)).mean())
