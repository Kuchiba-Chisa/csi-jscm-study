"""反馈链路的统一实现（编码器/解码器/星座/信道/指标）。

所有方案都用 STE 端到端训练，星座均为一维实值 16-PAM（不是 QAM），没有信道编码。

Scheme（旧名仍可用，见 ALIASES）:
    analog            : 无星座约束，连续符号直接过 AWGN（参考上限）
    fixed_hard        : 固定均匀 16-PAM + 接收端硬判决        （旧名 sscc；它不是分离式系统）
    learned_hard      : 可学习 16-PAM + 接收端硬判决           （旧名 lc_hard）
    learned_soft      : 可学习 16-PAM + 解码器直接吃带噪观测   （旧名 jscm）
    learned_soft_snr  : 同上，解码器额外输入噪声水平           （旧名 jscm_snr）

SNR 定义（power_ref=True，默认）：SNR = E[|s|^2] / sigma^2，E[|s|^2] 是星座映射之后
实际发送符号的平均功率（按 batch 统计）。旧实现按 z 的功率（恒为 1）定噪声，量化后
各方案实际功率不同（0.85~0.94），导致 SNR 不可比；power_ref=False 可复现旧结果。
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 网络
class Encoder(nn.Module):
    def __init__(self, ns):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 8 * 8, ns)

    def forward(self, x):
        z = self.fc(self.body(x).flatten(1))
        return torch.tanh(z) * 3.0          # 有界输出 [-3, 3]，便于均匀量化


class Decoder(nn.Module):
    def __init__(self, ns, extra=0):
        super().__init__()
        self.fc = nn.Linear(ns + extra, 128 * 8 * 8)
        self.body = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, 1, 1),
        )

    def forward(self, y):
        h = self.fc(y).view(-1, 128, 8, 8)
        return self.body(h)


# ---------------------------------------------------------------- 星座
class CsiNetEnc(nn.Module):
    """CsiNet 风格的编码网络（文献标准骨干，用于结论的外部效度复核）。

    与 ours 的区别：3x3 卷积 + BN + LeakyReLU(0.3)、最大池化式下采样，
    末端 FC 直接输出 Ns 个实数（仍接同一个量化器/星座，便于对照）。
    """

    def __init__(self, ns):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3, inplace=True),
        )
        self.fc = nn.Linear(16 * 8 * 8, ns)

    def forward(self, x):
        return torch.tanh(self.fc(self.body(x).flatten(1))) * 3.0


def _refine(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.3, inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.3, inplace=True),
    )


class CsiNetDec(nn.Module):
    """CsiNet 风格的 RefineNet 解码器：FC -> 4x8x8 -> ConvT x2，无 sigmoid。"""

    def __init__(self, ns, extra=0):
        super().__init__()
        self.fc = nn.Linear(ns + extra, 16 * 8 * 8)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.3, inplace=True))
        self.ref1 = _refine(32, 32)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.3, inplace=True))
        self.ref2 = _refine(32, 32)
        self.out = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, y):
        h = self.fc(y).view(-1, 16, 8, 8)
        return self.out(self.ref2(self.up2(self.ref1(self.up1(h)))))


BACKBONE = os.environ.get("JSCM_BACKBONE", "ours")


class LearnedConstellation(nn.Module):
    """可学习 PAM 星座点：位置可训练，功率归一化到 1。"""

    def __init__(self, m=16):
        super().__init__()
        init = torch.linspace(-1, 1, m)
        self.points = nn.Parameter(init.clone())
        self.m = m

    def get_points(self):
        p = self.points.float()
        p = p - p.mean()
        return p / (p.pow(2).mean().sqrt() + 1e-8)


def uniform_pam(m=16):
    """固定均匀 PAM：[-3,3] 等间隔，功率归一化到 1。"""
    p = torch.linspace(-3, 3, m)
    return p / (p.pow(2).mean().sqrt() + 1e-8)


# ---------------------------------------------------------------- 收发机
ALIASES = {"sscc": "fixed_hard", "lc_hard": "learned_hard",
           "jscm": "learned_soft", "jscm_snr": "learned_soft_snr"}
BASES = ("analog", "fixed_hard", "learned_hard", "learned_soft", "learned_soft_snr")
HARD_RX = ("fixed_hard", "learned_hard")


def canon(scheme):
    """旧名 -> 新名；新名原样返回。"""
    return ALIASES.get(scheme, scheme)


class Link(nn.Module):
    """完整反馈链路。scheme 决定星座/接收端处理方式。"""

    def __init__(self, scheme, ns=64, m=16, power_ref=True, backbone=None):
        super().__init__()
        scheme = canon(scheme)
        assert scheme in BASES, scheme
        self.scheme = scheme
        self.ns = ns
        self.m = m
        self.power_ref = power_ref
        self.last_tx_power = 1.0
        # 评估用的冻结功率：在训练集上标定后写入，None 表示按 batch 估计（训练时）
        self.p_cal = None
        bb = backbone or os.environ.get("JSCM_BACKBONE", "ours")
        if bb == "csinet":
            self.enc = CsiNetEnc(ns)
            self.dec = CsiNetDec(ns, extra=1 if scheme == "learned_soft_snr" else 0)
        else:
            self.enc = Encoder(ns)
            self.dec = Decoder(ns, extra=1 if scheme == "learned_soft_snr" else 0)
        if scheme in ("learned_hard", "learned_soft", "learned_soft_snr"):
            self.const = LearnedConstellation(m)
        else:
            self.register_buffer("const", uniform_pam(m), persistent=False)

    # ---- 发送端
    def modulate(self, z, tau=0.05):
        if self.scheme == "analog":
            return z, None                                        # 模拟：直接发连续值
        pts = self.const.get_points() if isinstance(self.const, LearnedConstellation) else self.const
        d = (z.unsqueeze(-1) - pts.view(1, 1, -1)).abs()          # (B,Ns,M)
        idx = d.argmin(-1)
        hard = pts[idx]
        # 直通估计（STE）：前向发硬量化符号，反向把梯度按恒等近似传回编码器。
        # 早先试过「用软加权代替」的做法，温度小时梯度被压到接近 0，
        # 编码器会在「输出恒为零、所有符号落到同一层」的退化解上停几十个 epoch。
        return hard + (z - z.detach()), idx

    # ---- 接收端
    def receive(self, y, idx, tau=0.05):
        if self.scheme in HARD_RX:
            pts = self.const.get_points() if isinstance(self.const, LearnedConstellation) else self.const
            d = (y.unsqueeze(-1) - pts.view(1, 1, -1)).abs()       # 硬判决，丢弃软信息
            hard = pts[d.argmin(-1)]
            # 前向严格是硬判决；反向同样用 STE 按恒等传梯度，
            # 否则整条链路对编码器不可导，编码器会停在初始化状态（那不是「分离式」，是「没训练」）。
            return hard + (y - y.detach())
        return y                                                   # 模拟/JSCM：软信息直通

    def forward(self, x, snr_db):
        z = self.enc(x)
        # 逐样本功率控制：把每个样本的符号能量归一化到 1（等价于 UE 侧 AGC）。
        # 没有这一步时，编码器初始输出幅度很小、所有符号落进同一个量化层，
        # 链路前向退化成常数，要几十个 epoch 才能逃出来。
        z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
        tx, idx = self.modulate(z)
        if torch.is_tensor(snr_db) and snr_db.dim() > 0:
            # B2：逐样本 SNR（形状 (B,) -> (B,1)），标量路径数值完全不变
            rel = 10 ** (-snr_db.view(-1, 1) / 20.0)
        else:
            rel = 10 ** (-snr_db / 20.0)                 # sigma / sqrt(信号功率)
        p_tx = tx.detach().pow(2).mean()                 # 星座映射后的实际平均发送功率
        self.last_tx_power = float(p_tx)
        p_ref = p_tx if self.p_cal is None else self.p_cal
        sigma = rel * (p_ref ** 0.5) if self.power_ref else rel
        y = tx + sigma * torch.randn_like(tx)
        y = self.receive(y, idx)
        if self.scheme == "learned_soft_snr":
            # 把噪声水平（相对信号功率）告诉解码器
            if torch.is_tensor(rel) and rel.dim() > 0:
                s_t = rel.to(dtype=y.dtype, device=y.device)
            else:
                s_t = torch.full((y.shape[0], 1), float(rel), dtype=y.dtype,
                                 device=y.device)
            y = torch.cat([y, s_t], dim=1)
        return self.dec(y)


# ---------------------------------------------------------------- 功率标定
@torch.no_grad()
def calibrate(net, xtr, bs=512):
    """在训练集上标定全局平均发送功率 P_tx，写入 net.p_cal（冻结用）。

    SNR 定义为 P_tx / sigma^2，P_tx 取星座映射后的实际平均功率。
    评估时用这个全局值确定 sigma，避免测试 batch 之间的功率波动。
    """
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(xtr), bs):
        x = xtr[i:i + bs]
        z = net.enc(x)
        z = z / (z.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
        tx, _ = net.modulate(z)
        tot += float(tx.pow(2).mean()) * len(x)
        n += len(x)
    net.p_cal = tot / max(n, 1)
    return net.p_cal


# ---------------------------------------------------------------- 指标
def nmse_db(pred, target):
    return 10 * torch.log10(((pred - target).pow(2).sum((1, 2, 3)) /
                             target.pow(2).sum((1, 2, 3))).mean())


def sgcs(pred, target):
    """标准 SGCS：逐子载波的复向量余弦相似度 |h^H h_hat| / (|h||h_hat|)，再对子载波和样本取均值。

    张量布局 (B, 2[实/虚], Nt[天线], Nc[子载波])，频域 CSI。
    取模长，所以对每个子载波上的公共相位不敏感（预编码只关心方向）。
    """
    hp = torch.complex(pred[:, 0], pred[:, 1])      # (B, Nt, Nc)
    ht = torch.complex(target[:, 0], target[:, 1])
    num = (ht.conj() * hp).sum(1).abs()             # (B, Nc)
    den = ht.abs().pow(2).sum(1).sqrt() * hp.abs().pow(2).sum(1).sqrt() + 1e-12
    return (num / den).mean()


def cos_flat(pred, target):
    """旧版"SGCS"：把整个实值张量拉平后求余弦。不是标准 SGCS，仅用于与旧结果对照。"""
    a = pred.reshape(pred.shape[0], -1)
    b = target.reshape(target.shape[0], -1)
    num = (a * b).sum(1)
    den = a.norm(dim=1) * b.norm(dim=1) + 1e-8
    return (num / den).mean()
