"""Read-only access to the existing project; all new outputs stay beside this file."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parents[2] / 'simulation'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from model_rdae import AFModule, LinkRDAE  # noqa: E402
from quality_v5 import load_split, nmse_db_per, summarize  # noqa: E402

torch.set_num_threads(8)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data():
    return tuple(load_split(s, root=str(PROJECT / 'data')) / 3.0 for s in ('train', 'val', 'test'))


def build(seed, xtr):
    path = PROJECT / 'ckpt_rdae' / f'rdae_sample_s{seed}.pt'
    net = LinkRDAE(cond=False)
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    net.eval()
    # Source-domain data only; once per source model, then identical for every arm.
    seed_all(48000 + seed)
    net.calibrate(xtr[:2000], ns_list=(16, 32, 64), snrs=(0, 5, 10, 15, 20))
    return net, {'checkpoint': str(path), 'sha256': sha(path), 'p_cal': repr(net.p_cal),
                 'p_cal_mode': net.p_cal_mode, 'source_model_sha256': sha(PROJECT / 'model_rdae.py')}


def decoder_bns(net):
    return [m for m in net.dec.modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def recalibrate(net, xtr, snr=None, ns=None, seed=0, bs=128):
    """Match the existing cumulative BN protocol; one full training-data pass.

    All decoder BNs use batch statistics during this pass. This is an empirical
    calibration protocol, not an exact layerwise population-moment calculation.
    """
    net.eval()
    for m in decoder_bns(net):
        m.reset_running_stats()
        m.momentum = None
        m.train()
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(xtr), generator=generator)
    pool = torch.tensor([16, 32, 64])
    for i in range(0, len(xtr), bs):
        xb = xtr[permutation[i:i + bs]]
        n = len(xb)
        ss = torch.rand(n, generator=generator) * 20 if snr is None else torch.full((n,), float(snr))
        nn_ = pool[torch.randint(3, (n,), generator=generator)] if ns is None else torch.full((n,), int(ns))
        net(xb, ss, nn_)
    net.eval()
    for m in decoder_bns(net):
        m.momentum = 0.1


@torch.no_grad()
def evaluate(net, x, snr, ns, repeats=3, noise_seed=81000, bs=128):
    net.eval()
    losses = []
    for repeat in range(repeats):
        parts = []
        for i in range(0, len(x), bs):
            seed_all(noise_seed + repeat * 1000 + i)
            xb = x[i:i + bs]
            parts.append(nmse_db_per(net(xb, snr, ns), xb).cpu())
        losses.append(torch.cat(parts))
    losses = torch.stack(losses)
    return float(losses.mean()), losses.numpy()


def copy_net(net):
    return copy.deepcopy(net)


def summarize_predictions(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    result = summarize(pred, target, 'quality_regression')
    result['mae_db'] = float(np.abs(pred - target).mean())
    result['bias_db'] = float((pred - target).mean())
    return result


def changed_state(original, updated, prefix):
    return [k for k in original if k.startswith(prefix) and not torch.equal(original[k], updated[k])]
