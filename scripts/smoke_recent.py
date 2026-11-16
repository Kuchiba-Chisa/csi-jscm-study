"""Weight-free checks for the recent Q1 experiment paths and math."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation'))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    sampling = module('recent_sampling', 'experiments/rdae_ns_sampling_20261115/run_sampling.py')
    x = torch.randn(4, 2, 32, 32) / 3
    net = sampling.LinkRDAE().eval()
    ns = torch.tensor([16, 32, 64, 16])
    snr = torch.tensor([0., 5., 10., 20.])
    torch.manual_seed(900)
    expected = net(x, snr, ns)
    torch.manual_seed(900)
    actual = sampling.forward(net, x, snr, ns, torch.randn(4, 64, 2))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    (actual-x).square().mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
               for m in (net.enc, net.dec))
    for epoch in (0, 1):
        _, a, b, _, _ = sampling.epoch_schedule(epoch, 6000)
        for length in sampling.LENGTHS:
            assert (a == length).sum() == (b == length).sum()
        assert all(torch.unique(row).numel() == 1 for row in a)
        assert any(torch.unique(row).numel() > 1 for row in b)
    ft = module('recent_finetune', 'experiments/rdae_distill_finetune_20261115/run_finetune.py')
    assert ft.SOURCE == ROOT / 'experiments/rdae_q1_supplement'
    from affine_calib import fit_affine
    pred = torch.randn(7, 2, 32, 32)
    a, b = fit_affine(pred, 0.65 * pred + 0.02)
    assert abs(a - .65) < 1e-6 and abs(b - .02) < 1e-6
    from quality_v5 import sgcs_per
    target = torch.randn_like(pred)
    torch.testing.assert_close(sgcs_per(pred, target), sgcs_per(.65 * pred, target), atol=3e-7, rtol=1e-5)
    gn = module('recent_gn', 'experiments/gn_sharing_gap_20261116/run_gn_gap.py')
    checks = gn.validate_forward(gn.LinkRDAE_GN(cond=False).eval(), x)
    print(json.dumps({'status': 'PASS', 'mixed_length_forward_and_gradients': True,
                      'per_epoch_length_counts_match': True, 'finetune_dependency_resolves': True,
                      'affine_fit_and_SGCS_invariance': True, 'gn_forward': checks}, indent=2))


if __name__ == '__main__':
    main()
