"""Target-condition receiver adaptation with a frozen transmitter.

No claim of a new adaptation algorithm or an upper bound on achievable quality.
All learned arms start at the SAME target-calibrated state and keep BN buffers
fixed, so the comparison measures trainable-parameter scope under one protocol.
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn as nn

from common import (AFModule, OUT, build, changed_state, copy_net, data, dump,
                    evaluate, recalibrate, seed_all)


def choose_parameters(net, mode):
    for p in net.parameters():
        p.requires_grad_(False)
    if mode == 'decoder':
        modules = [net.dec]
    elif mode == 'af':
        modules = [m for m in net.dec.modules() if isinstance(m, AFModule)]
    elif mode == 'bn_affine':
        modules = [m for m in net.dec.modules() if isinstance(m, nn.BatchNorm2d)]
    else:
        raise ValueError(mode)
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]
    return params


def adapt(base, mode, xtr, xval, snr, ns, seed, epochs, lr, bs=128):
    net = copy_net(base)
    initial = copy.deepcopy(net.state_dict())
    params = choose_parameters(net, mode)
    count = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Epoch zero is eligible: selection is entirely on validation CSI/noise.
    best, _ = evaluate(net, xval, snr, ns, repeats=2, noise_seed=62000)
    state = copy.deepcopy(net.state_dict())
    trace = [{'epoch': 0, 'val_nmse_db': best}]
    selected = 0
    for ep in range(epochs):
        net.eval()  # fixed transmitter AND all BN running statistics
        seed_all(72000 + seed * 1000 + ep)
        order = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(9000 + seed * 100 + ep))
        total = 0.0
        for i in range(0, len(xtr), bs):
            xb = xtr[order[i:i + bs]]
            loss = (net(xb, snr, ns) - xb).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert not any(p.grad is not None for p in net.enc.parameters())
            opt.step()
            total += float(loss.detach()) * len(xb)
        scheduler.step()
        value, _ = evaluate(net, xval, snr, ns, repeats=2, noise_seed=62000)
        trace.append({'epoch': ep + 1, 'train_mse': total / len(xtr), 'val_nmse_db': value})
        if value < best:
            best, selected = value, ep + 1
            state = copy.deepcopy(net.state_dict())
        if ep == 0 or (ep + 1) % 4 == 0 or ep == epochs - 1:
            print(f'  {mode} seed={seed} snr={snr} ns={ns} ep={ep+1} val={value:.4f} best={best:.4f}', flush=True)
    net.load_state_dict(state)
    assert not changed_state(initial, state, 'enc.'), 'Transmitter changed'
    for key in initial:
        if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            assert torch.equal(initial[key], state[key]), f'BN buffer changed: {key}'
    assert net.p_cal == base.p_cal
    actual = sum(not torch.equal(initial[k], state[k]) for k in initial)
    return net, {'trainable_parameters': count, 'extra_fp32_bytes_per_condition': 4 * count,
                 'selected_epoch': selected, 'selected_val_nmse_db': best,
                 'changed_state_tensors': actual, 'trace': trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--ntrain', type=int, default=2048)
    parser.add_argument('--nval', type=int, default=400)
    parser.add_argument('--ntest', type=int, default=400)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--decoder-lrs', type=float, nargs='+', default=[3e-4, 3e-5])
    parser.add_argument('--lengths', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--out', default='q1_results.json')
    args = parser.parse_args()
    output = OUT / args.out
    xtr, xval, xte = data()
    result = {'protocol': vars(args), 'scope': 'RD-AE adaptation, not official end-to-end RD-JSCC',
              'bn_policy': 'All learned arms start at S2; all BN buffers fixed during adaptation',
              'selection': 'validation only, epoch 0 eligible', 'cells': []}
    for seed in args.seeds:
        started = time.perf_counter()
        original, meta = build(seed, xtr)
        mixed = copy_net(original)
        recalibrate(mixed, xtr, seed=51000 + seed)
        for ns in args.lengths:
            snr = 20.0
            target = copy_net(original)
            recalibrate(target, xtr, snr=snr, ns=ns, seed=52000 + seed)
            row = {'seed': seed, 'snr_db': snr, 'ns': ns, 'source': meta, 'arms': {}}
            for label, net in [('S0', original), ('S1', mixed), ('S2', target)]:
                value, _ = evaluate(net, xte[:args.ntest], snr, ns)
                row['arms'][label] = {'test_nmse_db': value, 'trainable_parameters': 0}
            state_dir = OUT / 'q1_states'
            state_dir.mkdir(exist_ok=True)
            torch.save(target.state_dict(), state_dir / f's{seed}_ns{ns}_S2.pt')
            print(f'BASE seed={seed} ns={ns}: '+str({k:round(v["test_nmse_db"],4) for k,v in row["arms"].items()}), flush=True)
            for mode in ['bn_affine', 'af', 'decoder']:
                rates = args.decoder_lrs if mode == 'decoder' else [args.lr]
                candidates = []
                for rate in rates:
                    net_i, info_i = adapt(target, mode, xtr[:args.ntrain], xval[:args.nval],
                                         snr, ns, seed, args.epochs, rate)
                    info_i['learning_rate'] = rate
                    candidates.append((net_i, info_i))
                net, info = min(candidates, key=lambda pair: pair[1]['selected_val_nmse_db'])
                info['optimizer_trials'] = [{'learning_rate': c['learning_rate'],
                                            'selected_val_nmse_db': c['selected_val_nmse_db'],
                                            'selected_epoch': c['selected_epoch']} for _,c in candidates]
                value, losses = evaluate(net, xte[:args.ntest], snr, ns)
                info['test_nmse_db'] = value
                info['gain_over_S2_db'] = row['arms']['S2']['test_nmse_db'] - value
                info['duration_since_seed_start_s'] = time.perf_counter() - started
                row['arms'][mode] = info
                torch.save(net.state_dict(), state_dir / f's{seed}_ns{ns}_{mode}.pt')
                print(f'RESULT seed={seed} ns={ns} {mode}: {value:.4f}, gain_over_S2={info["gain_over_S2_db"]:.4f}, params={info["trainable_parameters"]}', flush=True)
            result['cells'].append(row)
            dump(output, result)
        print(f'SEED {seed} complete in {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
