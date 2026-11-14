"""Prospective follow-up: larger receiver-only refiner, uniform vs tail weighting.

All model selection uses train/validation. Final evaluation uses test H600:1000
and new noise draws, withheld from the previous refinement experiment. These
H are not claimed to be unseen throughout the entire historical project.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / 'q2_selective_refinement_20261113'))
from run_refinement import (build, data, dump, sha, seed_all, decode_maps,
                            score_head, predict, model_input, cluster_ci,
                            nmse_db_per, sgcs_per, PREV,
                            normalize_power, to_symbols, qam_map_ste, from_symbols)

SNRS = (0, 10, 20)
VARIANTS = ('uniform', 'tail_weighted')


class Refiner(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.y_projection = nn.Linear(128, 4 * 8 * 8)
        self.c1 = nn.Conv2d(24, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.c3 = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, spatial, y):
        local = F.interpolate(self.y_projection(y).view(-1, 4, 8, 8),
                              size=(32, 32), mode='bilinear', align_corners=False)
        return self.c3(F.silu(self.c2(F.silu(self.c1(torch.cat([spatial, local], 1))))))


def costs(width=16):
    net = Refiner(width)
    return dict(parameters=sum(p.numel() for p in net.parameters()),
                weights_bytes=sum(p.numel() * 4 for p in net.parameters()),
                refiner_MACs=128 * 256 + 32 * 32 * 9 * (24 * width + width * width + width * 2),
                monitor_MACs=24638, base_decoder_MACs=22340048,
                caveat='Conv/linear MACs exclude normalization, activation, resize and memory traffic; actual latency separately measured.')


@torch.no_grad()
def make_cache(net, x, seed, mode):
    """One balanced condition/H for training; three fresh noises/H for final test."""
    ys, ss, hs = [], [], []
    if mode == 'train':
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(213000 + seed))
        groups = [(s, order[j::3], 1) for j, s in enumerate(SNRS)]
        noise_base = 1510000000 + seed * 10000000
    else:
        groups = [(s, torch.arange(len(x)), 3) for s in SNRS]
        noise_base = 1710000000 + seed * 10000000
    for s, indices, repeats in groups:
        for start in range(0, len(indices), 128):
            ids = indices[start:start + 128]
            xb = x[ids]
            af = torch.full((len(xb), 1), 10 ** (-s / 20))
            latent = normalize_power(net.enc(xb, af), 64)
            tx = from_symbols(qam_map_ste(to_symbols(latent)))
            sigma = math.sqrt(net.p_ref_of(64, s) / 2) * 10 ** (-s / 20)
            for repeat in range(repeats):
                g = torch.Generator().manual_seed(noise_base + s * 10000 + repeat * 100000 + start)
                ys.append(tx + torch.randn(tx.shape, generator=g) * sigma)
                ss.append(torch.full((len(xb),), float(s)))
                hs.append(ids)
    y, snr, hid = torch.cat(ys), torch.cat(ss), torch.cat(hs)
    out = decode_maps(net, y, snr, x[hid])
    out.update(y=y, snr=snr, h_index=hid)
    return out


def subset(cache, mask):
    return {k: v[mask] for k, v in cache.items()}


def head_inputs(cache):
    features = F.adaptive_avg_pool2d(cache['maps'][:, 2:], (2, 2)).flatten(1)
    condition = torch.stack([cache['snr'] / 20, torch.ones_like(cache['snr'])], 1)
    return torch.cat([features, condition], 1)


def normalizers(cache):
    return dict(mean=cache['maps'].mean((0, 2, 3), keepdim=True),
                sd=cache['maps'].std((0, 2, 3), keepdim=True).clamp_min(1e-4),
                y_mean=cache['y'].mean(0, keepdim=True),
                y_sd=cache['y'].std(0, keepdim=True).clamp_min(1e-4))


def output(model, cache, norm, ix):
    spatial = model_input(cache['maps'][ix], cache['snr'][ix], norm['mean'], norm['sd'])
    y = (cache['y'][ix] - norm['y_mean']) / norm['y_sd']
    return cache['base'][ix] + model(spatial, y)


@torch.no_grad()
def evaluate(model, cache, norm):
    loss, sgcs = [], []
    for ix in torch.arange(len(cache['snr'])).split(128):
        rec = output(model, cache, norm, ix)
        loss.append(nmse_db_per(rec, cache['target'][ix]))
        sgcs.append(sgcs_per(rec, cache['target'][ix]))
    return dict(loss=torch.cat(loss), sgcs=torch.cat(sgcs))


def training_weights(cache, quality, train_cuts, variant):
    weights = torch.ones(len(quality))
    if variant == 'tail_weighted':
        for s, cut in train_cuts.items():
            # Same fixed detector at train and deployment; no true-quality gate.
            weights[(cache['snr'] == s) & (quality >= cut)] = 5.
    return weights


def train_one(train, valid, quality, seed, variant, norm, args):
    seed_all(421000 + seed)
    model = Refiner(args.width)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    cuts = {s: float(torch.quantile(quality['train'][train['snr'] == s], .6)) for s in SNRS}
    tw = training_weights(train, quality['train'], cuts, variant)
    vw = training_weights(valid, quality['select'], cuts, variant)
    # Identity is an eligible model. The same weighted loss selects epoch;
    # policy comparisons and test outcomes do not choose epoch.
    best = float((valid['base_loss'] * vw).sum() / vw.sum())
    state, selected, stale = copy.deepcopy(model.state_dict()), 0, 0
    history = []
    gen = torch.Generator().manual_seed(422000 + seed)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for ix in torch.randperm(len(train['snr']), generator=gen).split(128):
            rec = output(model, train, norm, ix)
            # Directly match the reported mean per-observation NMSE_dB metric.
            # Weight mean is fixed across batches, rather than renormalized per batch.
            loss = (nmse_db_per(rec, train['target'][ix]) * tw[ix]).mean() / tw.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        schedule.step()
        model.eval()
        vout = evaluate(model, valid, norm)
        objective = float((vout['loss'] * vw).sum() / vw.sum())
        history.append(dict(epoch=epoch + 1, train_objective=total / len(train['snr']),
                            validation_objective=objective,
                            validation_mean_nmse_db=float(vout['loss'].mean())))
        if objective < best - 1e-4:
            state, best, selected, stale = copy.deepcopy(model.state_dict()), objective, epoch + 1, 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('TRAIN', seed, variant, epoch + 1, 'objective', round(objective, 5),
                  'selected', selected, 'seconds', round(time.perf_counter() - started, 1), flush=True)
        if stale >= 8:
            break
    model.load_state_dict(state)
    model.eval()
    meta = dict(selected_epoch=selected, trace=history, seconds=time.perf_counter() - started,
                train_predicted_worst40_thresholds=cuts,
                weights='1 everywhere; tail_weighted assigns 5 to predicted worst40% within each training SNR',
                epoch_selection='weighted validation mean NMSE_dB with training-derived quality cuts; no test selection')
    torch.save(dict(state_dict=state, norm=norm, width=args.width, meta=meta),
               OUT / f'{variant}_s{seed}.pt')
    return model, meta


def policies(cache, quality, refined, gate_cuts, bad_cuts, seed, cost):
    arrays, cells = {}, {}
    gains = (cache['base_loss'] - refined['loss']).numpy()
    sgain = (refined['sgcs'] - cache['base_sgcs']).numpy()
    base = cache['base_loss'].numpy()
    fin = refined['loss'].numpy()
    snrs, hid = cache['snr'].numpy(), cache['h_index'].numpy()
    offset = cost['monitor_MACs'] / cost['refiner_MACs']
    for s in SNRS:
        ix = snrs == s
        call = (quality[torch.from_numpy(ix)] >= gate_cuts[s]).numpy()
        gain, sg, h = gains[ix], sgain[ix], hid[ix]
        rate = float(call.mean())
        rcompute = min(1., rate + offset)
        bbad, rbad = (base[ix] > bad_cuts[s]), (fin[ix] > bad_cuts[s])
        risk_delta = np.where(call, rbad, bbad).astype(float) - ((1 - rcompute) * bbad + rcompute * rbad)
        delta = gain * call - gain * rcompute
        cells[str(s)] = dict(base_nmse_db=float(base[ix].mean()),
                            all_refinement_gain_db=float(gain.mean()),
                            all_refinement_sgcs_gain=float(sg.mean()),
                            rate=rate, gate_gain_db=float((gain * call).mean()),
                            random_same_calls_gain_db=float(gain.mean() * rate),
                            random_same_MAC_gain_db=float(gain.mean() * rcompute),
                            gate_minus_random_same_MAC_db=float(delta.mean()),
                            delta_H_CI95=cluster_ci(delta, h, 601000 + seed * 100 + s),
                            gate_sgcs_gain=float((sg * call).mean()),
                            random_same_MAC_sgcs_gain=float(sg.mean() * rcompute),
                            base_bad_rate=float(bbad.mean()),
                            gate_bad_rate=float(np.where(call, rbad, bbad).mean()),
                            random_same_MAC_bad_rate=float(((1 - rcompute) * bbad + rcompute * rbad).mean()),
                            bad_risk_delta_H_CI95=cluster_ci(risk_delta, h, 602000 + seed * 100 + s),
                            fraction_all_harmed=float((gain < 0).mean()),
                            fraction_selected_harmed=float((gain[call] < 0).mean()) if call.any() else 0.,
                            true_bad_refinement_gain_db=float(gain[bbad].mean()) if bbad.any() else None)
        arrays[f'gate_{s}'] = call
        arrays[f'delta_{s}'] = delta
    return cells, arrays


def run(seed, args, truths):
    start = time.perf_counter()
    tr, va, te = truths
    net, source = build(seed, tr)
    original = {k: v.clone() for k, v in net.state_dict().items()}
    net.requires_grad_(False)
    oldpaths = [PREV / f'cache_s{seed}_val.pt', PREV / f'quality_bs_only_s{seed}.pt']
    before = {str(p): sha(p) for p in oldpaths}
    rows = torch.load(oldpaths[0], weights_only=True)
    head, hbundle = score_head(seed)
    print('CACHE', seed, 'train', args.ntrain, flush=True)
    train = make_cache(net, tr[:args.ntrain], seed, 'train')
    val = decode_maps(net, rows['y'], rows['snr'], va[rows['h_index']])
    val.update(y=rows['y'], snr=rows['snr'], h_index=rows['h_index'])
    assert torch.max(torch.abs(val['base_loss'] - rows['target'])) < 1e-5
    qtr, qval = predict(head, hbundle, head_inputs(train)), predict(head, hbundle, head_inputs(val))
    fitmask = rows['h_index'] < 250
    select, cal = subset(val, fitmask), subset(val, ~fitmask)
    quality = dict(train=qtr, select=qval[fitmask], cal=qval[~fitmask])
    norm = normalizers(train)
    gate_cuts = {s: float(torch.quantile(quality['cal'][cal['snr'] == s], .8)) for s in SNRS}
    bad_cuts = {s: float(torch.quantile(select['base_loss'][select['snr'] == s], .8)) for s in SNRS}
    cost = costs(args.width)
    models, metadata, calibration = {}, {}, {}
    for variant in VARIANTS:
        models[variant], metadata[variant] = train_one(train, select, quality, seed, variant, norm, args)
        vout = evaluate(models[variant], cal, norm)
        calibration[variant], _ = policies(cal, quality['cal'], vout, gate_cuts, bad_cuts, seed, cost)
        print('CALIBRATION', seed, variant,
              {s: round(calibration[variant][str(s)]['gate_minus_random_same_MAC_db'], 5) for s in SNRS}, flush=True)
    del train
    # Choice made before generating final-test observations. Both variants are
    # nevertheless evaluated and reported; test does not change the selection.
    validation_scores = {v: float(np.mean([calibration[v][str(s)]['gate_minus_random_same_MAC_db'] for s in SNRS])) for v in VARIANTS}
    eligible = [v for v in VARIANTS if all(calibration[v][str(s)]['gate_gain_db'] > 0 for s in SNRS)
                and np.mean([calibration[v][str(s)]['all_refinement_gain_db'] for s in SNRS]) >= 0]
    selected = max(eligible, key=validation_scores.get) if eligible else 'none'
    calibration_record = dict(seed=seed, source=source, cost=cost, training=metadata,
                              gate_cuts=gate_cuts, bad_cuts=bad_cuts,
                              calibration=calibration, validation_scores=validation_scores,
                              eligible=eligible, selected_before_test=selected)
    dump(OUT / f'calibration_s{seed}.json', calibration_record)
    print('LOCKED BEFORE TEST', seed, selected, validation_scores, flush=True)
    # Fresh noise, and a disjoint H segment from the earlier 600-H refiner test.
    test = make_cache(net, te[600:1000], seed, 'test')
    qt = predict(head, hbundle, head_inputs(test))
    arrays = dict(snr=test['snr'].numpy(), h_index=test['h_index'].numpy(),
                  base_loss=test['base_loss'].numpy(), base_sgcs=test['base_sgcs'].numpy(),
                  quality=qt.numpy(), val_snr=cal['snr'].numpy(), val_h_index=cal['h_index'].numpy(),
                  val_quality=quality['cal'].numpy(), val_base_loss=cal['base_loss'].numpy())
    result = dict(calibration_record, conditions={}, seconds=None)
    for variant in VARIANTS:
        out = evaluate(models[variant], test, norm)
        cells, extra = policies(test, qt, out, gate_cuts, bad_cuts, seed, cost)
        result['conditions'][variant] = cells
        arrays[f'{variant}_refined_loss'] = out['loss'].numpy()
        arrays[f'{variant}_refined_sgcs'] = out['sgcs'].numpy()
        for key, valarr in extra.items():
            arrays[f'{variant}_{key}'] = valarr
        print('FINAL', seed, variant,
              {s: round(cells[str(s)]['gate_minus_random_same_MAC_db'], 5) for s in SNRS}, flush=True)
    result['audit'] = dict(base_state_unchanged=all(torch.equal(v, net.state_dict()[k]) for k, v in original.items()),
                           no_base_gradients=all(p.grad is None for p in net.parameters()),
                           originals_unchanged=all(sha(Path(p)) == h for p, h in before.items()),
                           gate_fixed_before_test=True, test_truth_not_in_gate=True,
                           variants_reported_without_test_selection=True)
    assert all(result['audit'].values())
    result['seconds'] = time.perf_counter() - start
    np.savez_compressed(OUT / f'outcomes_s{seed}.npz', **arrays)
    dump(OUT / f'results_s{seed}.json', result)
    print('DONE', seed, 'seconds', round(result['seconds'], 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--ntrain', type=int, default=6000)
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--width', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    protocol = dict(date='2026-11-13', args=vars(args), script_sha256=sha(Path(__file__)),
                    task='Attempt a genuine quality-worst-first refinement advantage; success is not guaranteed.',
                    architecture='one shared residual CNN, 128->256 observed-y projection reshaped 4x8x8 and bilinear resized, spatial conv(24,w,3)-SiLU-conv(w,w,3)-SiLU-conv(w,2,3); no BN; zero output initialization',
                    frozen='RD-AE adaptation, finite16QAM, AWGN, Ns64, SNR0/10/20; encoder, decoder, constellation, power calibration and BS monitor frozen',
                    train='all6000trainH once each, balanced random assignment to one SNR, independent fixed noise; same cache, initialization and minibatch order for both variants',
                    objective='mean per-observation NMSE_dB; uniform weights or 5x weight for predicted-worst40% in each SNR, determined on training scores',
                    splits='valH0:250 selects epoch using the respective training objective; valH250:500 fixes 20% quality gates and selects variant before final test; final testH600:1000 with three new independent noises',
                    validation_caveat='existing BS monitor previously selected using full500valH; these are not new independent validation H',
                    historical_test_caveat='testH600:1000 was withheld from previous refinement experiment but may have been used by other project experiments; new noise and fixed follow-up protocol do not establish independent-population replication',
                    primary='pooled equal-SNR quality gate minus random same conv/linear MAC, positive absolute gate gain, compare to uniform-refiner random same budget; per-SNR results and bad-risk/SGCS/harm always reported',
                    stop='run seed0; expand to source seeds1/2 if at least one variant has positive mean calibration gate-vs-random advantage and nonnegative all-refinement mean gain; otherwise report failure of this fixed design without test-driven search',
                    cost=costs(args.width))
    path = OUT / 'protocol.json'
    if path.exists():
        prev = json.loads(path.read_text(encoding='utf8'))
        assert prev['script_sha256'] == protocol['script_sha256'], 'Code changed after protocol lock'
        protocol['previous_runs'] = prev.get('previous_runs', [prev['args']]) + [vars(args)]
    dump(path, protocol)
    truths = data()
    torch.set_num_threads(args.threads)
    for seed in args.seeds:
        if (OUT / f'results_s{seed}.json').exists():
            print('EXISTS', seed, 'skip', flush=True)
        else:
            run(seed, args, truths)


if __name__ == '__main__':
    main()
