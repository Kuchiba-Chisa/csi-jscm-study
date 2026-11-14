"""UE pre-screen -> BS observed-y retry decision, with fixed controls.

Primary comparison is an offline matched repetition budget on the same eligible
H. A validation-threshold evaluation and control-cost sensitivities are separate.
No new CSI codec training; no true H or future observation in BS policy inputs.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
PROXY = WORK / 'guo_proxy_experiments_20261113'
PREV = WORK / 'two_sided_experiments_20261113'
sys.path.insert(0, str(PROXY))
from run_proxy import (build, data, dump, sha, fit_arrays, predict, apply_cal,
                       make_proxy, nmse_db_per, sgcs_per)
from run_cooperative import cache
from run_actions import add_future
from quality_v5 import spearman

SNRS = (0., 10., 20.)
POLICIES = ('ue_gain', 'ue_quality_best', 'bs_unconditional_gain',
            'bs_conditional_gain', 'bs_conditional_quality_worst')


def bs_x(rows):
    return torch.cat([rows['y'], rows['feature'], rows['condition']], 1)


def ue_x(rows, quality):
    return torch.cat([quality[:, None], rows['condition']], 1)


def eligible(rows, quality, cuts):
    result = torch.zeros(len(quality), dtype=torch.bool)
    for s in SNRS:
        ix = rows['snr'] == s
        result[ix] = quality[ix] <= cuts[s]
    return result


def h_group(rows, X, target, mask):
    xs, ys, snrs, hh = [], [], [], []
    for s in SNRS:
        ix = (rows['snr'] == s) & mask
        for h in rows['h_index'][ix].unique(sorted=True):
            use = ix & (rows['h_index'] == h)
            v = X[use]
            torch.testing.assert_close(v, v[0].expand_as(v), rtol=0, atol=0)
            xs.append(v[0]); ys.append(target[use].mean()); snrs.append(s); hh.append(int(h))
    return dict(x=torch.stack(xs), target=torch.stack(ys), snr=torch.tensor(snrs), h_index=torch.tensor(hh))


def ci(values, h, seed, repeats=1500):
    values, h = np.asarray(values), np.asarray(h)
    ids, inv = np.unique(h, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=repeats)
    draws = (w @ sums) / (w @ counts)
    return dict(mean=float(values.mean()), CI95=np.quantile(draws, [.025, .975]).tolist())


def rank_weights(rows, score, mask, count, source_only):
    """Fractional boundary = expected randomized inclusion; no outcome access."""
    weights = torch.zeros(len(score))
    if not source_only:
        indices = torch.where(mask)[0]
        order = indices[torch.argsort(score[indices], descending=True, stable=True)]
        full = min(len(order), int(math.floor(count)))
        weights[order[:full]] = 1
        if full < len(order):
            weights[order[full]] = max(0., min(1., count - full))
        return weights
    ids = rows['h_index'][mask].unique(sorted=True)
    candidates = []
    for h in ids:
        use = torch.where(mask & (rows['h_index'] == h))[0]
        torch.testing.assert_close(score[use], score[use][0].expand_as(score[use]), rtol=0, atol=0)
        candidates.append((float(score[use][0]), int(h), use))
    candidates.sort(key=lambda v: (-v[0], v[1]))
    remaining = float(count)
    for _, _, use in candidates:
        p = max(0., min(1., remaining / len(use)))
        weights[use] = p
        remaining -= p * len(use)
    return weights


def fixed_cuts(rows, scores, mask, fraction=.2):
    return {s: float(torch.quantile(scores[(rows['snr'] == s) & mask], 1 - fraction)) for s in SNRS}


@torch.no_grad()
def proxy_quality(net, rows, truth, bundle, calibration):
    proxy = make_proxy(net.dec, bundle['rank'])
    proxy.load_state_dict(bundle['state_dict']); proxy.eval()
    result = torch.empty(len(rows['target']))
    for s in SNRS:
        ix = rows['snr'] == s
        h = rows['h_index'][ix]
        for start in range(0, int(h.max()) + 1, 128):
            ids = torch.arange(start, min(start + 128, int(h.max()) + 1))
            positions = torch.stack([torch.where(ix & (rows['h_index'] == i))[0][0] for i in ids])
            rec = proxy(rows['symbols'][positions], torch.full((len(ids), 1), 10 ** (-s / 20)))
            raw = nmse_db_per(rec, truth[ids])
            a, b = calibration[s]
            for i, val in zip(ids, raw * a + b):
                result[ix & (rows['h_index'] == i)] = val
    return result


@torch.no_grad()
def clean_loss(net, rows, truth):
    result = torch.empty(len(rows['target']))
    for s in SNRS:
        ix = rows['snr'] == s
        for start in range(0, int(rows['h_index'][ix].max()) + 1, 128):
            ids = torch.arange(start, min(start + 128, int(rows['h_index'][ix].max()) + 1))
            positions = torch.stack([torch.where(ix & (rows['h_index'] == i))[0][0] for i in ids])
            rec = net.dec(rows['symbols'][positions], torch.full((len(ids), 1), 10 ** (-s / 20)))
            loss = nmse_db_per(rec, truth[ids])
            for i, val in zip(ids, loss):
                result[ix & (rows['h_index'] == i)] = val
    return result


def center_by_h(values, h):
    out = values.clone()
    for i in h.unique():
        use = h == i
        out[use] -= values[use].mean()
    return out


def evaluate(rows, future, scores, passed, bad_cuts, val_action_cuts, seed,
             clean=None, test_draw_half=False):
    cells, arrays = {}, {}
    mid = future['loss'].shape[1] // 2 if test_draw_half else 0
    fl = future['loss'][:, mid:]
    fs = future['sgcs'][:, mid:]
    gain = rows['target'] - fl.mean(1)
    sgain = fs.mean(1) - rows['sgcs']
    for s in SNRS:
        mask = (rows['snr'] == s) & passed
        n = int(mask.sum())
        assert n > 0
        h = rows['h_index'][mask].numpy()
        count_h = len(np.unique(h))
        # Three current noises/H for all test cohorts; use a multiple of their
        # count so UE and BS exactly match resources without fractional choices.
        reps = n // count_h
        count = round(count_h * .2) * reps
        rate = count / n
        flags = {name: rank_weights(rows, scores[name], mask, count, name.startswith('ue_')) for name in POLICIES}
        flags['random'] = mask.float() * rate
        conditions = {}
        b = (rows['target'][mask] > bad_cuts[s]).float()
        rf = (fl[mask] > bad_cuts[s]).float().mean(1)
        for name, w in flags.items():
            ww = w[mask]
            assert abs(float(ww.sum()) - count) < 1e-4
            if name.startswith('ue_'):
                for i in rows['h_index'][mask].unique():
                    v = ww[rows['h_index'][mask] == i]
                    assert bool(torch.all(v == v[0]))
            conditions[name] = dict(retry_rate=float(ww.mean()), expected_retries=float(ww.sum()),
                                    gain_db=float((ww * gain[mask]).mean()),
                                    final_nmse_db=float(rows['target'][mask].mean() - (ww * gain[mask]).mean()),
                                    sgcs_gain=float((ww * sgain[mask]).mean()),
                                    final_sgcs=float(rows['sgcs'][mask].mean() + (ww * sgain[mask]).mean()),
                                    bad_fraction=float((b + ww * (rf - b)).mean()),
                                    symbols_per_passed_observation=64 * (1 + float(ww.mean())),
                                    total_actual_tx_energy=float((rows['symbols'][mask].square().sum(1) * (1 + ww)).sum()),
                                    selected_negative_expected_gain_fraction=float((ww * (gain[mask] < 0)).sum() / ww.sum()))
            arrays[f'flag_{int(s)}_{name}'] = ww.numpy()
        pairs = {}
        for name in POLICIES:
            for baseline in ('random', 'ue_gain', 'ue_quality_best', 'bs_unconditional_gain'):
                if name == baseline:
                    continue
                diff = flags[name][mask] - flags[baseline][mask]
                pairs[f'{name}_minus_{baseline}'] = dict(
                    nmse_gain=ci((diff * gain[mask]).numpy(), h, 217000 + seed * 100 + int(s)),
                    sgcs_gain=ci((diff * sgain[mask]).numpy(), h, 218000 + seed * 100 + int(s)),
                    bad_fraction=ci((diff * (rf - b)).numpy(), h, 219000 + seed * 100 + int(s)))
        # Thresholds fixed on validation; counts naturally drift on test.
        # Matched-usage baselines are offline resource curves, not UE access to y.
        threshold = {}
        for name in POLICIES:
            flag = mask.float() * (scores[name] > val_action_cuts[name][s]).float()
            calls = float(flag.sum())
            random_weight = calls / n
            ue_matched = rank_weights(rows, scores['ue_gain'], mask, calls, True)
            delta = (flag[mask] - ue_matched[mask]) * gain[mask]
            threshold[name] = dict(retry_rate=calls / n, gain_db=float((flag[mask] * gain[mask]).mean()),
                                   random_same_usage_gain_db=float(random_weight * gain[mask].mean()),
                                   UE_same_expected_usage_gain_db=float((ue_matched[mask] * gain[mask]).mean()),
                                   minus_UE_matched_usage=ci(delta.numpy(), h, 220000 + seed * 100 + int(s)),
                                   note='Fixed validation threshold; matched UE count is an offline expected-resource comparison with possible fractional boundary, not a deployed causal UE controller.')
        # Abstract equivalent-symbol cost sensitivity; not an implemented
        # downlink control channel. UE proactive repeat has no BS request.
        costs = {}
        for flag_cost, req_cost in ((0, 1), (1, 1), (1, 4)):
            per_obs_budget = .2 * 64
            counts = {name: max(0., (per_obs_budget - flag_cost) * n / (64 + (0 if name.startswith('ue_') else req_cost))) for name in POLICIES}
            sensitivity = {}
            chosen = {}
            for name in POLICIES:
                ww = rank_weights(rows, scores[name], mask, counts[name], name.startswith('ue_'))[mask]
                chosen[name] = ww
                sensitivity[name] = dict(gain_db=float((ww * gain[mask]).mean()), retry_rate=float(ww.mean()),
                                          overhead_equivalent_symbols_per_observation=float(flag_cost + (0 if name.startswith('ue_') else req_cost) * ww.mean()),
                                          total_extra_equivalent_symbols_per_observation=float(flag_cost + (64 + (0 if name.startswith('ue_') else req_cost)) * ww.mean()))
            sensitivity['bs_conditional_minus_ue_gain'] = ci(((chosen['bs_conditional_gain'] - chosen['ue_gain']) * gain[mask]).numpy(), h, 221000 + seed * 100 + int(s))
            costs[f'passflag{flag_cost}_request{req_cost}'] = sensitivity
        diagnostic = dict(all_passed_repeat_gain_db=float(gain[mask].mean()),
                          within_H_BS_gain_score_vs_actual_repeat_gain_spearman=spearman(
                              center_by_h(scores['bs_conditional_gain'][mask], rows['h_index'][mask]).numpy(),
                              center_by_h(gain[mask], rows['h_index'][mask]).numpy()))
        if clean is not None:
            excess = rows['target'][mask] - clean[mask]
            diagnostic.update(noise_excess_mean_db=float(excess.mean()),
                              BS_gain_score_vs_noise_excess_spearman=spearman(scores['bs_conditional_gain'][mask].numpy(), excess.numpy()),
                              noise_excess_definition='Current NMSE_dB minus same frozen decoder on noiseless symbols at original SNR input; a controlled diagnostic, not additive source/channel mean-error decomposition.')
        cells[str(int(s))] = dict(original_observations=int((rows['snr'] == s).sum()),
                                  passed_observations=n, passed_unique_H=count_h,
                                  pass_fraction=float(passed[rows['snr'] == s].float().mean()),
                                  base_passed_nmse_db=float(rows['target'][mask].mean()),
                                  base_passed_bad_fraction=float(b.mean()), matched_retry_count=count,
                                  policies=conditions, paired=pairs, fixed_threshold=threshold,
                                  control_cost_sensitivity=costs, diagnostics=diagnostic)
    return cells, arrays


def run(seed, args, truths):
    started = time.perf_counter()
    tr, va, te = truths
    net, source = build(seed, tr)
    original = {k: v.clone() for k, v in net.state_dict().items()}
    net.requires_grad_(False)
    paths = [PREV / f'cache_s{seed}_{k}.pt' for k in ('train', 'val', 'test')]
    paths += [PREV / f'future_s{seed}_{k}.pt' for k in ('train', 'val', 'test')]
    paths += [PROXY / f'ue_scores_s{seed}.pt', PROXY / f'proxy_s{seed}.pt', PROXY / f'single_pass_s{seed}.json']
    hashes = {str(p): sha(p) for p in paths}
    rows = {k: torch.load(PREV / f'cache_s{seed}_{k}.pt', weights_only=True) for k in ('train', 'val', 'test')}
    futures = {k: torch.load(PREV / f'future_s{seed}_{k}.pt', weights_only=True) for k in rows}
    aligned = torch.load(PROXY / f'ue_scores_s{seed}.pt', weights_only=True)['aligned']
    previous = json.loads((PROXY / f'single_pass_s{seed}.json').read_text(encoding='utf8'))
    calibration = {float(k): v for k, v in previous['calibration'].items()}
    quality = {k: apply_cal(r, aligned[k]['proxy_clean'], calibration) for k, r in rows.items()}
    fit_val = rows['val']['h_index'] < 250
    cal_val = ~fit_val
    cuts = {}
    for s in SNRS:
        use = fit_val & (rows['val']['snr'] == s)
        unique = h_group(rows['val'], ue_x(rows['val'], quality['val']), rows['val']['target'], use)
        cuts[s] = float(torch.quantile(unique['x'][:, 0], .8))
    passed = {k: eligible(r, quality[k], cuts) for k, r in rows.items()}
    target = {k: r['target'] - futures[k]['loss'].mean(1) for k, r in rows.items()}
    Xb = {k: bs_x(r) for k, r in rows.items()}
    Xu = {k: ue_x(r, quality[k]) for k, r in rows.items()}
    models, bundles, training = {}, {}, {}
    btrain = passed['train']
    bval = passed['val'] & fit_val
    for name, maskt, maskv, yt, yv in (
        ('bs_unconditional_gain', torch.ones_like(btrain), fit_val, target['train'], target['val']),
        ('bs_conditional_gain', btrain, bval, target['train'], target['val']),
        ('bs_conditional_quality', btrain, bval, rows['train']['target'], rows['val']['target']),
    ):
        tick = time.perf_counter()
        model, bundle, meta = fit_arrays(Xb['train'][maskt], yt[maskt], Xb['val'][maskv], yv[maskv],
                                        231000 + seed, 25000, args.epochs)
        models[name], bundles[name] = model, bundle
        meta.update(train_observations=int(maskt.sum()), val_observations=int(maskv.sum()), seconds=time.perf_counter() - tick)
        training[name] = meta
        torch.save(bundle, OUT / f'{name}_s{seed}.pt')
        print('FIT', seed, name, 'epoch', meta['selected_epoch'], 'seconds', round(meta['seconds'], 1), flush=True)
    grouped = {k: h_group(rows[k], Xu[k], target[k], passed[k] & (fit_val if k == 'val' else True)) for k in ('train', 'val')}
    models['ue_gain'], bundles['ue_gain'], training['ue_gain'] = fit_arrays(
        grouped['train']['x'], grouped['train']['target'], grouped['val']['x'], grouped['val']['target'],
        232000 + seed, 25000, args.epochs)
    torch.save(bundles['ue_gain'], OUT / f'ue_gain_s{seed}.pt')

    def get_scores(r, q):
        b, u = bs_x(r), ue_x(r, q)
        return dict(ue_gain=predict(models['ue_gain'], bundles['ue_gain'], u), ue_quality_best=-q,
                    bs_unconditional_gain=predict(models['bs_unconditional_gain'], bundles['bs_unconditional_gain'], b),
                    bs_conditional_gain=predict(models['bs_conditional_gain'], bundles['bs_conditional_gain'], b),
                    bs_conditional_quality_worst=predict(models['bs_conditional_quality'], bundles['bs_conditional_quality'], b))

    vscores = get_scores(rows['val'], quality['val'])
    action_cuts = {name: fixed_cuts(rows['val'], score, passed['val'] & cal_val) for name, score in vscores.items()}
    bad_cuts = {s: float(torch.quantile(rows['val']['target'][bval & (rows['val']['snr'] == s)], .8)) for s in SNRS}
    lock = dict(seed=seed, source=source, eligibility_thresholds=cuts, action_thresholds=action_cuts,
                bad_thresholds=bad_cuts, training=training, input_hashes=hashes)
    dump(OUT / f'locked_s{seed}.json', lock)
    # Retrospective screen with frozen rules; never used to tune or stop fitting.
    cached_scores = get_scores(rows['test'], quality['test'])
    cached, _ = evaluate(rows['test'], futures['test'], cached_scores, passed['test'], bad_cuts,
                         action_cuts, seed, test_draw_half=True)
    print('CACHED', seed, {s: round(c['paired']['bs_conditional_gain_minus_ue_gain']['nmse_gain']['mean'], 5) for s, c in cached.items()}, flush=True)
    del rows, futures, Xb, Xu, aligned
    # Confirmation observations: H600:1000, entirely new current/future noises.
    # H themselves have been used by refinement follow-up; no claim of a wholly
    # untouched historical source test set is made.
    fresh_path = OUT / f'fresh_cache_s{seed}.pt'
    fresh_future_path = OUT / f'fresh_future_s{seed}.pt'
    if fresh_path.exists():
        fresh = torch.load(fresh_path, weights_only=True)
    else:
        fresh = cache(net, te[600:1000], SNRS, 3, 811000000 + seed * 30000000)
        torch.save(fresh, fresh_path)
    if fresh_future_path.exists():
        ff = torch.load(fresh_future_path, weights_only=True)
    else:
        fl, fs = add_future(net, fresh, te[600:1000], args.future_draws, 1011000000 + seed * 30000000)
        ff = dict(loss=fl, sgcs=fs)
        torch.save(ff, fresh_future_path)
    pbundle = torch.load(PROXY / f'proxy_s{seed}.pt', weights_only=True)
    fq = proxy_quality(net, fresh, te[600:1000], pbundle, calibration)
    fp = eligible(fresh, fq, cuts)
    fresh_scores = get_scores(fresh, fq)
    noiseless = clean_loss(net, fresh, te[600:1000])
    cells, decisions = evaluate(fresh, ff, fresh_scores, fp, bad_cuts, action_cuts, seed, clean=noiseless)
    costs = {}
    for name, model in models.items():
        costs[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                           linear_MACs=sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear)))
    result = dict(lock, cached_retrospective=cached, fresh_test=cells, costs=costs,
                  proxy_cost=dict(parameters=112198, conv_linear_MACs=4680720),
                  seconds=time.perf_counter() - started)
    result['audit'] = dict(source_state_unchanged=all(torch.equal(v, net.state_dict()[k]) for k, v in original.items()),
                           source_gradients_empty=all(p.grad is None for p in net.parameters()),
                           original_files_unchanged=all(sha(Path(p)) == v for p, v in hashes.items()),
                           gate_and_training_locked_before_fresh_test=True,
                           BS_inputs_y_features_condition_only=True)
    assert all(result['audit'].values())
    np.savez_compressed(OUT / f'decisions_s{seed}.npz', snr=fresh['snr'].numpy(),
                        h_index=fresh['h_index'].numpy(), ue_quality=fq.numpy(), passed=fp.numpy(),
                        current_loss=fresh['target'].numpy(), noiseless_loss=noiseless.numpy(),
                        future_loss=ff['loss'].numpy(), future_sgcs=ff['sgcs'].numpy(),
                        **{'score_' + k: v.numpy() for k, v in fresh_scores.items()}, **decisions)
    dump(OUT / f'results_s{seed}.json', result)
    print('FRESH', seed, {s: round(c['paired']['bs_conditional_gain_minus_ue_gain']['nmse_gain']['mean'], 5) for s, c in cells.items()}, flush=True)
    print('DONE', seed, 'seconds', round(result['seconds'], 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=65)
    ap.add_argument('--future-draws', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    protocol = dict(date='2026-11-13', args=vars(args), script_sha256=sha(Path(__file__)),
                    frozen='Guo-inspired one-pass UE proxy, finite16QAM RD-AE adaptation, AWGN, Ns64, SNR0/10/20; no codec/proxy retraining',
                    eligibility='calibrated UE single-pass proxy quality <= within-SNR valH0:250 80th percentile; H/noise invariant; rejected samples excluded, their fallback not simulated',
                    fit='existing trainH3000:5000, 3 current noises, 8 independent future repeats; train/val/test H separate',
                    BS_inputs='raw current y128 + deterministic decoder feature64 + known condition2; no UE scalar quality or trueH or transmitted symbols at BS inference',
                    heads='matched ~25k MLP BS unconditional gain, conditional gain, conditional current quality; UE conditional gain sees only local proxy quality and condition and averages current-noise training labels byH',
                    selection='early stopping valH0:250, action20% thresholds valH250:500; all arms retained, no test-driven search',
                    historic_validation='frozen proxy calibration and earlier models used full500valH; this split is not a new independent validation sample',
                    primary='fresh confirmation: BS conditional gain vs random, UE conditional gain and UE quality-best on identical eligible cohort with exactly matched20% repetitions; report unconditional BS and BS quality-worst too',
                    new_evaluation='testH600:1000, 3 new current noises, 16 new independent future repeats; these H were used by another refinement experiment, so new noise is not wholly new population data',
                    action='one repeat of same64complexsymbols; average two received vectors; original SNR input retained for continuity; no effective-SNR retraining or HARQ protocol',
                    controls='offline pool top20% resource comparison; fixed validation thresholds separately evaluate actual usage; abstract equivalent-symbol passflag0/1 and request1/4 sensitivity includes UE no-request advantage, no physical control-error/latency simulation',
                    secondary='SGCS, fixed-val bad risk, within-H score/gain correlation, noiseless-decoder diagnostic, actual transmit energy',
                    cost_caveat='fallback of rejected UE samples, ACK/NACK error, CSI aging, round-trip latency, real multiuser scheduling and throughput not modeled; no claim of system net benefit')
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf8'))
        assert old['script_sha256'] == protocol['script_sha256']
    else:
        dump(path, protocol)
    truths = data()
    torch.set_num_threads(args.threads)
    for seed in args.seeds:
        if (OUT / f'results_s{seed}.json').exists():
            print('EXISTS', seed, 'skip', flush=True)
            continue
        run(seed, args, truths)


if __name__ == '__main__':
    main()
