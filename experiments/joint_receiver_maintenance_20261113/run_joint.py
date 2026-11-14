"""Locked, paired receiver/monitor/refiner maintenance experiment.

No encoder or decoder weight training. Ns64 only: existing refiner's domain.
Source artifacts are read-only. Test observations are generated AFTER policy lock.
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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'q2_refinement_followup_20261113'))
import run_followup as rf
from common import decoder_bns
from quality_v5 import auroc, spearman

SNRS = (0, 10, 20)
OLD = ROOT / 'q2_refinement_followup_20261113'
ANCHORS = ROOT / 'deployment_followup_20261113' / 'anchor_states'


@torch.no_grad()
def observations(net, x, seed, split):
    ys, snrs, ids = [], [], []
    if split == 'train':
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(519000 + seed))
        groups = [(s, order[j::3], 1) for j, s in enumerate(SNRS)]
    else:
        groups = [(s, torch.arange(len(x)), 2 if split == 'val' else 3) for s in SNRS]
    key = {'train': 180000000, 'val': 280000000, 'test': 380000000}[split] + seed * 10000000
    for s, indices, repeats in groups:
        for start in range(0, len(indices), 128):
            ix = indices[start:start + 128]
            feat = torch.full((len(ix), 1), 10 ** (-s / 20))
            z = rf.normalize_power(net.enc(x[ix], feat), 64)
            tx = rf.from_symbols(rf.qam_map_ste(rf.to_symbols(z)))
            sigma = math.sqrt(net.p_ref_of(64, s) / 2) * 10 ** (-s / 20)
            for rep in range(repeats):
                g = torch.Generator().manual_seed(key + s * 10000 + rep * 100000 + start)
                ys.append(tx + sigma * torch.randn(tx.shape, generator=g))
                snrs.append(torch.full((len(ix),), float(s)))
                ids.append(ix)
    return dict(y=torch.cat(ys), snr=torch.cat(snrs), h_index=torch.cat(ids))


def anchor_state(seed, s):
    a = torch.load(ANCHORS / f's{seed}_ns64_0dB.pt', weights_only=True)
    b = torch.load(ANCHORS / f's{seed}_ns64_20dB.pt', weights_only=True)
    return {k: (1 - s / 20) * a[k] + s / 20 * b[k] for k in a}


@torch.no_grad()
def decoded(net, obs, x, seed, adapted):
    output = None
    for s in SNRS:
        ix = torch.where(obs['snr'] == s)[0]
        if adapted:
            net.load_state_dict(anchor_state(seed, s), strict=False)
        part = rf.decode_maps(net, obs['y'][ix], obs['snr'][ix], x[obs['h_index'][ix]])
        if output is None:
            output = {k: torch.empty((len(obs['snr']), *v.shape[1:]), dtype=v.dtype) for k, v in part.items()}
        for k, v in part.items():
            output[k][ix] = v
    output.update(obs)
    return output


def old_refiner(seed):
    bundle = torch.load(OLD / f'uniform_s{seed}.pt', weights_only=True)
    model = rf.Refiner(bundle['width'])
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    return model, bundle


def fit_affine(pred, cache):
    params = {}
    for s in SNRS:
        ix = cache['snr'] == s
        p, y = pred[ix], cache['base_loss'][ix]
        slope = float(((p-p.mean())*(y-y.mean())).mean() / (p.var(unbiased=False) + 1e-8))
        slope = max(slope, 1e-5)
        params[s] = [slope, float(y.mean() - slope * p.mean())]
    return params


def affine(pred, cache, params):
    out = pred.clone()
    for s, (a, b) in params.items():
        ix = cache['snr'] == s
        out[ix] = a * pred[ix] + b
    return out


def update_monitor(old_head, bundle, train, valid, seed, epochs):
    model = copy.deepcopy(old_head)
    x = (rf.head_inputs(train) - bundle['mean']) / bundle['sd']
    v = (rf.head_inputs(valid) - bundle['mean']) / bundle['sd']
    y = (train['base_loss'] - bundle['target_mean']) / bundle['target_sd']
    yv = (valid['base_loss'] - bundle['target_mean']) / bundle['target_sd']
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(619000 + seed)
    with torch.no_grad():
        best = float((model(v)-yv).square().mean())
    state, selected, stale, trace = copy.deepcopy(model.state_dict()), 0, 0, []
    begin = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for ix in torch.randperm(len(x), generator=gen).split(256):
            loss = (model(x[ix])-y[ix]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            score = float((model(v)-yv).square().mean())
        trace.append(score * float(bundle['target_sd'] ** 2))
        if score < best - 1e-5:
            best, state, selected, stale = score, copy.deepcopy(model.state_dict()), ep+1, 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(state); model.eval()
    updated = dict(bundle, state_dict=state)
    meta = dict(selected_epoch=selected, trace_validation_mse_db2=trace,
                seconds=time.perf_counter()-begin,
                parameters=sum(p.numel() for p in model.parameters()))
    torch.save(dict(bundle=updated, meta=meta), OUT / f'updated_monitor_s{seed}.pt')
    return model, updated, meta


def update_refiner(old_model, bundle, train, valid, seed, epochs):
    model = copy.deepcopy(old_model)
    norm = bundle['norm']
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(719000 + seed)
    best = float(rf.evaluate(model, valid, norm)['loss'].mean())
    state, selected, stale, trace = copy.deepcopy(model.state_dict()), 0, 0, []
    begin = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for ix in torch.randperm(len(train['snr']), generator=gen).split(128):
            output = rf.output(model, train, norm, ix)
            loss = rf.nmse_db_per(output, train['target'][ix]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
        model.eval()
        score = float(rf.evaluate(model, valid, norm)['loss'].mean())
        trace.append(score)
        if score < best - 1e-4:
            best, state, selected, stale = score, copy.deepcopy(model.state_dict()), ep+1, 0
        else:
            stale += 1
        if ep == 0 or (ep+1) % 4 == 0:
            print('REFINER UPDATE', seed, ep+1, round(score, 5), 'seconds', round(time.perf_counter()-begin, 1), flush=True)
        if stale >= 5:
            break
    model.load_state_dict(state); model.eval()
    meta = dict(selected_epoch=selected, validation_nmse_db=trace,
                seconds=time.perf_counter()-begin, parameters=sum(p.numel() for p in model.parameters()))
    torch.save(dict(state_dict=state, norm=norm, width=bundle['width'], meta=meta), OUT / f'updated_refiner_s{seed}.pt')
    return model, meta


def quality_metrics(cache, pred, cuts):
    result = {}
    for s in SNRS:
        ix = cache['snr'] == s
        p, y = pred[ix].numpy(), cache['base_loss'][ix].numpy()
        b = y > cuts[s]
        result[str(s)] = dict(mae_db=float(np.abs(p-y).mean()), bias_db=float((p-y).mean()),
                              spearman=float(spearman(p, y)), auroc=float(auroc(p, b)),
                              true_bad_rate=float(b.mean()))
    return result


def gate_cuts(cache, quality):
    return {s: float(torch.quantile(quality[cache['snr'] == s], .8)) for s in SNRS}


def policy(cache, quality, out, cuts, badcuts, seed, fixed_budget=False):
    result, arrays = {}, {}
    cost = rf.costs()
    for s in SNRS:
        ix = cache['snr'] == s
        base, fin = cache['base_loss'][ix].numpy(), out['loss'][ix].numpy()
        bsg, fsg = cache['base_sgcs'][ix].numpy(), out['sgcs'][ix].numpy()
        score = quality[ix].numpy()
        call = score >= cuts[s]
        if fixed_budget:
            call = np.zeros(len(score), dtype=bool)
            call[np.argsort(score, kind='stable')[-round(.2*len(score)):]] = True
        rate = float(call.mean())
        rrate = min(1., rate + cost['monitor_MACs']/cost['refiner_MACs'])
        selected = np.where(call, fin, base)
        sbad, bbad, fbad = selected > badcuts[s], base > badcuts[s], fin > badcuts[s]
        rloss = (1-rrate)*base + rrate*fin
        rbad = (1-rrate)*bbad + rrate*fbad
        delta = sbad.astype(float)-rbad
        h = cache['h_index'][ix].numpy()
        result[str(s)] = dict(base_nmse_db=float(base.mean()), final_nmse_db=float(selected.mean()),
            base_bad_rate=float(bbad.mean()), final_bad_rate=float(sbad.mean()),
            base_sgcs=float(bsg.mean()), final_sgcs=float(np.where(call, fsg, bsg).mean()),
            refiner_call_rate=rate, random_same_MAC_call_rate=rrate,
            additional_MACs=cost['monitor_MACs'] + rate * cost['refiner_MACs'],
            random_same_MAC_nmse_db=float(rloss.mean()), random_same_MAC_bad_rate=float(rbad.mean()),
            gain_vs_random_db=float((rloss-selected).mean()),
            gain_vs_random_H_CI95=rf.cluster_ci(rloss-selected, h, 820000+seed*100+s),
            bad_rate_minus_random=float(delta.mean()),
            bad_rate_minus_random_H_CI95=rf.cluster_ci(delta, h, 830000+seed*100+s),
            always_refined_nmse_db=float(fin.mean()), always_refined_bad_rate=float(fbad.mean()),
            fixed_validation_gate=not fixed_budget)
        arrays[f'call_{s}'] = call
    return result, arrays


def run(seed, args, truths):
    begin = time.perf_counter()
    tr, va, te = truths
    net, source = rf.build(seed, tr)
    net.requires_grad_(False)
    original = copy.deepcopy(net.state_dict())
    codec_params = {n: p.clone() for n, p in net.named_parameters()}
    pcal = copy.deepcopy(net.p_cal)
    paths = [Path(source['checkpoint']), OLD/f'uniform_s{seed}.pt',
             rf.PREV/f'quality_bs_only_s{seed}.pt'] + [ANCHORS/f's{seed}_ns64_{s}dB.pt' for s in (0,20)]
    hashes = {str(p): rf.sha(p) for p in paths}
    head, hbundle = rf.score_head(seed)
    refiner, rbundle = old_refiner(seed)
    print('BUILD', seed, 'training/validation observations', flush=True)
    trainobs = observations(net, tr[:args.ntrain], seed, 'train')
    valobs = observations(net, va[:500], seed, 'val')
    oldval = decoded(net, valobs, va[:500], seed, False)
    oldqval = rf.predict(head, hbundle, rf.head_inputs(oldval))
    fitmask, gatemask = valobs['h_index'] < 250, valobs['h_index'] >= 250
    oldcuts = gate_cuts(rf.subset(oldval, gatemask), oldqval[gatemask])
    # Same per-seed, per-condition absolute thresholds for ALL receiver/policy arms.
    badcuts = {s: float(torch.quantile(oldval['base_loss'][fitmask & (valobs['snr']==s)], .8)) for s in SNRS}
    adaptedtrain = decoded(net, trainobs, tr[:args.ntrain], seed, True)
    adaptedval = decoded(net, valobs, va[:500], seed, True)
    fit, gate = rf.subset(adaptedval, fitmask), rf.subset(adaptedval, gatemask)
    oldqa = rf.predict(head, hbundle, rf.head_inputs(adaptedval))
    affine_params = fit_affine(oldqa[fitmask], fit)
    calibration_q = affine(oldqa, adaptedval, affine_params)
    moment_norm = copy.deepcopy(rbundle['norm'])
    fresh_norm = rf.normalizers(adaptedtrain)
    moment_norm['mean'], moment_norm['sd'] = fresh_norm['mean'], fresh_norm['sd']
    print('UPDATE', seed, 'monitor', flush=True)
    newhead, newbundle, hm = update_monitor(head, hbundle, adaptedtrain, fit, seed, args.monitor_epochs)
    print('UPDATE', seed, 'refiner', flush=True)
    newref, rm = update_refiner(refiner, rbundle, adaptedtrain, fit, seed, args.refiner_epochs)
    newqval = rf.predict(newhead, newbundle, rf.head_inputs(adaptedval))
    newcuts = gate_cuts(gate, newqval[gatemask])
    rethreshold = gate_cuts(gate, oldqa[gatemask])
    affinecuts = gate_cuts(gate, calibration_q[gatemask])
    lock = dict(seed=seed, source=source, input_hashes=hashes,
        script_sha256=rf.sha(Path(__file__)), bad_thresholds=badcuts,
        old_gate_thresholds=oldcuts, recalibrated_old_gate_thresholds=rethreshold,
        affine_params=affine_params, affine_gate_thresholds=affinecuts, updated_gate_thresholds=newcuts,
        monitor_update=hm, refiner_update=rm,
        maintenance=dict(train_H=args.ntrain, labelled_fit_val_H=250, gate_val_H=250,
             affine_parameters=6, refiner_moment_parameters=36,
             monitor_updated_parameters=hm['parameters'], refiner_updated_parameters=rm['parameters']),
        heads_and_policies_locked_before_test=True)
    rf.dump(OUT/f'locked_s{seed}.json', lock)
    del adaptedtrain
    print('LOCKED', seed, 'generating fresh final-test observations', flush=True)
    net.load_state_dict(original)
    testobs = observations(net, te[600:1000], seed, 'test')
    s0 = decoded(net, testobs, te[600:1000], seed, False)
    adapted = decoded(net, testobs, te[600:1000], seed, True)
    # Compact observations allow independent direct replay, without keeping huge feature maps.
    torch.save(dict(observations=testobs, local_H_offset=600, badcuts=badcuts), OUT/f'test_observations_s{seed}.pt')
    q0 = rf.predict(head, hbundle, rf.head_inputs(s0))
    qa = rf.predict(head, hbundle, rf.head_inputs(adapted))
    qaff = affine(qa, adapted, affine_params)
    qnew = rf.predict(newhead, newbundle, rf.head_inputs(adapted))
    oldout = rf.evaluate(refiner, s0, rbundle['norm'])
    aoldout = rf.evaluate(refiner, adapted, rbundle['norm'])
    momentout = rf.evaluate(refiner, adapted, moment_norm)
    newout = rf.evaluate(newref, adapted, rbundle['norm'])
    arms = {
        'S0_old_monitor_old_refiner': (s0, q0, oldout, oldcuts),
        'A_old_modules_old_gate': (adapted, qa, aoldout, oldcuts),
        'A_rethreshold_old_modules': (adapted, qa, aoldout, rethreshold),
        'A_affine_old_modules': (adapted, qaff, aoldout, affinecuts),
        'A_updated_monitor_old_refiner': (adapted, qnew, aoldout, newcuts),
        'A_updated_monitor_moment_refiner': (adapted, qnew, momentout, newcuts),
        'A_updated_monitor_updated_refiner': (adapted, qnew, newout, newcuts)}
    results = dict(lock, policies={}, exact20_diagnostic={}, quality={}, base_comparison={}, refiner_comparison={})
    arrays = dict(snr=testobs['snr'].numpy(), h_index=testobs['h_index'].numpy(),
        S0_base=s0['base_loss'].numpy(), A_base=adapted['base_loss'].numpy(),
        S0_sgcs=s0['base_sgcs'].numpy(), A_sgcs=adapted['base_sgcs'].numpy(),
        q_S0=q0.numpy(), q_A_old=qa.numpy(), q_A_affine=qaff.numpy(), q_A_updated=qnew.numpy(),
        S0_refined=oldout['loss'].numpy(), A_old_refined=aoldout['loss'].numpy(),
        A_moment_refined=momentout['loss'].numpy(), A_updated_refined=newout['loss'].numpy())
    for name, (c, q, out, cuts) in arms.items():
        results['policies'][name], calls = policy(c, q, out, cuts, badcuts, seed)
        results['exact20_diagnostic'][name], _ = policy(c, q, out, cuts, badcuts, seed, fixed_budget=True)
        arrays.update({name+'_'+k: v for k,v in calls.items()})
    results['quality'] = dict(S0=quality_metrics(s0,q0,badcuts), A_old=quality_metrics(adapted,qa,badcuts),
        A_affine=quality_metrics(adapted,qaff,badcuts), A_updated=quality_metrics(adapted,qnew,badcuts))
    for s in SNRS:
        ix = testobs['snr']==s
        h = testobs['h_index'][ix].numpy()
        delta = (s0['base_loss'][ix]-adapted['base_loss'][ix]).numpy()
        results['base_comparison'][str(s)] = dict(S0_nmse_db=float(s0['base_loss'][ix].mean()),
            A_nmse_db=float(adapted['base_loss'][ix].mean()), gain_db=float(delta.mean()),
            gain_H_CI95=rf.cluster_ci(delta,h,840000+seed*100+s),
            S0_bad_rate=float((s0['base_loss'][ix]>badcuts[s]).float().mean()),
            A_bad_rate=float((adapted['base_loss'][ix]>badcuts[s]).float().mean()))
        results['refiner_comparison'][str(s)] = {}
        for name, c, out in [('S0_old',s0,oldout),('A_old',adapted,aoldout),
                              ('A_moment',adapted,momentout),('A_updated',adapted,newout)]:
            gain = (c['base_loss'][ix]-out['loss'][ix]).numpy()
            results['refiner_comparison'][str(s)][name] = dict(always_refine_gain_db=float(gain.mean()),
                always_refine_gain_H_CI95=rf.cluster_ci(gain,h,850000+seed*100+s),
                harmed_fraction=float((gain<0).mean()))
    assert net.p_cal == pcal
    assert all(torch.equal(codec_params[n], p) for n,p in net.named_parameters())
    assert all(rf.sha(Path(p))==h for p,h in hashes.items())
    assert all(p.grad is None for p in net.parameters())
    assert torch.equal(s0['y'],adapted['y'])
    # Positive affine recalibration cannot change rankings; independently verify exact-budget actions.
    assert all(np.array_equal(results_array, arrays['A_rethreshold_old_modules_call_'+str(s)])
               for s in SNRS for results_array in [arrays['A_affine_old_modules_call_'+str(s)]])
    results['audit'] = dict(codec_parameters_unchanged=True, p_cal_unchanged=True, original_artifact_hashes_unchanged=True,
        no_codec_gradients=True, identical_received_y=True, affine_rethreshold_same_actions=True,
        targets_not_monitor_inputs=True, policies_locked_before_test=True)
    results['seconds'] = time.perf_counter()-begin
    np.savez_compressed(OUT/f'outcomes_s{seed}.npz', **arrays)
    rf.dump(OUT/f'results_s{seed}.json',results)
    print('DONE',seed,'seconds',round(results['seconds'],1),
          'adaptation gain',{s:round(results['base_comparison'][str(s)]['gain_db'],4) for s in SNRS},flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--ntrain',type=int,default=2000)
    ap.add_argument('--monitor-epochs',type=int,default=30)
    ap.add_argument('--refiner-epochs',type=int,default=12)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    protocol=dict(args=vars(args),script_sha256=rf.sha(Path(__file__)),
        scope='PyTorch RD-AE adaptation; finite16QAM; AWGN; Ns64; three pretrained source seeds; NOT official full RD-JSCC',
        receiver='frozen weights; decoder BN running_mean/var from existing SNR0/20 training-calibrated anchors; linear interpolation at10',
        old_refiner='uniform variant fixed prospectively, no test-driven choice between uniform and tail-weighted',
        data='train first2000 H one balanced random SNR/H; val first500 H two fresh noises/condition; test H600:1000 three fresh noises/condition',
        validation='first250 H fits affine / early stops small-module updates; last250 H sets runtime score80th-percentile gates',
        bad_event='per-condition S0-loss80th-percentile from first250 validation H; fixed for all arms, never redefined on test/adapted output',
        historical_caveat='validation/test H have been used in previous project work; fresh noises and prospective policy lock are not independent-population replication',
        runtime_gate='validation-fixed thresholds; random benchmark has same actual conv/linear MAC including monitor overhead; call-rate drift reported',
        ranking_diagnostic='also report exactly20% offline-pool top scores; not a causal runtime scheduler',
        calibration='positive per-SNR affine fit,6scalars; does not change within-condition rankings; rethreshold baseline separates gate drift',
        refiner_moments='update input map mean/std using2000 training H,36scalars; y normalization unchanged; weights unchanged',
        updates='offline Adam lr1e-4 monitor30/refiner12 epochs max; identity old state eligible by val; original codec frozen; all arms reported',
        cost=rf.costs(), hardware='local CPU timings only; no RF/PUCCH/CQI/MCS implementation')
    p=OUT/'protocol.json'
    if p.exists():
        assert json.loads(p.read_text(encoding='utf8'))==protocol, 'Do not alter locked experiment'
    else:
        rf.dump(p,protocol)
    truths=rf.data()
    torch.set_num_threads(args.threads)
    for seed in args.seeds:
        if (OUT/f'results_s{seed}.json').exists():
            print('EXISTS',seed,flush=True)
            continue
        run(seed,args,truths)


if __name__=='__main__':
    main()
