"""Frozen validation thresholds + incremental monitoring cost. No training."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'guo_proxy_experiments_20261113'))
from run_proxy import make_proxy, build, data, bs_inputs, predict, sha, dump
from q2_quality_networks import Monitor
from common import nmse_db_per

BS = ROOT / 'two_sided_experiments_20261113'
UE = ROOT / 'guo_proxy_experiments_20261113'
METHODS = ('bs_small', 'bs_large', 'ue_proxy')
TARGETS = (0.8, 0.9, 0.95)
SNRS = (0, 10, 20)


def load_head(path):
    bundle = torch.load(path, weights_only=True)
    model = Monitor(bundle['input_dim'], bundle['budget'])
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    return model, bundle


def confusion(y, alarm):
    return np.array([np.sum(y & alarm), np.sum(~y & alarm),
                     np.sum(y & ~alarm), np.sum(~y & ~alarm)], dtype=np.int64)


def rates(c):
    tp, fp, fn, tn = np.asarray(c).T
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return {'tpr': divide(tp, tp+fn), 'fpr': divide(fp, fp+tn),
            'precision': divide(tp, tp+fp), 'alarm_fraction': divide(tp+fp, tp+fp+fn+tn),
            'bad_prevalence': divide(tp+fn, tp+fp+fn+tn)}


def alarm_cut(scores, bad, target):
    positives = np.sort(scores[bad])
    assert len(positives) > 0
    required = math.ceil(target * len(positives) - 1e-10)
    return float(positives[len(positives)-required])


def scalar_metrics(y, alarm):
    c = confusion(y, alarm)
    return {'counts_tp_fp_fn_tn': c.tolist(), **{k: float(v) for k, v in rates(c).items()}}


def bootstrap_metrics(y, alarm, h, weights):
    n = int(h.max()) + 1
    counts = np.stack([np.bincount(h, weights=v.astype(int), minlength=n)
                       for v in (y & alarm, ~y & alarm, y & ~alarm, ~y & ~alarm)], 1)
    return rates(weights @ counts)


def cost_model(model, call):
    total = [0]
    def hook(m, args, result):
        if isinstance(m, nn.Linear):
            total[0] += result.numel() * m.in_features
        else:
            total[0] += result.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Linear, nn.Conv2d))]
    try:
        call()
    finally:
        for handle in handles:
            handle.remove()
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'buffer_bytes': sum(p.numel()*p.element_size() for p in model.buffers()),
            'conv_linear_macs': total[0]}


def benchmark(functions, seed, blocks=9, iterations=100):
    # All paths batch1, same CPU and one Torch thread. Randomized interleaving
    # makes drift less correlated with a particular path. No device transfer.
    for fn in functions.values():
        for _ in range(25):
            fn()
    rng = np.random.default_rng(704000 + seed)
    results = {name: [] for name in functions}
    for _ in range(blocks):
        for name in rng.permutation(list(functions)):
            tick = time.perf_counter_ns()
            for _ in range(iterations):
                functions[name]()
            results[name].append((time.perf_counter_ns()-tick)/iterations/1e6)
    return {name: {'block_ms_per_sample': vals, 'median_ms': float(np.median(vals)),
                   'min_ms': float(min(vals)), 'max_ms': float(max(vals))}
            for name, vals in results.items()}


def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
    except Exception:
        return platform.processor()


def run(seed, a, truths):
    started = time.perf_counter()
    paths = [BS/f'cache_s{seed}_{split}.pt' for split in ('val', 'test')]
    paths += [BS/f'quality_{name}_s{seed}.pt' for name in ('bs_only', 'bs_large')]
    paths += [UE/f'proxy_s{seed}.pt', UE/f'ue_scores_s{seed}.pt',
              UE/f'results_s{seed}.json', UE/f'single_pass_s{seed}.json',
              BS/f'quality_predictions_s{seed}.npz']
    source_hashes = {str(p): sha(p) for p in paths}
    net, source = build(seed, truths[0])
    previous = json.loads((UE/f'results_s{seed}.json').read_text(encoding='utf8'))
    assert source['sha256'] == previous['source']['sha256']
    assert source['source_model_sha256'] == previous['source']['source_model_sha256']
    frozen = {k: v.clone() for k, v in net.state_dict().items()}
    rows = {k: torch.load(BS/f'cache_s{seed}_{k}.pt', weights_only=True) for k in ('val','test')}
    heads = {name: load_head(BS/f'quality_{old}_s{seed}.pt')
             for name, old in [('bs_small','bs_only'), ('bs_large','bs_large')]}
    bundle = torch.load(UE/f'proxy_s{seed}.pt', weights_only=True)
    proxy = make_proxy(net.dec, bundle['rank'])
    proxy.load_state_dict(bundle['state_dict']); proxy.eval()
    proxy_state = {k: v.clone() for k, v in proxy.state_dict().items()}
    ue = torch.load(UE/f'ue_scores_s{seed}.pt', weights_only=True)
    scores = {split: {name: predict(head, b, bs_inputs(r)).numpy()
                      for name,(head,b) in heads.items()} for split,r in rows.items()}
    for split in rows:
        scores[split]['ue_proxy'] = ue['aligned'][split]['proxy_clean'].numpy()

    old = np.load(BS/f'quality_predictions_s{seed}.npz')
    prediction_errors = {}
    for name, old_name in [('bs_small','bs_only'), ('bs_large','bs_large')]:
        err = float(np.max(np.abs(scores['test'][name] - old[old_name])))
        assert err < 1e-5, (name, err)
        prediction_errors[name] = err

    # Check independent clean single-pass proxy computation on held-out H.
    with torch.inference_mode():
        for s in SNRS:
            ix = torch.where(rows['test']['snr'] == s)[0][:16]
            h = rows['test']['h_index'][ix]
            predicted = nmse_db_per(proxy(rows['test']['symbols'][ix],
                                  torch.full((len(ix),1), 10**(-s/20))), truths[2][h])
            err = float((predicted - torch.from_numpy(scores['test']['ue_proxy'][ix])).abs().max())
            assert err < 1e-4, (s,err)
            prediction_errors[f'proxy_{s}'] = err

    result = {'seed':seed, 'source':source, 'source_hashes':source_hashes,
              'prediction_replay_max_abs_error':prediction_errors, 'conditions':{}}
    arrays = {}
    for split,r in rows.items():
        for key in ('target','snr','h_index'):
            arrays[f'{split}_{key}'] = r[key].numpy()
        for method in METHODS:
            arrays[f'{split}_{method}'] = scores[split][method]
    vr, tr = rows['val'], rows['test']
    for s in SNRS:
        va = ((vr['snr']==s) & (vr['h_index']<250)).numpy()
        vb = ((vr['snr']==s) & (vr['h_index']>=250)).numpy()
        ti = (tr['snr']==s).numpy()
        assert va.sum()==vb.sum()==500 and ti.sum()==1800
        bad_cut = float(np.quantile(vr['target'].numpy()[va], 0.8))
        val_bad = vr['target'].numpy()[vb] > bad_cut
        test_bad = tr['target'].numpy()[ti] > bad_cut
        h = tr['h_index'].numpy()[ti]
        assert len(np.unique(h))==600
        rng = np.random.default_rng(330000+seed*100+s)
        weights = rng.multinomial(600, np.ones(600)/600, size=a.bootstrap)
        cond = {'bad_nmse_db_threshold':bad_cut,
                'quality_definition_H':250,'alarm_calibration_H':250,'test_H':600,
                'validation_B_bad_fraction':float(val_bad.mean()),
                'test_bad_fraction':float(test_bad.mean()),'targets':{}}
        for target in TARGETS:
            records = {}; boots = {}
            for name in METHODS:
                cut = alarm_cut(scores['val'][name][vb], val_bad, target)
                val_alarm = scores['val'][name][vb] >= cut
                test_alarm = scores['test'][name][ti] >= cut
                boot = bootstrap_metrics(test_bad, test_alarm, h, weights)
                boots[name] = boot
                records[name] = {'alarm_threshold':cut,
                                 'validation_B':scalar_metrics(val_bad,val_alarm),
                                 'test':scalar_metrics(test_bad,test_alarm),
                                 'H_cluster_bootstrap_CI95':{
                                     key:np.quantile(val,[.025,.975]).tolist() for key,val in boot.items()}}
                assert records[name]['validation_B']['tpr'] >= target-1e-10
                arrays[f'test_alarm_{s}_{target}_{name}'] = test_alarm
            comparisons = {}
            for name in ('bs_small','bs_large'):
                comparisons[name+'_minus_ue_proxy'] = {
                    key:{'difference':records[name]['test'][key]-records['ue_proxy']['test'][key],
                         'H_cluster_CI95':np.quantile(boots[name][key]-boots['ue_proxy'][key],[.025,.975]).tolist()}
                    for key in ('tpr','fpr','precision','alarm_fraction')}
            cond['targets'][str(target)] = {'methods':records, 'paired_differences':comparisons}
        result['conditions'][str(s)] = cond
        primary=cond['targets']['0.9']['methods']
        print('DETECTION',seed,s,{m:{k:round(primary[m]['test'][k],4) for k in ('tpr','fpr','precision')} for m in METHODS},flush=True)

    # Incremental compute and single-sample CPU costs. Encoder / main decoder
    # are common communication work, excluded from additional monitor costs.
    sample_index = int(torch.where(tr['snr']==10)[0][0])
    y = tr['y'][sample_index:sample_index+1].clone()
    x = tr['symbols'][sample_index:sample_index+1].clone()
    h = truths[2][tr['h_index'][sample_index:sample_index+1]].clone()
    af = torch.full((1,1), .1)
    condition = torch.tensor([[.5,1.]])
    captured = {}
    def capture(module, args, value):
        captured['map'] = value
    with torch.inference_mode():
        hook = net.dec.blocks[-1].af2.register_forward_hook(capture)
        net.dec(y,af)
        feature_map = captured['map'].clone()
        hook.remove()
        cuts = {m: result['conditions']['10']['targets']['0.9']['methods'][m]['alarm_threshold'] for m in METHODS}
        def bs_monitor(name, fmap):
            head,b = heads[name]
            feature = torch.cat([F.adaptive_avg_pool2d(fmap,(2,2)).flatten(1),condition],1)
            score = head((feature-b['mean'])/b['sd'])*b['target_sd']+b['target_mean']
            return score >= cuts[name]
        def ue_monitor():
            return nmse_db_per(proxy(x,af),h) >= cuts['ue_proxy']
        def integrated(name):
            rec = net.dec(y,af)
            return rec,bs_monitor(name,captured['map'])
        functions = {'ue_proxy_increment':ue_monitor,
                     'bs_small_increment':lambda:bs_monitor('bs_small',feature_map),
                     'bs_large_increment':lambda:bs_monitor('bs_large',feature_map),
                     'decoder_baseline':lambda:net.dec(y,af)}
        # No hooks in baseline or direct-monitor timing.
        timing = benchmark(functions,seed,a.timing_blocks,a.timing_iterations)
        # Integrated timings include the Python feature tap; baseline also has
        # that tap to avoid counting hook overhead as a learned monitor effect.
        hook = net.dec.blocks[-1].af2.register_forward_hook(capture)
        integrated_timing = benchmark({'decoder_tapped_baseline':lambda:net.dec(y,af),
                                       'decoder_plus_bs_small':lambda:integrated('bs_small'),
                                       'decoder_plus_bs_large':lambda:integrated('bs_large')},
                                      seed+100,a.timing_blocks,a.timing_iterations)
        hook.remove()
        model_cost = {}
        for name,(head,b) in heads.items():
            model_cost[name] = cost_model(head,lambda n=name:bs_monitor(n,feature_map))
            model_cost[name]['preprocessing_tensor_bytes'] = sum(b[k].numel()*b[k].element_size()
                                              for k in ('mean','sd','target_mean','target_sd'))
            model_cost[name]['alarm_threshold_bytes'] = 3*4
        model_cost['ue_proxy'] = cost_model(proxy,ue_monitor)
        model_cost['ue_proxy']['preprocessing_tensor_bytes'] = 0
        model_cost['ue_proxy']['alarm_threshold_bytes'] = 3*4
        model_cost['common_decoder_reference'] = cost_model(net.dec,lambda:net.dec(y,af))
        for value in model_cost.values():
            value['persistent_tensor_bytes_including_preprocessing'] = sum(value.get(k,0) for k in (
                'parameter_bytes','buffer_bytes','preprocessing_tensor_bytes','alarm_threshold_bytes'))
    result['cost'] = model_cost
    result['timing'] = timing
    result['integrated_timing'] = integrated_timing
    result['audit'] = {'no_training':True,'thresholds_fit_without_test_data':True,
                       'quality_definition_and_alarm_H_disjoint':True,
                       'source_files_unchanged':all(sha(Path(p))==v for p,v in source_hashes.items()),
                       'backbone_state_unchanged':all(torch.equal(v,net.state_dict()[k]) for k,v in frozen.items()),
                       'proxy_state_unchanged':all(torch.equal(v,proxy.state_dict()[k]) for k,v in proxy_state.items())}
    assert all(result['audit'].values())
    np.savez_compressed(OUT/f'predictions_s{seed}.npz',**arrays)
    result['seconds']=time.perf_counter()-started
    dump(OUT/f'results_s{seed}.json',result)
    print('DONE',seed,'seconds',round(result['seconds'],1),'MACs',{k:v['conv_linear_macs'] for k,v in model_cost.items()},flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--timing-blocks',type=int,default=9)
    ap.add_argument('--timing-iterations',type=int,default=100)
    a=ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(90113)
    dump(OUT/'protocol.json',{
        'date':'2026-11-13','args':vars(a),'script_sha256':sha(Path(__file__)),
        'primary_target_validation_recall':.9,'secondary_targets':list(TARGETS),
        'methods':list(METHODS),'training':False,
        'bad_definition':'per model/SNR, NMSE_dB > 80th percentile from first250 validation H, both cached noise repeats',
        'alarm_calibration':'last250 validation H, choose largest observed bad score cutoff achieving >= target recall; alarm score >= cutoff',
        'test':'600 H per SNR,3 actual noise repeats per H; fixed bad definition and alarm thresholds; report actual test recall',
        'validation_reuse':'validation used previously in model selection; A/B split newly separates quality definition and threshold setting, not a new independent validation dataset',
        'UE_score':'single pass frozen proxy on clean codeword, compare to UE H; raw NMSE_dB, no full-validation affine fit required for rankings',
        'cost':'incremental monitor only; conv/linear MACs exclude normalization/activation/pooling; timing includes them',
        'hardware':{'cpu':cpu_name(),'os':platform.platform(),'torch':torch.__version__,
                    'torch_threads':torch.get_num_threads(),'batch':1,'device':'cpu','dtype':'float32'},
        'latency':'preloaded tensors, no IO/network/encoder, no device transfer; randomized timing blocks; same PC, not actual UE/BS hardware',
        'scope':'synthetic CSI,Ns64,finite16QAM,AWGN,our frozen RD-AE adaptation; not official Guo or RD-JSCC reproduction'})
    truths=data()
    # common.py sets8 threads on import; restore explicitly after all imports.
    torch.set_num_threads(1)
    with torch.inference_mode():
        for seed in a.seeds:
            run(seed,a,truths)


if __name__=='__main__':
    main()
