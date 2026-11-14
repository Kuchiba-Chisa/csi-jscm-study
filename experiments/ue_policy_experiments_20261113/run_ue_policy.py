"""UE-only source-before-noise action policy vs BS-after-noise selection.

Reuses frozen models and paired current/future observations. The UE action is
constant over all actual feedback noise repetitions of a given H/condition.
Exact top20% is an offline matched resource comparison, not a scheduler.
Validation-fixed UE thresholds are additionally evaluated with actual usage.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
PROXY=OUT.parent/'guo_proxy_experiments_20261113'
PREV=OUT.parent/'two_sided_experiments_20261113'
sys.path.insert(0,str(PROXY))
from run_proxy import fit_arrays,predict,apply_cal,dump,sha
from run_actions import cluster_ci
from quality_v5 import spearman


def by_H(r,X,y):
    """One label per (H,SNR), averaging independent offline current noises."""
    xs=[];ys=[];ss=[];hh=[]
    for s in [0.,10.,20.]:
        ix=r['snr']==s;h=r['h_index'][ix];xx=X[ix];yy=y[ix]
        for i in h.unique(sorted=True):
            use=h==i
            torch.testing.assert_close(xx[use],xx[use][0].expand_as(xx[use]),rtol=0,atol=0)
            xs.append(xx[use][0]);ys.append(yy[use].mean());ss.append(s);hh.append(int(i))
    return {'x':torch.stack(xs),'target':torch.stack(ys),'snr':torch.tensor(ss),'h_index':torch.tensor(hh)}


def top_H_flags(r,score,fraction):
    """Select unique H, then repeat EXACT same decision for every actual noise."""
    flag=torch.zeros(len(score));chosen={}
    for s in [0.,10.,20.]:
        inds=torch.where(r['snr']==s)[0];h=r['h_index'][inds];ids=h.unique(sorted=True)
        vals=torch.stack([score[inds[h==i][0]] for i in ids])
        for i in ids:
            v=score[inds[h==i]]
            torch.testing.assert_close(v,v[0].expand_as(v),rtol=0,atol=0)
        order=torch.argsort(vals,descending=True,stable=True)[:round(len(ids)*fraction)]
        selected=ids[order];chosen[str(int(s))]=selected.tolist()
        flag[inds]=torch.isin(h,selected).float()
    return flag,chosen


def top_obs_flags(r,score,fraction):
    flag=torch.zeros(len(score))
    for s in [0.,10.,20.]:
        inds=torch.where(r['snr']==s)[0]
        selected=inds[torch.argsort(score[inds],descending=True,stable=True)[:round(len(inds)*fraction)]]
        flag[selected]=1
    return flag


def fixed_threshold(val_rows,val_score,test_rows,test_score,fraction):
    flags=torch.zeros(len(test_score));thresholds={}
    for s in [0.,10.,20.]:
        ix=val_rows['snr']==s;ids=val_rows['h_index'][ix].unique(sorted=True)
        q=torch.stack([val_score[ix][val_rows['h_index'][ix]==i][0] for i in ids])
        order=torch.sort(q,descending=True).values;k=round(len(q)*fraction)
        threshold=float((order[k-1]+order[k])/2)
        thresholds[str(int(s))]=threshold
        use=test_rows['snr']==s;flags[use]=(test_score[use]>threshold).float()
    return flags,thresholds


def summarize_policy(rows,future,sgcsfuture,flags,fraction):
    out={}
    for s in [0.,10.,20.]:
        ix=rows['snr']==s;y=rows['target'][ix];h=rows['h_index'][ix]
        # Matched to prior experiment: second half used for evaluation.
        mid=future.shape[1]//2
        gain=y-future[ix,mid:].mean(1)
        sg=sgcsfuture[ix,mid:].mean(1)-rows['sgcs'][ix]
        energy=rows['symbols'][ix].square().sum(1)
        n=len(y);cell={}
        for name,ff in flags.items():
            f=ff[ix];contrib=f*gain
            cell[name]={'nmse_improvement_db':float(contrib.mean()),'mean_nmse_db':float(y.mean()-contrib.mean()),
                'mean_sgcs':float(rows['sgcs'][ix].mean()+(f*sg).mean()),
                'actual_append_fraction':float(f.mean()),'actual_append_count':int(f.sum()),
                'main_plus_repeat_symbols':64*n+64*int(f.sum()),
                'mean_symbols_per_sample':64*(1+float(f.mean())),
                'actual_total_tx_energy':float((energy*(1+f)).sum()),
                'incremental_vs_random_at_own_fraction_db':float(contrib.mean()-f.mean()*gain.mean())}
        cell['random_20_expected']={'nmse_improvement_db':float(fraction*gain.mean()),
            'mean_nmse_db':float(y.mean()-fraction*gain.mean()),
            'mean_sgcs':float(rows['sgcs'][ix].mean()+fraction*sg.mean()),
            'actual_append_fraction':fraction,'mean_symbols_per_sample':64*(1+fraction),
            'main_plus_repeat_symbols':64*n+64*round(n*fraction),
            'actual_total_tx_energy_expected':float((1+fraction)*energy.sum()),
            'note':'random-selection expectation, not one random realization'}
        pair={}
        for name,ff in flags.items():
            if name in ['bs_gain','bs_quality']:continue
            if not name.startswith('threshold_'):
                pair[name+'_minus_bs_gain']=cluster_ci(h,(ff[ix]-flags['bs_gain'][ix])*gain)
                pair[name+'_minus_random']=cluster_ci(h,(ff[ix]-fraction)*gain)
        pair['bs_gain_minus_ue_gain']=cluster_ci(h,(flags['bs_gain'][ix]-flags['ue_gain'][ix])*gain)
        pair['ue_gain_minus_ue_quality']=cluster_ci(h,(flags['ue_gain'][ix]-flags['ue_quality'][ix])*gain)
        pair['ue_gain_minus_quality_best']=cluster_ci(h,(flags['ue_gain'][ix]-flags['ue_quality_best'][ix])*gain)
        pair['threshold_ue_gain_minus_bs_matched_usage']=cluster_ci(h,(flags['threshold_ue_gain'][ix]-flags['bs_matched_threshold_ue_gain'][ix])*gain)
        out[str(int(s))]={'policies':cell,'paired':pair,'all_append_gain_db':float(gain.mean()),
            'threshold_policy_warning':'threshold deployment is validation-fixed, actual test resource usage differs; not a matched-budget superiority comparison'}
    return out


def run(seed,a):
    tick=time.perf_counter()
    rows={k:torch.load(PREV/f'cache_s{seed}_{k}.pt',weights_only=True) for k in ['train','val','test']}
    futures={k:torch.load(PREV/f'future_s{seed}_{k}.pt',weights_only=True) for k in rows}
    scores=torch.load(PROXY/f'ue_scores_s{seed}.pt',weights_only=True)
    single=json.loads((PROXY/f'single_pass_s{seed}.json').read_text(encoding='utf8'))
    calibration={float(k):v for k,v in single['calibration'].items()}
    # Inference information: local one-pass proxy score and known (SNR,Ns).
    local={k:apply_cal(rows[k],scores['aligned'][k]['proxy_clean'],calibration) for k in rows}
    inputs={k:torch.cat([r['condition'],local[k][:,None]],1) for k,r in rows.items()}
    targets={k:r['target']-futures[k]['loss'].mean(1) for k,r in rows.items()}
    grouped={k:by_H(rows[k],inputs[k],targets[k]) for k in ['train','val']}
    head,bundle,meta=fit_arrays(grouped['train']['x'],grouped['train']['target'],
              grouped['val']['x'],grouped['val']['target'],1930000+seed,25000,a.epochs)
    ue_gain={k:predict(head,bundle,X) for k,X in inputs.items()}
    torch.save(bundle,OUT/f'ue_gain_head_s{seed}.pt')
    # Test score from exactly the same inputs is invariant across actual noise.
    flags={};chosen={}
    flags['ue_quality'],chosen['ue_quality']=top_H_flags(rows['test'],local['test'],a.fraction)
    flags['ue_quality_best'],chosen['ue_quality_best']=top_H_flags(rows['test'],-local['test'],a.fraction)
    flags['ue_gain'],chosen['ue_gain']=top_H_flags(rows['test'],ue_gain['test'],a.fraction)
    # Old UE small head has no new training here; source-level quality control.
    flags['old_ue_quality'],chosen['old_ue_quality']=top_H_flags(rows['test'],rows['test']['ue_score'],a.fraction)
    old_action=np.load(PREV/f'action_predictions_s{seed}.npz')
    old_quality=np.load(PREV/f'quality_predictions_s{seed}.npz')
    bs_gain=torch.from_numpy(old_action['bs_gain'].copy())
    flags['bs_gain']=top_obs_flags(rows['test'],bs_gain,a.fraction)
    flags['bs_quality']=top_obs_flags(rows['test'],torch.from_numpy(old_quality['bs_only'].copy()),a.fraction)
    thresholds={}
    for name,pred in [('ue_quality',local),('ue_gain',ue_gain)]:
        flags['threshold_'+name],thresholds[name]=fixed_threshold(rows['val'],pred['val'],rows['test'],pred['test'],a.fraction)
    # Fair comparison at the ACTUAL usage of validation-fixed UE thresholds.
    # Thresholds remain frozen; only BS's allowed count follows UE's usage.
    matched=torch.zeros(len(bs_gain))
    for s in [0.,10.,20.]:
        inds=torch.where(rows['test']['snr']==s)[0];k=int(flags['threshold_ue_gain'][inds].sum())
        matched[inds[torch.argsort(bs_gain[inds],descending=True,stable=True)[:k]]]=1
    flags['bs_matched_threshold_ue_gain']=matched
    result={'seed':seed,'script_sha256':sha(Path(__file__)),'args':vars(a),
        'frozen_proxy_SHA256':sha(PROXY/f'proxy_s{seed}.pt'),'source_single_pass_SHA256':sha(PROXY/f'single_pass_s{seed}.json'),
        'UE_gain_input':'local calibrated one-pass proxy NMSE score + known SNR,Ns; no current y or future noise',
        'gain_training_target':'current_loss - mean future loss, averaged over the 3 offline current realizations per H/condition',
        'gain_head':meta,'thresholds_from_validation_only':thresholds,
        'comparison':'exact top20% of unique H for UE vs top20% current observations for BS, offline matched mean resource test',
        'control':'no UE score sent to BS; control/identification/scheduling overhead omitted for all; not zero-overhead deployed protocol',
        'test':summarize_policy(rows['test'],futures['test']['loss'],futures['test']['sgcs'],flags,a.fraction)}
    for s in [0.,10.,20.]:
        ix=rows['test']['snr']==s;h=rows['test']['h_index'][ix]
        gain=targets['test'][ix]
        result['test'][str(int(s))]['score_spearman_with_repeat_gain']={
            'ue_quality':spearman(local['test'][ix].numpy(),gain.numpy()),
            'ue_gain':spearman(ue_gain['test'][ix].numpy(),gain.numpy()),
            'bs_gain':spearman(bs_gain[ix].numpy(),gain.numpy())}
        for name in ['ue_quality','ue_gain','threshold_ue_quality','threshold_ue_gain']:
            ff=flags[name][ix]
            assert all(bool(torch.all(ff[h==i]==ff[h==i][0])) for i in h.unique())
        expected=64*len(h)+64*round(len(h)*a.fraction)
        for name in ['ue_quality','ue_quality_best','ue_gain','old_ue_quality','bs_gain','bs_quality']:
            assert result['test'][str(int(s))]['policies'][name]['main_plus_repeat_symbols']==expected
    result['audit']={'UE_decisions_same_across_actual_noise_repetitions':True,
        'unique_H_budget_equals_BS_observation_budget':True,
        'previous_source_code_SHA256':sha(PREV/'run_cooperative.py'),
        'proxy_gain_input_dim':int(inputs['train'].shape[1]),
        'UE_gain_training_unique_H':int(rows['train']['h_index'].unique().numel()),
        'UE_gain_rows_after_grouping':len(grouped['train']['target'])}
    np.savez_compressed(OUT/f'decisions_s{seed}.npz',**{'flag_'+k:v.numpy() for k,v in flags.items()},
        score_ue_gain=ue_gain['test'].numpy(),score_ue_quality=local['test'].numpy(),score_bs_gain=bs_gain.numpy(),
        h_index=rows['test']['h_index'].numpy(),snr=rows['test']['snr'].numpy(),
        current_loss=rows['test']['target'].numpy(),future_loss=futures['test']['loss'].numpy())
    result['duration_s']=time.perf_counter()-tick;dump(OUT/f'results_s{seed}.json',result)
    for s,c in result['test'].items():
        print('UE POLICY',seed,s,{k:round(v['nmse_improvement_db'],5) for k,v in c['policies'].items()},flush=True)
        print('PAIRED BS minus UE gain',c['paired']['bs_gain_minus_ue_gain'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--epochs',type=int,default=65);ap.add_argument('--fraction',type=float,default=.2)
    a=ap.parse_args();torch.set_num_threads(4)
    dump(OUT/'protocol.json',{'args':vars(a),'script_sha256':sha(Path(__file__)),
        'existing_proxy':'frozen Guo-inspired single-pass proxy; no additional proxy training',
        'main_comparison':'same initial/repeat symbols, same current/future observations; no quality hint symbol for UE-only',
        'timing':'UE chooses before actual noise; BS chooses after observing current received y',
        'budget':'exact top20% is offline pool selection; validation-fixed thresholds separately report actual use',
        'scope':'Ns64, synthetic CSI, finite16QAM AWGN, frozen RD-AE adaptation; not official Guo/RD-JSCC reproduction'})
    for seed in a.seeds:run(seed,a)
