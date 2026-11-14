"""Predict EXPECTED value of one repeat transmission, not current NMSE.

Freeze the complete source model. Repeat exactly the same QAM symbols, average
the two observations, and keep the original SNR feature at the decoder. This
isolates a fixed action; it does not optimize the retransmission receiver.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'extra_experiments_20261113'))
from common import build,data,dump,nmse_db_per,seed_all
from q2_quality_networks import fit,predict
from model_v4 import normalize_power,to_symbols,qam_map_ste,add_noise,from_symbols,pair_mask
from quality_v5 import spearman
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(net,x,snrs,current_repeats,future_draws,seed_base,ns=64,bs=128):
    """Each row = one current y. Future outcomes are MC labels, never inputs."""
    net.eval()
    rows={k:[] for k in ('y','feature','condition','current_loss','future_losses','snr','h_index')}
    feature={}
    def hook(module,args,out):feature['z']=F.adaptive_avg_pool2d(out,(2,2)).flatten(1).detach()
    handle=net.dec.blocks[-1].af2.register_forward_hook(hook)
    try:
        for si,snr in enumerate(snrs):
            began=time.perf_counter()
            for rep in range(current_repeats):
                for start in range(0,len(x),bs):
                    xb=x[start:start+bs];n=len(xb)
                    nt=torch.full((n,),ns,dtype=torch.long)
                    st=torch.full((n,),float(snr));feat=(10**(-st/20)).view(-1,1)
                    symbols=qam_map_ste(to_symbols(normalize_power(net.enc(xb,feat),nt)))
                    power=torch.full((n,),float(net.p_ref_of(ns,snr)))
                    seed_all(seed_base+si*1000000+rep*100000+start)
                    first,sigma,noise=add_noise(symbols,nt,st,power)
                    mask=pair_mask(nt)
                    y=from_symbols(first)*mask
                    reconstruction=net.dec(y,feat)
                    z=feature['z'].clone()
                    current=nmse_db_per(reconstruction,xb)
                    futures=[]
                    for j in range(future_draws):
                        seed_all(seed_base+50000000+si*1000000+rep*100000+start*100+j)
                        second,_,_=add_noise(symbols,nt,st,power)
                        combined=.5*(y+from_symbols(second)*mask)
                        recovered=net.dec(combined,feat)
                        futures.append(nmse_db_per(recovered,xb))
                    rows['y'].append(y)
                    rows['feature'].append(z)
                    rows['condition'].append(torch.stack([st/20,nt.float()/64],1))
                    rows['current_loss'].append(current)
                    rows['future_losses'].append(torch.stack(futures,1))
                    rows['snr'].append(st);rows['h_index'].append(torch.arange(start,start+n))
            print(f'CACHE nH={len(x)} snr={snr} current_repeats={current_repeats} future={future_draws} time={time.perf_counter()-began:.1f}s',flush=True)
    finally:handle.remove()
    result={k:torch.cat(v).cpu() for k,v in rows.items()}
    result['target']=result['current_loss']-result['future_losses'].mean(1)
    return result


def as_target(rows,target):
    other=dict(rows);other['target']=target
    return other


def policy_scores(rows,gain_score,quality_score,fractions=(.2,.3)):
    """Decisions based on current information; test future draws split for oracle."""
    out={}
    for snr in sorted(rows['snr'].unique().tolist()):
        mask=(rows['snr']==snr).numpy()
        current=rows['current_loss'].numpy()[mask]
        future=rows['future_losses'].numpy()[mask]
        mid=future.shape[1]//2
        g_select=current-future[:,:mid].mean(1)
        g_eval=current-future[:,mid:].mean(1)
        gs,qs=gain_score[mask],quality_score[mask]
        cell={'single_nmse_db':float(current.mean()),'all_retransmit_gain_db':float(g_eval.mean()),
              'gain_score_spearman':spearman(gs,g_eval),'quality_score_spearman_with_gain':spearman(qs,g_eval),
              'expected_gain_std_db':float(g_eval.std()),'policies':{},'n_unique_H':int(rows['h_index'][mask].unique().numel())}
        for p in fractions:
            k=max(1,int(round(len(current)*p)))
            candidates={'predicted_gain':gs,'predicted_quality':qs,'true_worst_privileged':current,
                        'gain_reference_independent_future':g_select,'hindsight_same_future':g_eval}
            entry={'random_expected_gain_db':float(k/len(current)*g_eval.mean()),'extra_resource_fraction':k/len(current)}
            for label,score in candidates.items():
                chosen=np.argsort(-score,kind='mergesort')[:k]
                gain=float(g_eval[chosen].sum()/len(current))
                entry[label]={'gain_db':gain,'nmse_db':float(current.mean()-gain)}
            entry['learned_gain_minus_random_db']=entry['predicted_gain']['gain_db']-entry['random_expected_gain_db']
            entry['learned_gain_minus_quality_db']=entry['predicted_gain']['gain_db']-entry['predicted_quality']['gain_db']
            # Paired cluster bootstrap: all repetitions from one H stay together.
            h=rows['h_index'].numpy()[mask];ids=np.unique(h)
            flags={name:np.zeros(len(current)) for name in ('predicted_gain','predicted_quality')}
            for label in flags:
                score=gs if label=='predicted_gain' else qs
                flags[label][np.argsort(-score,kind='mergesort')[:k]]=1
            by_h=np.array([[np.mean(flags['predicted_gain'][h==i]*g_eval[h==i]),
                            np.mean(flags['predicted_quality'][h==i]*g_eval[h==i]),
                            np.mean((k/len(current))*g_eval[h==i])] for i in ids])
            rng=np.random.default_rng(73000+int(snr)+int(100*p))
            sample=by_h[rng.integers(0,len(ids),(1000,len(ids)))].mean(1)
            entry['cluster_bootstrap_fixed_decision_CI95_gain_minus_random']=[float(v) for v in np.quantile(sample[:,0]-sample[:,2],[.025,.975])]
            entry['cluster_bootstrap_fixed_decision_CI95_gain_minus_quality']=[float(v) for v in np.quantile(sample[:,0]-sample[:,1],[.025,.975])]
            cell['policies'][str(p)]=entry
        out[str(int(snr))]=cell
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--ntrain',type=int,default=2000)
    ap.add_argument('--nval',type=int,default=400)
    ap.add_argument('--ntest',type=int,default=400)
    ap.add_argument('--train-future',type=int,default=8)
    ap.add_argument('--val-future',type=int,default=8)
    ap.add_argument('--test-future',type=int,default=32)
    ap.add_argument('--current-repeats',type=int,default=2)
    ap.add_argument('--test-repeats',type=int,default=3)
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--out',default='q2_action_results.json')
    a=ap.parse_args();torch.set_num_threads(a.threads)
    assert a.test_future>=4 and a.test_future%2==0
    tr,va,te=data()
    result={'protocol':vars(a),'ns':64,'source_scope':'frozen RD-AE adaptation',
            'action':'repeat identical symbols, average observations, decode with original SNR feature',
            'input':'same frozen decoder feature + SNR + Ns for gain/quality heads, about 25k parameters',
            'training_label':'current NMSE_dB minus average future NMSE_dB over independent repeat noises',
            'selection_reference':'oracle estimated with first half of future noises, evaluated with second half',
            'models':[]}
    for seed in a.seeds:
        began=time.perf_counter();net,meta=build(seed,tr);original=copy.deepcopy(net.state_dict())
        training=generate(net,tr[:a.ntrain],(0,10,20),a.current_repeats,a.train_future,100000000)
        validation=generate(net,va[:a.nval],(0,10,20),1,a.val_future,300000000)
        gain_head,gain_bundle,gain_info=fit(training,validation,'feature',seed,a.epochs)
        quality_training=as_target(training,training['current_loss'])
        quality_validation=as_target(validation,validation['current_loss'])
        quality_head,quality_bundle,quality_info=fit(quality_training,quality_validation,'feature',seed,a.epochs)
        print(f'HEADS seed={seed} gain_val_mse={gain_info["val_mse_db2"]:.4f} quality_val_mse={quality_info["val_mse_db2"]:.4f}',flush=True)
        test=generate(net,te[:a.ntest],(0,10,20),a.test_repeats,a.test_future,500000000)
        gain_pred=predict(gain_head,gain_bundle,test)
        quality_pred=predict(quality_head,quality_bundle,test)
        row={'seed':seed,'source':meta,'gain_head':gain_info,'quality_head':quality_info,
             'test':policy_scores(test,gain_pred,quality_pred),'duration_s':time.perf_counter()-began}
        assert all(torch.equal(original[k],v) for k,v in net.state_dict().items()),'Frozen backbone changed'
        for label,bundle in [('gain',gain_bundle),('quality',quality_bundle)]:
            torch.save(bundle,ROOT/f'action_head_s{seed}_{label}.pt')
        np.savez_compressed(ROOT/f'action_test_s{seed}.npz',**{k:v.numpy() for k,v in test.items()},
                            predicted_gain=gain_pred,predicted_quality=quality_pred)
        result['models'].append(row);dump(ROOT/a.out,result)
        for snr,c in row['test'].items():
            p=c['policies']['0.2']
            print(f'ACTION seed={seed} snr={snr} p=.2 random={p["random_expected_gain_db"]:.4f} '+
                  f'quality={p["predicted_quality"]["gain_db"]:.4f} value={p["predicted_gain"]["gain_db"]:.4f} '+
                  f'independent_ref={p["gain_reference_independent_future"]["gain_db"]:.4f}',flush=True)
        print(f'SEED {seed} complete {time.perf_counter()-began:.1f}s',flush=True)


if __name__=='__main__':main()
