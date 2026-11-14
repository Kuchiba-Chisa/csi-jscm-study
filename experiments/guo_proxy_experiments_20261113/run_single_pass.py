"""Low-cost single-pass proxy control, sharing the already trained proxy.

Added to compare inference cost fairly with MC8: same validation calibration,
same BS head budget, same original observations/future outcomes and hint AWGN.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from run_proxy import (OUT,PREV,data,build,scalar_calibration,apply_cal,fit_arrays,predict,bs_inputs,
                       metrics,paired,quantizer,classes_and_values,transmit_hint,constellation,evaluate,dump,sha)


def run(seed):
    tr,_,_=data();net,source=build(seed,tr)
    rows={k:torch.load(PREV/f'cache_s{seed}_{k}.pt',weights_only=True) for k in ['train','val','test']}
    bundle=torch.load(OUT/f'ue_scores_s{seed}.pt',weights_only=True)
    raw={k:bundle['aligned'][k]['proxy_clean'] for k in rows}
    table=scalar_calibration(rows['val'],raw['val'],rows['val']['target'])
    score={k:apply_cal(rows[k],raw[k],table) for k in rows}
    result={'seed':seed,'source':source,'calibration':table,'script_sha256':sha(Path(__file__)),
            'scope':'one proxy pass on clean transmitted codeword; no local MC; same trained noisy-input proxy',
            'quality':{'single_proxy_calibrated':{'test':metrics(rows['test'],score['test'])}},'paired':{}}
    quant=quantizer(rows['train'],score['train'],2);hints={};scales=None;energies={};stats={}
    for split,r in rows.items():
        classes,clean=classes_and_values(r,score[split],quant)
        noisy,rx,scales,st=transmit_hint(r,score[split],quant,2,scales,
                      base=500000+{'train':10000,'val':20000,'test':30000}[split])
        hints[split]={'clean':clean,'noisy':noisy};stats[split]=st
        if split=='test':
            energy=torch.empty(len(classes));pts=constellation(2)
            for s in [0.,10.,20.]:
                ix=r['snr']==s
                energy[ix]=pts[classes[ix]].square().sum(1)*scales[s]**2*float(net.p_ref_of(64,s))
            energies={2:energy,4:energy}
    result['hint']={'stats':stats,'scales':scales,'table':{str(s):{k:v.tolist() for k,v in t.items()} for s,t in quant.items()}}
    previous=np.load(PREV/f'quality_predictions_s{seed}.npz');bs=torch.from_numpy(previous['bs_only'].copy())
    preds={'bs_only':bs}
    for name,kind in [('single_proxy_fusion_float','float'),('single_proxy_fusion_noisy2','noisy')]:
        ins={k:torch.cat([bs_inputs(r),(score[k] if kind=='float' else hints[k]['noisy'])[:,None]],1) for k,r in rows.items()}
        head,bundle,meta=fit_arrays(ins['train'],rows['train']['target'],ins['val'],rows['val']['target'],710000+seed,25000,65)
        p=predict(head,bundle,ins['test']);preds[name]=p
        meta['test']=metrics(rows['test'],p);result['quality'][name]=meta
        result['paired'][name+'_minus_bs']=paired(rows['test'],p,bs)
        torch.save(bundle,OUT/f'{name}_s{seed}.pt')
        print('SINGLE QUALITY',seed,name,{s:round(x['mae_db'],4) for s,x in meta['test'].items()},flush=True)
    result['paired']['single_proxy_calibrated_minus_bs']=paired(rows['test'],score['test'],bs)
    futures={k:torch.load(PREV/f'future_s{seed}_{k}.pt',weights_only=True) for k in rows}
    targets={k:rows[k]['target']-futures[k]['loss'].mean(1) for k in rows}
    oldactions=np.load(PREV/f'action_predictions_s{seed}.npz')
    actions={'bs_gain':torch.from_numpy(oldactions['bs_gain'].copy())}
    for name,kind in [('single_proxy_float_gain','float'),('single_proxy_noisy2_gain','noisy')]:
        ins={k:torch.cat([bs_inputs(r),(score[k] if kind=='float' else hints[k]['noisy'])[:,None]],1) for k,r in rows.items()}
        head,bundle,meta=fit_arrays(ins['train'],targets['train'],ins['val'],targets['val'],930000+seed,25000,65)
        actions[name]=predict(head,bundle,ins['test']);result.setdefault('action_heads',{})[name]=meta
        torch.save(bundle,OUT/f'{name}_s{seed}.pt')
    result['actions']=evaluate(rows['test'],actions,futures['test']['loss'],futures['test']['sgcs'],preds,energies)
    dump(OUT/f'single_pass_s{seed}.json',result)
    for s in ['0','10','20']:print('SINGLE ACTION',seed,s,result['actions'][s]['paired_net_gain_db'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a=ap.parse_args()
    torch.set_num_threads(4)
    for seed in a.seeds:run(seed)
