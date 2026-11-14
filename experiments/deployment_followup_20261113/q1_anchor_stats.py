"""Sparse SNR anchor BN banks: no online gradient updates or calibration."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'extra_experiments_20261113'))
from common import build, data, decoder_bns, dump, evaluate, recalibrate
import numpy as np
import torch


def interpolate(a,b,t,mode):
    result=copy.deepcopy(a)
    for ma,mb,mr in zip(decoder_bns(a),decoder_bns(b),decoder_bns(result)):
        mean=(1-t)*ma.running_mean+t*mb.running_mean
        if mode=='linear_variance':
            variance=(1-t)*ma.running_var+t*mb.running_var
        elif mode=='second_moment':
            second=(1-t)*(ma.running_var+ma.running_mean.square())+t*(mb.running_var+mb.running_mean.square())
            variance=(second-mean.square()).clamp_min(0)
        else:raise ValueError(mode)
        mr.running_mean.copy_(mean);mr.running_var.copy_(variance)
    result.eval()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--lengths',type=int,nargs='+',default=[16,32,64])
    ap.add_argument('--targets',type=float,nargs='+',default=[5,10,15])
    ap.add_argument('--ntest',type=int,default=400)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--out',default='q1_anchor_results.json')
    args=ap.parse_args();torch.set_num_threads(args.threads)
    tr,va,te=data()
    result={'protocol':vars(args),'anchors_snr_db':[0,20],
            'scope':'fixed RD-AE weights, decoder BN buffers only; not official RD-JSCC',
            'interpolation':'linear in SNR dB; two prespecified variance rules, no test-set tuning',
            'target_reference':'one pass of target-condition training data; unavailable to anchor deployment methods',
            'cells':[]}
    for seed in args.seeds:
        started=time.perf_counter();base,meta=build(seed,tr)
        original=copy.deepcopy(base.state_dict())
        for ns in args.lengths:
            anchors=[]
            for snr in [0.,20.]:
                net=copy.deepcopy(base)
                recalibrate(net,tr,snr=snr,ns=ns,seed=93000+seed)
                anchors.append(net)
            stats_dir=ROOT/'anchor_states';stats_dir.mkdir(exist_ok=True)
            for snr,net in zip([0,20],anchors):
                state={name:val.clone() for name,val in net.state_dict().items()
                       if name.startswith('dec.') and name.endswith(('running_mean','running_var'))}
                torch.save(state,stats_dir/f's{seed}_ns{ns}_{snr}dB.pt')
            print(f'ANCHORS seed={seed} ns={ns} ready ({time.perf_counter()-started:.1f}s)',flush=True)
            for snr in args.targets:
                nearest=anchors[0 if snr<=10 else 1]
                target=copy.deepcopy(base)
                recalibrate(target,tr,snr=snr,ns=ns,seed=94000+seed)
                methods={'S0':base,'nearest':nearest,'target_reference':target,
                         'linear_variance':interpolate(*anchors,snr/20.,'linear_variance'),
                         'second_moment':interpolate(*anchors,snr/20.,'second_moment')}
                row={'seed':seed,'snr_db':snr,'ns':ns,'source':meta,'methods':{}}
                for label,net in methods.items():
                    value,losses=evaluate(net,te[:args.ntest],snr,ns,repeats=3,noise_seed=95000)
                    row['methods'][label]=value
                    assert net.p_cal==base.p_cal
                    assert all(torch.equal(original[k],v) for k,v in net.state_dict().items() if k.startswith('enc.'))
                for label in methods:
                    row.setdefault('gain_over_S0_db',{})[label]=row['methods']['S0']-row['methods'][label]
                result['cells'].append(row);dump(ROOT/args.out,result)
                print(f'ANCHOR RESULT seed={seed} ns={ns} snr={snr}: '+str({k:round(v,4) for k,v in row['methods'].items()}),flush=True)
        result.setdefault('duration_s',{})[str(seed)]=time.perf_counter()-started
        dump(ROOT/args.out,result)


if __name__=='__main__':main()
