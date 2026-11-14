"""Audit source-model invariance, exact channel path and decision information."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'extra_experiments_20261113'))
from common import build,data,dump,decoder_bns,seed_all
from q2_quality_networks import Monitor,predict
from model_v4 import normalize_power,to_symbols,qam_map_ste,add_noise,from_symbols,pair_mask
from q1_anchor_stats import interpolate
import numpy as np
import torch


def main():
    torch.set_num_threads(2)
    tr,va,te=data();net,_=build(0,tr)
    original=copy.deepcopy(net.state_dict())
    # Direct simulator must equal the trained Link.forward for identical noise.
    x=te[:32];n=len(x);ns=64;snr=0.;nt=torch.full((n,),ns,dtype=torch.long)
    st=torch.full((n,),snr);feat=(10**(-st/20)).view(-1,1)
    captured={}
    def hook(module,args):captured['y']=args[0].clone()
    handle=net.dec.register_forward_pre_hook(hook)
    seed_all(98765)
    with torch.no_grad():reference=net(x,snr,ns)
    handle.remove()
    with torch.no_grad():
        symbols=qam_map_ste(to_symbols(normalize_power(net.enc(x,feat),nt)))
        power=torch.full((n,),float(net.p_ref_of(ns,snr)))
        seed_all(98765);first,_,_=add_noise(symbols,nt,st,power)
        y=from_symbols(first)*pair_mask(nt)
        estimate=net.dec(y,feat)
    assert torch.equal(y,captured['y'])
    assert torch.equal(reference,estimate)
    # Independent complex noise averages have half the total noise power.
    sym=symbols[:1].repeat(2000,1,1);nn_=torch.full((len(sym),),64,dtype=torch.long)
    ss=torch.zeros(len(sym));pp=torch.full((len(sym),),float(power[0]))
    seed_all(12001);_,_,n1=add_noise(sym,nn_,ss,pp)
    seed_all(12002);_,_,n2=add_noise(sym,nn_,ss,pp)
    ratio=float(((n1+n2)/2).square().sum(-1).mean()/n1.square().sum(-1).mean())
    assert abs(ratio-.5)<.02
    stats_count=0
    for path in (ROOT/'anchor_states').glob('*.pt'):
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert all(k.startswith('dec.') and k.endswith(('running_mean','running_var')) for k in state)
        for k,v in state.items():
            assert torch.isfinite(v).all()
            if k.endswith('running_var'):assert (v>=0).all()
        stats_count+=1
    assert stats_count==18, f'Incomplete anchor bank: {stats_count}'
    # Endpoint preservation for both interpolation rules, before loading actual banks.
    second=copy.deepcopy(net)
    for m in decoder_bns(second):
        m.running_mean.add_(.2);m.running_var.mul_(1.1)
    for mode in ('linear_variance','second_moment'):
        for t,expected in ((0.,net),(1.,second)):
            candidate=interpolate(net,second,t,mode)
            for m,e in zip(decoder_bns(candidate),decoder_bns(expected)):
                assert torch.allclose(m.running_mean,e.running_mean,atol=1e-6)
                assert torch.allclose(m.running_var,e.running_var,atol=1e-6)
    b=json.loads((ROOT/'q2_action_results.json').read_text(encoding='utf8'))
    policy_recomputations=[]
    for row in b['models']:
        seed=row['seed'];raw=np.load(ROOT/f'action_test_s{seed}.npz')
        bundle=torch.load(ROOT/f'action_head_s{seed}_gain.pt',map_location='cpu',weights_only=True)
        head=Monitor(bundle['input_dim']);head.load_state_dict(bundle['state_dict']);head.eval()
        features={k:torch.from_numpy(raw[k]) for k in ('y','feature','condition','target')}
        before=predict(head,bundle,features)
        mutated=dict(features);mutated['target']=torch.full_like(features['target'],1e8)
        assert np.array_equal(before,predict(head,bundle,mutated))
        assert np.allclose(before,raw['predicted_gain'],atol=1e-6)
        for snr in (0,10,20):
            mask=raw['snr']==snr
            current=raw['current_loss'][mask];future=raw['future_losses'][mask]
            gain=current-future[:,future.shape[1]//2:].mean(1)
            for p in (.2,.3):
                k=round(len(current)*p);chosen=np.argsort(-raw['predicted_gain'][mask],kind='mergesort')[:k]
                actual=float(gain[chosen].sum()/len(current))
                stored=row['test'][str(snr)]['policies'][str(p)]['predicted_gain']['gain_db']
                assert abs(actual-stored)<1e-7
                policy_recomputations.append({'seed':seed,'snr':snr,'fraction':p,'passed':True})
    assert len(b['models'])==3
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in original.items())
    dump(ROOT/'audit.json',{'status':'PASS','direct_forward_exact':True,
                           'retransmission_noise_power_ratio':ratio,'anchor_bank_count':stats_count,
                           'interpolation_endpoints':True,'gain_prediction_ignores_labels':True,
                           'policy_recomputations':policy_recomputations})
    print('PASS: exact channel, fixed backbone, anchor banks/endpoints, label isolation, saved-policy recomputation')


if __name__=='__main__':main()
