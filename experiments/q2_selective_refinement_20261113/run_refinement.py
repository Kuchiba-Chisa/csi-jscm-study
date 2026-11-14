"""One shared residual CNN, frozen finite-QAM link; validation-fixed quality gate.

This is a feasibility experiment, not an RD-JSCC reproduction or novelty claim.
All refinements computed offline for evaluation only; deployment gate inputs
come strictly from the base decoder, before the optional module is executed.
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
import torch.nn as nn
import torch.nn.functional as F

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'two_sided_experiments_20261113'))
from run_cooperative import build,data,dump,sha,seed_all,bs_inputs,predict,Monitor
from common import nmse_db_per
from model_v4 import normalize_power,to_symbols,qam_map_ste,from_symbols
from quality_v5 import sgcs_per

PREV=ROOT/'two_sided_experiments_20261113'
SNRS=(0,10,20)


class Refiner(nn.Module):
    def __init__(self,width=8):
        super().__init__()
        self.conv1=nn.Conv2d(20,width,3,padding=1)
        self.conv2=nn.Conv2d(width,2,3,padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    def forward(self,z):
        return self.conv2(F.silu(self.conv1(z)))


def score_head(seed):
    bundle=torch.load(PREV/f'quality_bs_only_s{seed}.pt',weights_only=True)
    head=Monitor(bundle['input_dim'],bundle['budget'])
    head.load_state_dict(bundle['state_dict']);head.eval()
    return head,bundle


@torch.no_grad()
def decode_maps(net,y,snr,truth,bs=128):
    captured={}
    handle=net.dec.blocks[-1].af2.register_forward_hook(lambda m,a,o:captured.update(feature=o))
    columns={k:[] for k in ('maps','base','target','base_loss','base_sgcs')}
    try:
        for start in range(0,len(y),bs):
            yy=y[start:start+bs];s=snr[start:start+bs]
            base=net.dec(yy,10**(-s[:,None]/20))
            target=truth[start:start+bs]
            columns['maps'].append(torch.cat([base,captured['feature']],1))
            columns['base'].append(base)
            columns['target'].append(target)
            columns['base_loss'].append(nmse_db_per(base,target))
            columns['base_sgcs'].append(sgcs_per(base,target))
    finally:
        handle.remove()
    return {k:torch.cat(v) for k,v in columns.items()}


@torch.no_grad()
def train_cache(net,x,seed):
    y=[];snr=[];truth=[]
    for s in SNRS:
        for start in range(0,len(x),128):
            xb=x[start:start+128]
            af=torch.full((len(xb),1),10**(-s/20))
            sym=from_symbols(qam_map_ste(to_symbols(normalize_power(net.enc(xb,af),64))))
            gen=torch.Generator().manual_seed(1900000000+seed*10000000+s*10000+start)
            noise=torch.randn(sym.shape,generator=gen)*math.sqrt(net.p_ref_of(64,s)/2)*10**(-s/20)
            y.append(sym+noise);snr.append(torch.full((len(xb),),float(s)));truth.append(xb)
    snr=torch.cat(snr)
    result=decode_maps(net,torch.cat(y),snr,torch.cat(truth))
    result['snr']=snr
    return result


def model_input(maps,snr,mean,sd):
    normal=(maps-mean)/sd
    cond=torch.stack([snr/20,torch.ones_like(snr)],1)[:,:,None,None].expand(-1,-1,32,32)
    return torch.cat([normal,cond],1)


@torch.no_grad()
def outcomes(model,cache,mean,sd,bs=128):
    result={k:[] for k in ('loss','sgcs','correction_rms')}
    for ix in torch.arange(len(cache['snr'])).split(bs):
        correction=model(model_input(cache['maps'][ix],cache['snr'][ix],mean,sd))
        output=cache['base'][ix]+correction
        result['loss'].append(nmse_db_per(output,cache['target'][ix]))
        result['sgcs'].append(sgcs_per(output,cache['target'][ix]))
        result['correction_rms'].append(correction.square().flatten(1).mean(1).sqrt())
    return {k:torch.cat(v) for k,v in result.items()}


def train_refiner(train,val,seed,a):
    seed_all(812000+seed)
    mean=train['maps'].mean((0,2,3),keepdim=True)
    sd=train['maps'].std((0,2,3),keepdim=True).clamp_min(1e-4)
    model=Refiner(a.width)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    initial=float(val['base_loss'].mean())
    best=initial;selected=0;state=copy.deepcopy(model.state_dict());stale=0;trace=[]
    started=time.perf_counter()
    gen=torch.Generator().manual_seed(813000+seed)
    for ep in range(a.epochs):
        model.train();order=torch.randperm(len(train['snr']),generator=gen);total=0.
        for ix in order.split(128):
            output=train['base'][ix]+model(model_input(train['maps'][ix],train['snr'][ix],mean,sd))
            loss=F.mse_loss(output,train['target'][ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            total+=float(loss.detach())*len(ix)
        model.eval();v=float(outcomes(model,val,mean,sd)['loss'].mean())
        trace.append({'epoch':ep+1,'train_MSE':total/len(order),'val_mean_nmse_db':v})
        if v<best-1e-4:
            best=v;selected=ep+1;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        print('TRAIN',seed,ep+1,'val',round(v,5),'best',round(best,5),'seconds',round(time.perf_counter()-started,1),flush=True)
        if stale>=4:break
    model.load_state_dict(state);model.eval()
    bundle={'state_dict':state,'mean':mean,'sd':sd,'width':a.width,
            'meta':{'initial_val_nmse_db':initial,'best_val_nmse_db':best,
                    'selected_epoch':selected,'trace':trace,'seconds':time.perf_counter()-started}}
    torch.save(bundle,OUT/f'refiner_s{seed}.pt')
    return model,bundle


def subset(cache,ix):
    return {k:v[ix] for k,v in cache.items()}


def cluster_ci(values,h,seed,repeats=1000):
    n=int(h.max())+1
    sums=np.bincount(h,weights=values,minlength=n)
    count=np.bincount(h,minlength=n)
    rng=np.random.default_rng(seed)
    w=rng.multinomial(n,np.ones(n)/n,size=repeats)
    means=(w@sums)/(w@count)
    return np.quantile(means,[.025,.975]).tolist()


@torch.no_grad()
def direct_times(model,mean,sd,net,seed,a):
    torch.set_num_threads(1)
    z=torch.zeros(1,18,32,32);s=torch.tensor([10.])
    def refine():return model(model_input(z,s,mean,sd))
    for _ in range(20):refine()
    times=[]
    for _ in range(7):
        tick=time.perf_counter()
        for j in range(100):refine()
        times.append((time.perf_counter()-tick)*10)
    torch.set_num_threads(a.threads)
    return {'batch':1,'threads':1,'ms_blocks':times,'median_ms':float(np.median(times)),
            'note':'same PC CPU; normalization, condition concatenation and CNN included; not UE/BS hardware'}


def run(seed,a,truths):
    started=time.perf_counter();tr,va,te=truths
    net,source=build(seed,tr)
    original={k:v.clone() for k,v in net.state_dict().items()}
    for p in net.parameters():p.requires_grad_(False)
    paths=[PREV/f'cache_s{seed}_{k}.pt' for k in ('val','test')]+[PREV/f'quality_bs_only_s{seed}.pt']
    hashes={str(p):sha(p) for p in paths}
    old=json.loads((PREV/f'results_s{seed}.json').read_text(encoding='utf8'))
    assert source['sha256']==old['source']['sha256']
    assert source['source_model_sha256']==old['source']['source_model_sha256']
    valrows=torch.load(paths[0],weights_only=True)
    testrows=torch.load(paths[1],weights_only=True)
    print('CACHE START',seed,flush=True)
    train=train_cache(net,tr[:a.ntrain],seed)
    valid=decode_maps(net,valrows['y'],valrows['snr'],va[valrows['h_index']])
    valid['snr']=valrows['snr']
    assert float((valid['base_loss']-valrows['target']).abs().max())<1e-5
    select=valrows['h_index']<250
    gate=~select
    model,bundle=train_refiner(train,subset(valid,select),seed,a)
    del train
    valout=outcomes(model,valid,bundle['mean'],bundle['sd'])
    val_gain=valid['base_loss']-valout['loss']
    result={'seed':seed,'source':source,'source_hashes':hashes,'training':bundle['meta'],
            'validation_gate_H_mean_gain_db':float(val_gain[gate].mean()),'conditions':{}}
    if seed==0 and bundle['meta']['selected_epoch']==0:
        result['stop']='No validation improvement, epoch0 identity retained; no test evaluation or other seeds.'
        dump(OUT/f'results_s{seed}.json',result)
        return False
    print('REFINER PASS',seed,'validation_gate_gain',round(float(val_gain[gate].mean()),5),flush=True)
    test=decode_maps(net,testrows['y'],testrows['snr'],te[testrows['h_index']]);test['snr']=testrows['snr']
    assert float((test['base_loss']-testrows['target']).abs().max())<1e-5
    testout=outcomes(model,test,bundle['mean'],bundle['sd'])
    head,hbundle=score_head(seed)
    quality={k:predict(head,hbundle,bs_inputs(r)) for k,r in [('val',valrows),('test',testrows)]}
    result['cost']={'refiner_parameters':sum(p.numel() for p in model.parameters()),
                    'refiner_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                    'refiner_normalization_bytes':bundle['mean'].numel()*8,
                    'refiner_conv_linear_MACs':32*32*(20*a.width*9+a.width*2*9),
                    'monitor_parameters':sum(p.numel() for p in head.parameters()),
                    'monitor_conv_linear_MACs':sum(m.weight.numel() for m in head.modules() if isinstance(m,nn.Linear)),
                    'base_decoder_MACs':22340048,
                    'notes':'MAC excludes normalization/activation/pooling; storage always includes one refiner, even if rarely enabled'}
    arrays={'snr':testrows['snr'].numpy(),'h_index':testrows['h_index'].numpy(),
            'base_loss':test['base_loss'].numpy(),'refined_loss':testout['loss'].numpy(),
            'base_sgcs':test['base_sgcs'].numpy(),'refined_sgcs':testout['sgcs'].numpy(),
            'quality':quality['test'].numpy(),'val_snr':valrows['snr'].numpy(),
            'val_h_index':valrows['h_index'].numpy(),'val_quality':quality['val'].numpy(),
            'val_base_loss':valid['base_loss'].numpy(),'val_refined_loss':valout['loss'].numpy()}
    offset=result['cost']['monitor_conv_linear_MACs']/result['cost']['refiner_conv_linear_MACs']
    # All gates fixed on validation H250:500, with no test score quantiles.
    for s in SNRS:
        vi=(valrows['snr']==s)&gate;ti=testrows['snr']==s
        loss=test['base_loss'][ti].numpy();refined=testout['loss'][ti].numpy()
        gain=loss-refined
        s_gain=(testout['sgcs'][ti]-test['base_sgcs'][ti]).numpy()
        h=testrows['h_index'][ti].numpy()
        cell={'base_nmse_db':float(loss.mean()),'refined_all_nmse_db':float(refined.mean()),
              'all_gain_db':float(gain.mean()),'all_gain_CI95':cluster_ci(gain,h,870000+seed*100+s),
              'all_sgcs_gain':float(s_gain.mean()),'base_sgcs':float(test['base_sgcs'][ti].mean()),
              'refined_sgcs':float(testout['sgcs'][ti].mean()),
              'fraction_refinement_improves':float((gain>0).mean()),'gates':{}}
        for q in (.2,.5):
            cut=float(torch.quantile(quality['val'][vi],1-q))
            call=(quality['test'][ti]>=cut).numpy()
            rate=float(call.mean())
            policy_gain=gain*call
            random_calls=gain*rate
            random_compute=gain*min(1.,rate+offset)
            k=int(call.sum())
            oracle=np.sort(gain)[-k:].sum()/len(gain) if k else 0.
            delta=policy_gain-random_calls
            delta_compute=policy_gain-random_compute
            record={'threshold':cut,'validation_call_fraction':float((quality['val'][vi]>=cut).float().mean()),
                    'test_call_fraction':rate,'selected_observations':k,
                    'base_to_policy_gain_db':float(policy_gain.mean()),
                    'final_mean_nmse_db':float((loss-policy_gain).mean()),
                    'policy_sgcs_gain':float((s_gain*call).mean()),
                    'random_same_calls_gain_db':float(random_calls.mean()),
                    'SNR_only_same_calls_note':'Within fixed SNR, SNR-only cannot distinguish realizations; independent random enablement with same observed rate.',
                    'random_same_conv_linear_compute_call_fraction':min(1.,rate+offset),
                    'random_same_conv_linear_compute_gain_db':float(random_compute.mean()),
                    'quality_minus_random_calls_db':float(delta.mean()),
                    'quality_minus_random_calls_H_CI95':cluster_ci(delta,h,880000+seed*100+s+int(q*10)),
                    'quality_minus_random_compute_db':float(delta_compute.mean()),
                    'quality_minus_random_compute_H_CI95':cluster_ci(delta_compute,h,890000+seed*100+s+int(q*10)),
                    'hindsight_same_realization_oracle_gain_db':float(oracle),
                    'oracle_note':'True-H hindsight selection, not deployable; only diagnostic opportunity for this fixed refiner.',
                    'fraction_selected_samples_harmed':float((gain[call]<0).mean()) if k else 0.,
                    'mean_added_conv_linear_MACs':result['cost']['monitor_conv_linear_MACs']+rate*result['cost']['refiner_conv_linear_MACs']}
            cell['gates'][str(q)]=record
            arrays[f'gate_{s}_{q}']=call
        result['conditions'][str(s)]=cell
        print('RESULT',seed,s,'all_gain',round(cell['all_gain_db'],5),
              'q20', {k:round(cell['gates']['0.2'][k],5) for k in ('test_call_fraction','base_to_policy_gain_db','quality_minus_random_compute_db')},flush=True)
    result['timing']=direct_times(model,bundle['mean'],bundle['sd'],net,seed,a)
    result['audit']={'base_state_unchanged':all(torch.equal(v,net.state_dict()[k]) for k,v in original.items()),
                     'no_base_gradients':all(p.grad is None for p in net.parameters()),
                     'original_files_unchanged':all(sha(Path(p))==h for p,h in hashes.items()),
                     'gate_uses_base_features_only':True,'validation_selected_epoch_and_thresholds':True}
    assert all(result['audit'].values())
    np.savez_compressed(OUT/f'outcomes_s{seed}.npz',**arrays)
    result['seconds']=time.perf_counter()-started
    dump(OUT/f'results_s{seed}.json',result)
    print('DONE',seed,'seconds',round(result['seconds'],1),flush=True)
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0])
    ap.add_argument('--epochs',type=int,default=15)
    ap.add_argument('--ntrain',type=int,default=1200)
    ap.add_argument('--width',type=int,default=8)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    torch.set_num_threads(a.threads)
    protocol={'date':'2026-11-13','args':vars(a),'script_sha256':sha(Path(__file__)),
              'scope':'single small residual CNN feasibility, not diffusion or official RD-JSCC reproduction',
              'frozen_link':'existing RD-AE,finite16QAM,AWGN,Ns64,SNR0/10/20',
              'refiner':'20->8->2 conv3x3,SiLU,noBN,zero output initialization; same weights for all SNRs',
              'inputs':'Hhat0(2ch),last-block af2 feature(16ch),broadcast SNR/20 and Ns/64; no true H at inference',
              'splits':'refiner train first1200 trainH,earlystop valH0:250;gate threshold valH250:500;test600H*3noise',
              'prior_validation_reuse':'Existing quality head was previously selected using full500 validation H; not new independent calibration data.',
              'policies':'all,validation-fixed worst-quality gates20/50%,random/SNR-only same realized calls,random same conv-linear MAC budget,privileged hindsight',
              'budget':'No fixed top-k test gate; actual test invocation rate reported. Random controls matched actual invocation rates for diagnostic comparison.',
              'primary':'20% validation gate,paired per-SNR mean NMSE_dB improvement vs same-compute random; also SGCS and harm fraction',
              'stop':'If seed0 selected epoch0(no val improvement), stop before test and other seeds.'}
    oldpath=OUT/'protocol.json'
    if oldpath.exists():
        old=json.loads(oldpath.read_text(encoding='utf8'))
        assert old['script_sha256']==protocol['script_sha256']
        protocol['previous_runs']=old.get('previous_runs',[old['args']])+[vars(a)]
    dump(oldpath,protocol)
    truths=data();torch.set_num_threads(a.threads)
    for seed in a.seeds:
        if (OUT/f'results_s{seed}.json').exists():
            print('EXISTS',seed,'skip',flush=True);continue
        if not run(seed,a,truths):break


if __name__=='__main__':main()
