"""Prospectively locked SNR-error sensitivity and conservative BN adaptation.

BS-local estimate errors only. Encoder, actual channel, power reference and model
parameters stay fixed. Both a BN-only intervention and full BS-conditioning error
are measured, with input-matched unadapted baselines.
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

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'extra_experiments_20261113'))
from common import build,data,dump,sha
from model_v4 import normalize_power,to_symbols,qam_map_ste,from_symbols,pair_mask
from quality_v5 import nmse_db_per,sgcs_per

ANCHORS=OUT.parent/'deployment_followup_20261113'/'anchor_states'
LENGTHS=(16,32,64)
VAL_SNRS=(0,10,20)
VAL_ERRORS=(-3,0,3)
TEST_SNRS=(0,5,10,15,20)
TEST_ERRORS=(-3,-1,0,1,3)
MODES=('bn_only','all_receiver')
ALPHAS=(0.,.25,.5,.75,1.)


def ci(delta,h,seed):
    ids=np.unique(h)
    values=np.array([delta[h==i].mean() for i in ids])
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(1000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def stats_from_net(net):
    return {k:v.clone() for k,v in net.state_dict().items()
            if k.startswith('dec.') and k.endswith(('running_mean','running_var'))}


def anchors(seed,ns):
    return [torch.load(ANCHORS/f's{seed}_ns{ns}_{s}dB.pt',weights_only=True) for s in (0,20)]


def estimate(snr,error):
    return float(np.clip(snr+error,0,20))


def candidate(a,b,snr):
    t=snr/20
    return {k:(1-t)*a[k]+t*b[k] for k in a}


def blend(base,target,alpha):
    if alpha==0:return base
    if alpha==1:return target
    out={}
    for k in base:
        if not k.endswith('running_mean'):continue
        vk=k.replace('running_mean','running_var')
        m0,m1=base[k],target[k]
        v0,v1=base[vk],target[vk]
        mean=(1-alpha)*m0+alpha*m1
        # Population moments of a mixture: include between-mean variance.
        var=(1-alpha)*v0+alpha*v1+alpha*(1-alpha)*(m0-m1).square()
        out[k],out[vk]=mean,var.clamp_min(0)
    return out


@torch.no_grad()
def observe(net,x,snr,ns,seed,split):
    key={'val':460000000,'test':560000000}[split]+seed*10000000+snr*100000+ns*1000
    repeats=1 if split=='val' else 3
    yy,hh=[],[]
    for rep in range(repeats):
        for start in range(0,len(x),128):
            xb=x[start:start+128]
            af=torch.full((len(xb),1),10**(-snr/20))
            z=normalize_power(net.enc(xb,af),ns)
            tx=from_symbols(qam_map_ste(to_symbols(z)))
            gen=torch.Generator().manual_seed(key+rep*1000000+start)
            sigma=math.sqrt(net.p_ref_of(ns,snr)/2)*10**(-snr/20)
            received=(tx+sigma*torch.randn(tx.shape,generator=gen))*pair_mask(ns)
            yy.append(received);hh.append(torch.arange(start,start+len(xb)))
    return dict(y=torch.cat(yy),h=torch.cat(hh))


@torch.no_grad()
def decode(net,obs,x,decoder_snr,stats):
    net.load_state_dict(stats,strict=False)
    net.eval()
    loss,sgcs=[],[]
    for start in range(0,len(obs['h']),128):
        y=obs['y'][start:start+128]
        target=x[obs['h'][start:start+128]]
        rec=net.dec(y,torch.full((len(y),1),10**(-decoder_snr/20)))
        loss.append(nmse_db_per(rec,target));sgcs.append(sgcs_per(rec,target))
    return dict(loss=torch.cat(loss).numpy(),sgcs=torch.cat(sgcs).numpy())


def locked_protocol(args):
    return dict(args=vars(args),script_sha256=sha(Path(__file__)),
        scope='RD-AE PyTorch adaptation; not official complete RD-JSCC; frozen finite16QAM AWGN chain',
        intervention='BS-local SNR estimation error; UE encoder uses true SNR; physical tx/noise/mask and calibrated power fixed',
        modes=dict(bn_only='estimated SNR selects BN stats; AF receives true SNR',
                   all_receiver='estimated SNR selects BN stats and feeds receiver AF; encoder remains fixed'),
        estimate='clip(true_SNR+signed_bias,0,20); boundary points and interior reported separately; no extrapolation',
        validation=dict(H='val first250',draws=1,snrs=VAL_SNRS,errors=VAL_ERRORS),
        testing=dict(H='test600:1000',draws=3,snrs=TEST_SNRS,errors=TEST_ERRORS),
        candidates=ALPHAS,
        selection='per source seed, known Ns and mode: minimize maximum validation-cell mean loss regret relative to SAME-input unadapted receiver; ties within1e-7 choose larger alpha; alpha0 eligible; one scalar fixed across all true SNRs/errors',
        no_oracle='strength not selected per test point or true runtime SNR; no per-test loss access; known Ns and mode assumed configured',
        adaptation='0/20dB existing training-calibrated BN banks; linear interpolation of mean/variance at estimated SNR',
        conservative='moment mixture of original BN and estimated-condition bank: v=(1-a)v0+a*v1+a*(1-a)*(m0-m1)^2',
        baselines='S0 with same receiver AF information; accurate-information full adaptation separately retained; alpha1 vanilla adaptation',
        primary='paired NMSE gain relative to input-matched S0; count negative-gain deployment cells, worst mean gain, SGCS, accurate-vs-inaccurate difference; H-cluster CI',
        data_caveat='same historical project train/val/test populations; fresh deterministic noises; test not independent across all previous project work',
        bias_caveat='fixed signed SNR biases are sensitivity interventions, not a fitted physical estimator error distribution; validation and test share extreme bias magnitudes',
        algorithm_novelty='moment blending/shrinkage not claimed novel; estimates safety only under tested settings; no online training, no new calibration banks')


def select(net,x,seed,base):
    selected,trace={},[]
    for ns in LENGTHS:
        bank=anchors(seed,ns)
        cache={}
        means={mode:{a:[] for a in ALPHAS} for mode in MODES}
        for s in VAL_SNRS:
            obs=observe(net,x,s,ns,seed,'val')
            for mode in MODES:
                for e in VAL_ERRORS:
                    hat=estimate(s,e)
                    ds=s if mode=='bn_only' else hat
                    target=candidate(*bank,hat)
                    results={}
                    for a in ALPHAS:
                        key=(s,ds,hat if a else -1,a)
                        if key not in cache:
                            cache[key]=float(decode(net,obs,x,ds,blend(base,target,a))['loss'].mean())
                        results[a]=cache[key]
                    for a in ALPHAS:
                        means[mode][a].append(results[a]-results[0.])
                    trace.append(dict(ns=ns,true_snr=s,error=e,estimated_snr=hat,mode=mode,
                                      candidate_mean_nmse_db={str(a):v for a,v in results.items()}))
        for mode in MODES:
            worst={a:max(means[mode][a]) for a in ALPHAS}
            best=min(worst.values())
            alpha=max(a for a in ALPHAS if worst[a]<=best+1e-7)
            selected[f'{ns}_{mode}']=dict(alpha=alpha,worst_validation_regret_db={str(a):v for a,v in worst.items()},
                mean_validation_regret_db={str(a):float(np.mean(means[mode][a])) for a in ALPHAS})
            print('LOCK ALPHA',seed,ns,mode,alpha,{a:round(v,4) for a,v in worst.items()},flush=True)
    return selected,trace


def run(seed,args,truths):
    tick=time.perf_counter()
    tr,va,te=truths
    net,source=build(seed,tr)
    net.requires_grad_(False)
    initial=copy.deepcopy(net.state_dict())
    base=stats_from_net(net)
    params={k:v.clone() for k,v in net.named_parameters()}
    pcal=copy.deepcopy(net.p_cal)
    paths=[Path(source['checkpoint']),Path(__file__),Path(__file__).resolve().parents[1]/'extra_experiments_20261113/common.py']
    paths += [ANCHORS/f's{seed}_ns{ns}_{s}dB.pt' for ns in LENGTHS for s in (0,20)]
    hashes={str(p):sha(p) for p in paths}
    selection,trace=select(net,va[:250],seed,base)
    lock=dict(seed=seed,source=source,selection=selection,validation_trace=trace,input_hashes=hashes,
              locked_before_test=True,validation_seconds=time.perf_counter()-tick)
    dump(OUT/f'locked_s{seed}.json',lock)
    result=dict(lock,cells=[],accurate_reference=[],audit={})
    arrays={}
    for ns in LENGTHS:
        bank=anchors(seed,ns)
        for s in TEST_SNRS:
            obs=observe(net,te[600:1000],s,ns,seed,'test')
            prefix=f'ns{ns}_s{s}'
            arrays[prefix+'_h']=obs['h'].numpy()
            exact=decode(net,obs,te[600:1000],s,candidate(*bank,s))
            original=decode(net,obs,te[600:1000],s,base)
            arrays[prefix+'_accurate_loss']=exact['loss']
            arrays[prefix+'_accurate_sgcs']=exact['sgcs']
            result['accurate_reference'].append(dict(ns=ns,true_snr=s,
                S0_nmse_db=float(original['loss'].mean()),adapted_nmse_db=float(exact['loss'].mean()),
                gain_db=float((original['loss']-exact['loss']).mean())))
            cache={(s,-1,0.):original,(s,float(s),1.):exact}
            for mode in MODES:
                alpha=selection[f'{ns}_{mode}']['alpha']
                for e in TEST_ERRORS:
                    hat=estimate(s,e)
                    ds=s if mode=='bn_only' else hat
                    target=candidate(*bank,hat)
                    outputs={}
                    for label,a in [('S0',0.),('vanilla',1.),('conservative',alpha)]:
                        key=(ds,hat if a else -1,a)
                        if key not in cache:
                            cache[key]=decode(net,obs,te[600:1000],ds,blend(base,target,a))
                        outputs[label]=cache[key]
                    cell=dict(ns=ns,true_snr=s,error=e,estimated_snr=hat,effective_error=hat-s,mode=mode,alpha=alpha,methods={})
                    for label,out in outputs.items():
                        gain=outputs['S0']['loss']-out['loss']
                        key=f'{prefix}_{mode}_e{e}_{label}'
                        arrays[key+'_loss']=out['loss']
                        arrays[key+'_sgcs']=out['sgcs']
                        cell['methods'][label]=dict(nmse_db=float(out['loss'].mean()),sgcs=float(out['sgcs'].mean()),
                            gain_vs_input_matched_S0_db=float(gain.mean()),
                            gain_vs_input_matched_S0_H_CI95=ci(gain,obs['h'].numpy(),651000+seed*100+ns+s),
                            penalty_vs_accurate_adaptation_db=float((out['loss']-exact['loss']).mean()))
                    difference=outputs['vanilla']['loss']-outputs['conservative']['loss']
                    cell['conservative_minus_vanilla_gain_db']=float(difference.mean())
                    cell['conservative_minus_vanilla_gain_H_CI95']=ci(difference,obs['h'].numpy(),661000+seed*100+ns+s)
                    result['cells'].append(cell)
            print('TEST',seed,ns,s,'seconds',round(time.perf_counter()-tick,1),flush=True)
            dump(OUT/f'partial_s{seed}.json',dict(seed=seed,complete_cells=len(result['cells'])))
        print('LENGTH DONE',seed,ns,flush=True)
    assert all(torch.equal(params[k],v) for k,v in net.named_parameters())
    assert net.p_cal==pcal
    assert all(p.grad is None for p in net.parameters())
    assert all(sha(Path(p))==h for p,h in hashes.items())
    final=net.state_dict()
    changed=[k for k in initial if not torch.equal(initial[k],final[k])]
    assert all(k in base for k in changed)
    result['audit']=dict(all_weights_unchanged=True,power_reference_unchanged=True,
        no_gradients=True,original_hashes_unchanged=True,only_decoder_BN_moments_changed=True,
        same_received_y_within_comparison=True,selection_before_test=True)
    result['seconds']=time.perf_counter()-tick
    np.savez_compressed(OUT/f'outcomes_s{seed}.npz',**arrays)
    dump(OUT/f'results_s{seed}.json',result)
    print('DONE',seed,'seconds',round(result['seconds'],1),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    protocol=locked_protocol(args)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==json.loads(json.dumps(protocol)),'Locked design changed'
    else:dump(p,protocol)
    truths=data();torch.set_num_threads(args.threads)
    for seed in args.seeds:
        if (OUT/f'results_s{seed}.json').exists():print('SKIP',seed,flush=True)
        else:run(seed,args,truths)


if __name__=='__main__':main()
