"""Guo-style output-distilled proxy, adapted to a frozen noisy QAM CSI link.

Not an official Guo reproduction: CSI representation/backbone differ. A one-
block low-rank CNN proxy learns D(y,c). UE supplies clean codeword x, true H,
and independent simulated AWGN at inference; NEVER the actual received y.
"""
from __future__ import annotations
import argparse,copy,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'two_sided_experiments_20261113'
sys.path.insert(0,str(PREV))
from run_cooperative import (build,data,dump,seed_all,sha,fit_arrays,predict,
    bs_inputs,metrics,paired,quantizer,transmit_hint,classes_and_values,constellation,shuffle_by_H)
from common import nmse_db_per
from model_rdae import DecoderRDAE
from model_v4 import normalize_power,to_symbols,qam_map_ste,from_symbols
from quality_v5 import sgcs_per
from run_actions import evaluate


def make_proxy(teacher,rank=48):
    proxy=DecoderRDAE(residual_num=1)
    state=teacher.state_dict()
    proxy.load_state_dict({k:v for k,v in state.items() if k in proxy.state_dict()})
    # Own frozen teacher is accessible in this project. Explicit warm start,
    # not a claim of black-box/inter-vendor training as in Guo's deployment.
    proxy.fc=nn.Sequential(nn.Linear(128,rank,bias=False),nn.Linear(rank,2048))
    with torch.no_grad():
        u,s,vh=torch.linalg.svd(teacher.fc.weight,full_matrices=False)
        proxy.fc[0].weight.copy_(s[:rank].sqrt()[:,None]*vh[:rank])
        proxy.fc[1].weight.copy_(u[:,:rank]*s[:rank].sqrt()[None,:])
        proxy.fc[1].bias.copy_(teacher.fc.bias)
    return proxy


@torch.no_grad()
def teacher_set(net,x,base,bs=128):
    columns={k:[] for k in ['symbols','snr','output']}
    for si,s in enumerate([0.,10.,20.]):
        for start in range(0,len(x),bs):
            xb=x[start:start+bs];n=len(xb);c=torch.full((n,1),10**(-s/20))
            z=from_symbols(qam_map_ste(to_symbols(normalize_power(net.enc(xb,c),64))))
            gen=torch.Generator().manual_seed(base+si*1000000+start)
            sd=math.sqrt(float(net.p_ref_of(64,s)))*10**(-s/20)/math.sqrt(2)
            y=z+torch.randn(z.shape,generator=gen)*sd
            columns['symbols'].append(y);columns['snr'].append(torch.full((n,),s))
            columns['output'].append(net.dec(y,c))
    return {k:torch.cat(v) for k,v in columns.items()}


def train_proxy(net,x,val,seed,a):
    tick=time.perf_counter();seed_all(1800000+seed)
    train=teacher_set(net,x,1400000000+seed*10000000)
    validation=teacher_set(net,val,1500000000+seed*10000000)
    model=make_proxy(net.dec,a.rank)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    af=lambda r:10**(-r['snr'][:,None]/20)
    vfeat=af(validation);tfeat=af(train)
    best=float('inf');stale=0;selected=0;trace=[];state=None
    gen=torch.Generator().manual_seed(1810000+seed)
    @torch.no_grad()
    def val_loss():
        model.eval();num=den=0.
        for ix in torch.arange(len(vfeat)).split(128):
            e=(model(validation['symbols'][ix],vfeat[ix])-validation['output'][ix]).square()
            num+=float(e.sum());den+=e.numel()
        return num/den
    initial=val_loss()
    for ep in range(a.epochs):
        model.train()
        order=torch.randperm(len(tfeat),generator=gen)
        for ix in order.split(128):
            loss=F.mse_loss(model(train['symbols'][ix],tfeat[ix]),train['output'][ix])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        score=val_loss();trace.append(score)
        if score<best*(1-1e-4):
            best=score;state=copy.deepcopy(model.state_dict());selected=ep+1;stale=0
        else:
            stale+=1
        if (ep+1)%5==0 or ep==0:print(f'KD seed={seed} epoch={ep+1} val_mse={score:.7f} best={best:.7f} seconds={time.perf_counter()-tick:.1f}',flush=True)
        if stale>=7:break
    model.load_state_dict(state);model.eval()
    meta={'parameters':sum(p.numel() for p in model.parameters()),
          'teacher_parameters':sum(p.numel() for p in net.dec.parameters()),
          'initial_val_mse':initial,'selected_epoch':selected,'val_mse':best,'val_trace':trace,
          'seconds':time.perf_counter()-tick,'loss':'pure MSE to frozen teacher output on noisy inputs',
          'warm_start':'teacher first residual block and other compatible layers; rank48 SVD factorized FC',
          'train_H':len(x),'validation_H':len(val),'train_noise_draws_per_H_condition':1}
    torch.save({'state_dict':state,'rank':a.rank,'meta':meta},OUT/f'proxy_s{seed}.pt')
    return model,meta


@torch.no_grad()
def ue_scores(net,proxy,x,mc,base,teacher_draws=0,bs=128):
    rows={k:[] for k in ['snr','h_index','proxy_clean','proxy_mc','proxy_mc_sgcs','teacher_mc','kd_relative_error']}
    for si,s in enumerate([0.,10.,20.]):
        tick=time.perf_counter()
        for start in range(0,len(x),bs):
            xb=x[start:start+bs];n=len(xb);c=torch.full((n,1),10**(-s/20))
            z=from_symbols(qam_map_ste(to_symbols(normalize_power(net.enc(xb,c),64))))
            clean=proxy(z,c)
            sd=math.sqrt(float(net.p_ref_of(64,s)))*10**(-s/20)/math.sqrt(2)
            pl=[];ps=[];tl=[];kd=[]
            for j in range(max(mc,teacher_draws)):
                # Locally generated noise, disjoint from current BS cache seeds.
                gen=torch.Generator().manual_seed(base+si*1000000+start*100+j)
                y=z+torch.randn(z.shape,generator=gen)*sd
                if j<mc:
                    p=proxy(y,c);pl.append(nmse_db_per(p,xb));ps.append(sgcs_per(p,xb))
                if j<teacher_draws:
                    t=net.dec(y,c);tl.append(nmse_db_per(t,xb))
                    if j<mc:kd.append(((p-t)**2).flatten(1).sum(1)/t.square().flatten(1).sum(1).clamp_min(1e-12))
            rows['snr'].append(torch.full((n,),s));rows['h_index'].append(torch.arange(start,start+n))
            rows['proxy_clean'].append(nmse_db_per(clean,xb))
            rows['proxy_mc'].append(torch.stack(pl).mean(0));rows['proxy_mc_sgcs'].append(torch.stack(ps).mean(0))
            rows['teacher_mc'].append(torch.stack(tl).mean(0) if tl else torch.zeros(n))
            rows['kd_relative_error'].append(torch.stack(kd).mean(0) if kd else torch.zeros(n))
        print(f'UE SCORES H={len(x)} snr={s} MC={mc} teacher={teacher_draws} seconds={time.perf_counter()-tick:.1f}',flush=True)
    return {k:torch.cat(v) for k,v in rows.items()}


def align(scores,rows,key):
    # scores H index is local to its split, as in previous cached observations.
    out=torch.empty(len(rows['snr']))
    for s in [0.,10.,20.]:
        q=scores[key][scores['snr']==s]
        ix=rows['snr']==s;out[ix]=q[rows['h_index'][ix]]
    return out


def scalar_calibration(val,q,target):
    table={}
    for s in [0.,10.,20.]:
        ix=val['snr']==s;v=q[ix];y=target[ix]
        slope=((v-v.mean())*(y-y.mean())).mean()/v.var(unbiased=False).clamp_min(1e-8)
        slope=slope.clamp_min(0)
        table[s]=(float(slope),float(y.mean()-slope*v.mean()))
    return table


def apply_cal(rows,q,table):
    out=torch.empty_like(q)
    for s,(a,b) in table.items():
        ix=rows['snr']==s;out[ix]=a*q[ix]+b
    return out


@torch.no_grad()
def cost(proxy,teacher,rank):
    counts={}
    def hook(m,inp,out):
        if isinstance(m,nn.Linear):counts['macs']+=out.numel()*m.in_features
        elif isinstance(m,nn.Conv2d):counts['macs']+=out.numel()*(m.in_channels//m.groups)*math.prod(m.kernel_size)
    results={}
    y=torch.randn(32,128);c=torch.ones(32,1)
    for name,net in [('proxy',proxy),('BS_decoder',teacher)]:
        net.eval();counts['macs']=0;handles=[m.register_forward_hook(hook) for m in net.modules() if isinstance(m,(nn.Linear,nn.Conv2d))]
        net(y[:1],c[:1]);macs=counts['macs']
        for h in handles:h.remove()
        for _ in range(3):net(y,c)
        tick=time.perf_counter()
        for _ in range(8):net(y,c)
        results[name]={'parameters':sum(p.numel() for p in net.parameters()),'conv_linear_MACs_per_sample':macs,
                       'batched_cpu_ms_per_sample':(time.perf_counter()-tick)/8/32*1000}
    results['proxy_MC8_MACs']=8*results['proxy']['conv_linear_MACs_per_sample']
    results['note']='conv/linear MAC counts exclude normalization and activation; CPU batch32 timing excludes encoder and quality reductions'
    return results


def run(seed,a):
    tick=time.perf_counter();tr,va,te=data();net,source=build(seed,tr)
    initial={k:v.clone() for k,v in net.state_dict().items()}
    proxy,meta=train_proxy(net,tr[:3000],va[:500],seed,a)
    rows={k:torch.load(PREV/f'cache_s{seed}_{k}.pt',weights_only=True) for k in ['train','val','test']}
    truths={'train':tr[3000:5000],'val':va[:500],'test':te[:600]}
    score_rows={};q={}
    for i,split in enumerate(['train','val','test']):
        score_rows[split]=ue_scores(net,proxy,truths[split],a.mc,1600000000+seed*10000000+i*100000,
                                    teacher_draws=a.reference_draws if split=='test' else 0)
        q[split]={key:align(score_rows[split],rows[split],key) for key in ['proxy_clean','proxy_mc','teacher_mc']}
    calibration=scalar_calibration(rows['val'],q['val']['proxy_mc'],rows['val']['target'])
    for split in rows:q[split]['proxy_mc_calibrated']=apply_cal(rows[split],q[split]['proxy_mc'],calibration)
    result={'seed':seed,'source':source,'args':vars(a),'script_sha256':sha(Path(__file__)),
            'proxy':meta,'cost':cost(proxy,net.dec,a.rank),'quality':{},'paired':{},
            'calibration':calibration,'hint':{},'interpretation':'Guo-inspired output distillation on own CNN AE; not Guo paper reproduction'}
    rawpred={}
    for key in ['proxy_clean','proxy_mc','proxy_mc_calibrated','teacher_mc']:
        pred=q['test'][key];rawpred[key]=pred
        result['quality'][key]={'test':metrics(rows['test'],pred)}
    for s in [0.,10.,20.]:
        r=score_rows['test'];v=r['kd_relative_error'][r['snr']==s]
        result.setdefault('proxy_fidelity',{})[str(int(s))]={'mean_relative_MSE_to_teacher':float(v.mean()),
                        'dB_of_mean_relative_MSE':10*math.log10(max(float(v.mean()),1e-12))}
    # Show that scalar calibration is not exclusive to the proxy baseline.
    oldcal=scalar_calibration(rows['val'],rows['val']['ue_score'],rows['val']['target'])
    oldp=apply_cal(rows['test'],rows['test']['ue_score'],oldcal)
    result['quality']['old_ue_MLP_calibrated']={'test':metrics(rows['test'],oldp)}
    hintvals={};hintenergy={}
    for bits in [2,4]:
        tab=quantizer(rows['train'],q['train']['proxy_mc_calibrated'],bits)
        vals={};stats={};scales=None
        for split,r in rows.items():
            score=q[split]['proxy_mc_calibrated']
            cls,clean=classes_and_values(r,score,tab)
            noisy,rx,scales,st=transmit_hint(r,score,tab,bits,scales,
                          base=500000+{'train':10000,'val':20000,'test':30000}[split])
            vals[split]={'clean':clean,'noisy':noisy};stats[split]=st
            if split=='test':
                energy=torch.empty(len(score));pts=constellation(bits)
                for s in [0.,10.,20.]:
                    ix=r['snr']==s
                    energy[ix]=pts[cls[ix]].square().sum(1)*(scales[s]**2)*float(net.p_ref_of(64,s))
                hintenergy[bits]=energy
        hintvals[bits]=vals
        result['hint'][str(bits)]={'stats':stats,'scales':scales,'table':{str(s):{k:v.tolist() for k,v in t.items()} for s,t in tab.items()}}
    oldpred=np.load(PREV/f'quality_predictions_s{seed}.npz')
    oldbs=torch.from_numpy(oldpred['bs_only'].copy())
    preds={'bs_only':oldbs}
    for key,pred in rawpred.items():result['paired'][key+'_minus_bs']=paired(rows['test'],pred,oldbs)
    quality_models={}
    # Same BS-side capacity and train/val/test samples as preceding experiment.
    for name,kind in [('proxy_fusion_float','float'),('proxy_fusion_clean2','clean2'),
                      ('proxy_fusion_noisy2','noisy2'),('proxy_fusion_noisy4','noisy4')]:
        ins={}
        for split,r in rows.items():
            v=q[split]['proxy_mc_calibrated'] if kind=='float' else hintvals[int(kind[-1])][split]['clean' if kind.startswith('clean') else 'noisy']
            ins[split]=torch.cat([bs_inputs(r),v[:,None]],1)
        head,bundle,info=fit_arrays(ins['train'],rows['train']['target'],ins['val'],rows['val']['target'],710000+seed,25000,65)
        pred=predict(head,bundle,ins['test']);preds[name]=pred
        info['test']=metrics(rows['test'],pred);result['quality'][name]=info
        result['paired'][name+'_minus_bs']=paired(rows['test'],pred,oldbs)
        torch.save(bundle,OUT/f'{name}_s{seed}.pt');quality_models[name]=(head,bundle)
        if kind=='float':
            shuffled=shuffle_by_H(rows['test'],q['test']['proxy_mc_calibrated'],820000+seed)
            sp=predict(head,bundle,torch.cat([bs_inputs(rows['test']),shuffled[:,None]],1))
            result['quality']['proxy_fusion_shuffled']={'test':metrics(rows['test'],sp)}
        print('QUALITY seed='+str(seed)+' '+name+' '+str({s:round(v['mae_db'],4) for s,v in info['test'].items()}),flush=True)
    # Reuse independent future simulation from the preceding experiment.
    future={k:torch.load(PREV/f'future_s{seed}_{k}.pt',weights_only=True) for k in rows}
    targets={k:rows[k]['target']-future[k]['loss'].mean(1) for k in rows}
    existing=np.load(PREV/f'action_predictions_s{seed}.npz')
    actions={'bs_gain':torch.from_numpy(existing['bs_gain'].copy())}
    for name,kind in [('proxy_float_gain','float'),('proxy_clean2_gain','clean2'),
                      ('proxy_noisy2_gain','noisy2'),('proxy_noisy4_gain','noisy4'),('proxy_ue_noisy2_gain','ueonly')]:
        ins={}
        for split,r in rows.items():
            if kind=='ueonly':X=torch.cat([r['condition'],hintvals[2][split]['noisy'][:,None]],1)
            else:
                v=q[split]['proxy_mc_calibrated'] if kind=='float' else hintvals[int(kind[-1])][split]['clean' if kind.startswith('clean') else 'noisy']
                X=torch.cat([bs_inputs(r),v[:,None]],1)
            ins[split]=X
        head,bundle,info=fit_arrays(ins['train'],targets['train'],ins['val'],targets['val'],930000+seed,25000,65)
        actions[name]=predict(head,bundle,ins['test'])
        result.setdefault('action_heads',{})[name]=info
        torch.save(bundle,OUT/f'{name}_s{seed}.pt')
    result['actions']=evaluate(rows['test'],actions,future['test']['loss'],future['test']['sgcs'],
                       {k:v for k,v in preds.items() if k in ['bs_only','proxy_fusion_noisy2','proxy_fusion_noisy4']},hintenergy)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in initial.items()),'Frozen backbone changed'
    # Serialized raw UE scores allow confirming identical scores for every
    # current noise repetition of H, and reproducing all hint quantization.
    torch.save({'scores':score_rows,'calibration':calibration,'aligned':q},OUT/f'ue_scores_s{seed}.pt')
    np.savez_compressed(OUT/f'quality_predictions_s{seed}.npz',**{k:v.numpy() for k,v in preds.items()},
                       **{k:v.numpy() for k,v in rawpred.items()})
    np.savez_compressed(OUT/f'action_predictions_s{seed}.npz',**{k:v.numpy() for k,v in actions.items()})
    result['duration_s']=time.perf_counter()-tick
    dump(OUT/f'results_s{seed}.json',result)
    print(f'DONE seed={seed} seconds={result["duration_s"]:.1f}',flush=True)
    for s in ['0','10','20']:
        print('ACTION '+s+' '+str(result['actions'][s]['paired_net_gain_db']),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[0])
    ap.add_argument('--epochs',type=int,default=45);ap.add_argument('--rank',type=int,default=48)
    ap.add_argument('--mc',type=int,default=8);ap.add_argument('--reference-draws',type=int,default=32)
    ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();torch.set_num_threads(a.threads)
    dump(OUT/'protocol.json',{'args':vars(a),'script_sha256':sha(Path(__file__)),
        'source_protocol_sha256':sha(PREV/'protocol.json'),
        'method':'Output MSE distillation, one residual block versus five; rank48 FC warm-start from own teacher',
        'UE_information':'true source H, deterministic finite-QAM codeword, known SNR; own independent simulated AWGN',
        'scope':'Guo-inspired noisy-feedback adaptation, NOT paper reproduction or a test of official Guo results',
        'selection':'proxy chosen by validation KD MSE; quality/decision heads chosen by their validation MSE; no test selection',
        'split':'proxy train H[0:3000], BS train H[3000:5000], val first500 independent val H, test first600 independent test H'})
    for s in a.seeds:run(s,a)

if __name__=='__main__':main()
