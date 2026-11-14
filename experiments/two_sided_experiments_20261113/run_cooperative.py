"""Frozen finite-QAM CSI backbone: deployable UE hint + BS quality fusion.

UE fit and BS fit use disjoint H. Repeated noises stay with the same H.
No current realization or true test loss enters the UE input.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'extra_experiments_20261113'))
from common import build,data,dump,nmse_db_per,seed_all,sha
from q2_quality_networks import Monitor
from quality_v5 import sgcs_per,auroc,spearman
from model_v4 import normalize_power,to_symbols,qam_map_ste,add_noise,from_symbols,pair_mask


@torch.no_grad()
def cache(net,x,snrs,repeats,base,ns=64,bs=128,ue_only=False):
    net.eval();buffers={}
    def ue_hook(m,a,z): buffers['ue']=z.detach().flatten(1)
    def bs_hook(m,a,z): buffers['bs']=F.adaptive_avg_pool2d(z.detach(),(2,2)).flatten(1)
    handles=[net.enc.af2.register_forward_hook(ue_hook),net.dec.blocks[-1].af2.register_forward_hook(bs_hook)]
    columns={k:[] for k in ('ue','condition','snr','h_index','target')}
    if not ue_only:
        columns.update({k:[] for k in ('feature','y','symbols','sgcs')})
    try:
        for si,snr in enumerate(snrs):
            start_t=time.perf_counter()
            for start in range(0,len(x),bs):
                xb=x[start:start+bs];n=len(xb)
                st=torch.full((n,),float(snr));nt=torch.full((n,),ns,dtype=torch.long)
                af=(10**(-st/20)).view(-1,1)
                latent=net.enc(xb,af)
                uf=torch.cat([buffers['ue'],latent],1).clone()
                sym=qam_map_ste(to_symbols(normalize_power(latent,nt)))
                p=torch.full((n,),float(net.p_ref_of(ns,snr)))
                losses=[]
                for rep in range(repeats):
                    seed_all(base+si*1000000+rep*100000+start)
                    yn,_,_=add_noise(sym,nt,st,p)
                    y=from_symbols(yn)*pair_mask(nt)
                    rec=net.dec(y,af)
                    loss=nmse_db_per(rec,xb)
                    losses.append(loss)
                    if not ue_only:
                        columns['ue'].append(uf)
                        columns['condition'].append(torch.stack([st/20,nt.float()/64],1))
                        columns['snr'].append(st);columns['h_index'].append(torch.arange(start,start+n))
                        columns['target'].append(loss)
                        columns['feature'].append(buffers['bs'].clone())
                        columns['y'].append(y);columns['symbols'].append(from_symbols(sym))
                        columns['sgcs'].append(sgcs_per(rec,xb))
                if ue_only:
                    columns['ue'].append(uf)
                    columns['condition'].append(torch.stack([st/20,nt.float()/64],1))
                    columns['snr'].append(st);columns['h_index'].append(torch.arange(start,start+n))
                    columns['target'].append(torch.stack(losses).mean(0))
            print(f'CACHE base={base} snr={snr} nH={len(x)} draws={repeats} ue_only={ue_only} seconds={time.perf_counter()-start_t:.1f}',flush=True)
    finally:
        for h in handles:h.remove()
    return {k:torch.cat(v).cpu() for k,v in columns.items()}


def fit_arrays(X,Y,V,Yv,seed,budget=25000,epochs=65):
    seed_all(seed)
    mean,sd=X.mean(0),X.std(0).clamp_min(1e-4)
    X,V=(X-mean)/sd,(V-mean)/sd
    tm,ts=Y.mean(),Y.std().clamp_min(1e-4)
    Y,Yv=(Y-tm)/ts,(Yv-tm)/ts
    head=Monitor(X.shape[1],budget)
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-3)
    best=float('inf');state=None;stale=0;trace=[];selected=0
    gen=torch.Generator().manual_seed(seed+1)
    for ep in range(epochs):
        head.train();order=torch.randperm(len(X),generator=gen)
        for ix in order.split(512):
            loss=(head(X[ix])-Y[ix]).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():score=float((head(V)-Yv).square().mean())
        trace.append(score*float(ts**2))
        if score<best-1e-5:
            best=score;state=copy.deepcopy(head.state_dict());stale=0;selected=ep+1
        else:
            stale+=1
            if stale>=9:break
    head.load_state_dict(state);head.eval()
    bundle={'state_dict':state,'mean':mean,'sd':sd,'target_mean':tm,'target_sd':ts,
            'input_dim':X.shape[1],'budget':budget}
    return head,bundle,{'parameters':sum(p.numel() for p in head.parameters()),'selected_epoch':selected,
                        'val_mse_db2':best*float(ts**2),'trace_val_mse_db2':trace}


@torch.no_grad()
def predict(head,bundle,X):
    return torch.cat([head((xx-bundle['mean'])/bundle['sd'])*bundle['target_sd']+bundle['target_mean'] for xx in X.split(1024)]).cpu()


def ue_inputs(r):return torch.cat([r['ue'],r['condition']],1)
def bs_inputs(r):return torch.cat([r['feature'],r['condition']],1)


def quantizer(fit_rows,score,bits):
    out={};n=2**bits
    for snr in fit_rows['snr'].unique().tolist():
        q=score[fit_rows['snr']==snr]
        cuts=torch.quantile(q,torch.arange(1,n).float()/n)
        cls=torch.bucketize(q.contiguous(),cuts)
        centers=torch.stack([q[cls==i].mean() if (cls==i).any() else torch.quantile(q,(i+.5)/n) for i in range(n)])
        out[float(snr)]={'cuts':cuts,'centers':centers}
    return out


def classes_and_values(r,score,table):
    c=torch.zeros(len(score),dtype=torch.long);v=torch.zeros_like(score)
    for s,t in table.items():
        ix=r['snr']==s
        c[ix]=torch.bucketize(score[ix].contiguous(),t['cuts'])
        v[ix]=t['centers'][c[ix]]
    return c,v


def constellation(bits):
    if bits==2:
        return torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])/math.sqrt(2)
    assert bits==4
    lev=torch.tensor([-3.,-1.,3.,1.])/math.sqrt(10) # binary labels -> Gray PAM levels
    return torch.stack([torch.tensor([lev[i//4],lev[i%4]]) for i in range(16)])


def transmit_hint(r,score,table,bits,scales=None,base=400):
    c,clean=classes_and_values(r,score,table);pts=constellation(bits)
    received=torch.empty_like(c);values=torch.empty_like(score)
    if scales is None:
        scales={float(s):float((pts[c[r['snr']==s]]**2).sum(1).mean().rsqrt()) for s in r['snr'].unique().tolist()}
    stats={}
    for s,t in table.items():
        ix=r['snr']==s;n=int(ix.sum());scale=scales[s]
        tx=pts[c[ix]]*scale # normalized ensemble energy one; main P_ref divides out at receiver
        generator=torch.Generator().manual_seed(base+int(s)*1000+bits)
        noise=torch.randn((n,2),generator=generator)*(10**(-s/20)/math.sqrt(2))
        y=tx+noise
        got=((y[:,None,:]-pts[None,:,:]*scale)**2).sum(2).argmin(1)
        received[ix]=got;values[ix]=t['centers'][got]
        stats[str(int(s))]={'symbol_error_rate':float((got!=c[ix]).float().mean()),
                           'measured_energy_relative_to_main_Pref':float(tx.square().sum(1).mean()),
                           'scale':scale}
    return values,received,scales,stats


def shuffle_by_H(r,q,seed):
    out=q.clone();gen=torch.Generator().manual_seed(seed)
    for s in r['snr'].unique().tolist():
        ix=torch.where(r['snr']==s)[0];h=r['h_index'][ix];ids=h.unique(sorted=True)
        vals=torch.stack([q[ix[h==i][0]] for i in ids])
        vals=vals[torch.randperm(len(ids),generator=gen)]
        out[ix]=vals[torch.searchsorted(ids,h)]
    return out


def metrics(r,pred):
    result={}
    for s in r['snr'].unique().tolist():
        ix=(r['snr']==s);y=r['target'][ix].numpy();p=pred[ix].numpy()
        bad=y>=np.quantile(y,.8)
        order=np.argsort(p);keep=order[:round(.7*len(y))]
        denom=float(((y-y.mean())**2).sum())
        result[str(int(s))]={'mae_db':float(np.abs(p-y).mean()),'mse_db2':float(((p-y)**2).mean()),
            'r2':1-float(((p-y)**2).sum())/max(denom,1e-12),'spearman':spearman(p,y),'auroc_worst20':auroc(p,bad),
            'risk_keep70_nmse_db':float(y[keep].mean()),'sgcs_keep70':float(r['sgcs'][ix][keep].mean()),
            'all_nmse_db':float(y.mean()),'all_sgcs':float(r['sgcs'][ix].mean()),
            'observations':len(y),'unique_H':int(r['h_index'][ix].unique().numel())}
    return result


def paired(r,a,b):
    # a-b in absolute loss: negative favors a; H-cluster CIs for fixed fitted heads.
    out={};rng=np.random.default_rng(81200)
    for s in r['snr'].unique().tolist():
        ix=r['snr']==s;h=r['h_index'][ix];y=r['target'][ix]
        delta=(a[ix]-y).abs()-(b[ix]-y).abs()
        v=np.array([float(delta[h==i].mean()) for i in h.unique()])
        samples=v[rng.integers(0,len(v),(1500,len(v)))].mean(1)
        out[str(int(s))]={'mae_a_minus_b_db':float(v.mean()),'H_cluster_CI95':np.quantile(samples,[.025,.975]).tolist()}
    return out


def run_seed(a,seed):
    begin=time.perf_counter();tr,va,te=data();net,source=build(seed,tr)
    original={k:v.clone() for k,v in net.state_dict().items()}
    assert a.nue+a.nfusion<=len(tr)
    snrs=(0,10,20)
    u=cache(net,tr[:a.nue],snrs,a.ue_draws,110000000,ue_only=True)
    train=cache(net,tr[a.nue:a.nue+a.nfusion],snrs,a.train_repeats,210000000)
    val=cache(net,va[:a.nval],snrs,2,310000000)
    test=cache(net,te[:a.ntest],snrs,a.test_repeats,410000000)
    ue,ub,um=fit_arrays(ue_inputs(u),u['target'],ue_inputs(val),val['target'],700000+seed,100000,a.epochs)
    torch.save(ub,OUT/f'ue_head_s{seed}.pt')
    scores={k:predict(ue,ub,ue_inputs(r)) for k,r in [('u',u),('train',train),('val',val),('test',test)]}
    print(f'UE seed={seed} parameters={um["parameters"]} val_mse={um["val_mse_db2"]:.4f}',flush=True)
    result={'seed':seed,'source':source,'ue':um,'quality':{'ue_only':{'test':metrics(test,scores['test'])}},'paired':{},'hint':{}}
    hints={}
    for bits in (2,4):
        tab=quantizer(u,scores['u'],bits)
        values={};hintstats={};scales=None
        for label,r in [('train',train),('val',val),('test',test)]:
            cls,clean=classes_and_values(r,scores[label],tab)
            noisy,rx,scales,stats=transmit_hint(r,scores[label],tab,bits,scales,base=500000+{'train':10000,'val':20000,'test':30000}[label])
            values[label]={'clean':clean,'noisy':noisy,'class':cls,'received_class':rx}
            hintstats[label]=stats
        hints[bits]=values
        result['hint'][str(bits)]={'stats':hintstats,'table':{str(s):{k:v.tolist() for k,v in t.items()} for s,t in tab.items()},'scales':scales}
    prediction={'ue_only':scores['test']}
    modes=[('bs_only',None,25000),('bs_large',None,100000),('fusion_float','float',25000),
           ('fusion_clean2','clean2',25000),('fusion_clean4','clean4',25000),
           ('fusion_noisy2','noisy2',25000),('fusion_noisy4','noisy4',25000)]
    for mi,(name,kind,budget) in enumerate(modes):
        ins={}
        for label,r in [('train',train),('val',val),('test',test)]:
            X=bs_inputs(r)
            if kind:
                q=scores[label] if kind=='float' else hints[int(kind[-1])][label]['clean' if kind.startswith('clean') else 'noisy']
                X=torch.cat([X,q[:,None]],1)
            ins[label]=X
        head,bundle,meta=fit_arrays(ins['train'],train['target'],ins['val'],val['target'],710000+seed, budget,a.epochs)
        p=predict(head,bundle,ins['test']);prediction[name]=p
        meta['test']=metrics(test,p);result['quality'][name]=meta
        torch.save(bundle,OUT/f'quality_{name}_s{seed}.pt')
        if name=='fusion_float':
            shuffled=shuffle_by_H(test,scores['test'],820000+seed)
            ps=predict(head,bundle,torch.cat([bs_inputs(test),shuffled[:,None]],1))
            prediction['fusion_float_shuffled']=ps
            result['quality']['fusion_float_shuffled']={'test':metrics(test,ps),'note':'test-time H-level shuffle within SNR; fixed fusion head'}
        print('QUALITY seed='+str(seed)+' '+name+' '+str({s:{k:round(m[k],4) for k in ['mae_db','auroc_worst20']} for s,m in meta['test'].items()}),flush=True)
    for name,p in prediction.items():
        if name!='bs_only':result['paired'][name+'_minus_bs']=paired(test,p,prediction['bs_only'])
    for bits in (2,4):
        result['quality'][f'ue_only_noisy{bits}']={'test':metrics(test,hints[bits]['test']['noisy'])}
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in original.items()),'Backbone mutated'
    for label,r in [('train',train),('val',val),('test',test)]:
        # UE full feature not required by the follow-up; never expose it to BS.
        cache_rows={k:v for k,v in r.items() if k!='ue'}
        cache_rows['ue_score']=scores[label]
        for bits in (2,4):
            cache_rows[f'hint_clean{bits}']=hints[bits][label]['clean']
            cache_rows[f'hint_noisy{bits}']=hints[bits][label]['noisy']
        torch.save(cache_rows,OUT/f'cache_s{seed}_{label}.pt')
    np.savez_compressed(OUT/f'quality_predictions_s{seed}.npz',**{k:v.numpy() for k,v in prediction.items()},
                        target=test['target'].numpy(),h_index=test['h_index'].numpy(),snr=test['snr'].numpy())
    result['duration_s']=time.perf_counter()-begin
    dump(OUT/f'results_s{seed}.json',result)
    print(f'SEED DONE {seed} seconds={result["duration_s"]:.1f}',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0])
    ap.add_argument('--nue',type=int,default=3000);ap.add_argument('--nfusion',type=int,default=2000)
    ap.add_argument('--nval',type=int,default=500);ap.add_argument('--ntest',type=int,default=600)
    ap.add_argument('--ue-draws',type=int,default=8);ap.add_argument('--train-repeats',type=int,default=3)
    ap.add_argument('--test-repeats',type=int,default=3);ap.add_argument('--epochs',type=int,default=65)
    ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    torch.set_num_threads(a.threads)
    dump(OUT/'protocol.json',{'args':vars(a),'script_sha256':sha(Path(__file__)),
         'scope':'frozen RD-AE adaptation with finite 16-QAM, AWGN; not official full RD-JSCC',
         'splits':'UE train first nue H; BS train next nfusion H; validation/test separate source files',
         'ns':64,'statistics':'per condition; repeated noises clustered by H; worst20 test labels diagnostic only',
         'hint_power':'constellation scaled with BS-fit H class frequencies per SNR; one symbol, same average energy as main P_ref',
         'ue_input':'full encoder af2 pre-FC feature and raw latent + known SNR,Ns; no current noise or actual decoder output'})
    for seed in a.seeds:run_seed(a,seed)

if __name__=='__main__':main()
