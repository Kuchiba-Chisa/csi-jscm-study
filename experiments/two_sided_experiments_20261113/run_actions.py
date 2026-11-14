"""Expected repeat-transmission gains under matched mean symbol budget.

Uses the frozen quality experiment's caches; future outcomes only label/evaluate.
"""
from __future__ import annotations
import argparse,copy,math,time
from pathlib import Path
import numpy as np
import torch
from run_cooperative import OUT,build,data,dump,seed_all,fit_arrays,predict,bs_inputs,metrics,sha,constellation
from quality_v5 import sgcs_per,spearman
from common import nmse_db_per


@torch.no_grad()
def add_future(net,rows,truth,draws,base,bs=128):
    loss=torch.empty((len(rows['target']),draws));sgcs=torch.empty_like(loss)
    for snr in rows['snr'].unique().tolist():
        tick=time.perf_counter();inds=torch.where(rows['snr']==snr)[0]
        for ix in inds.split(bs):
            n=len(ix);sym=rows['symbols'][ix];y=rows['y'][ix]
            x=truth[rows['h_index'][ix]]
            cond=torch.full((n,1),10**(-snr/20))
            sd=math.sqrt(float(net.p_ref_of(64,snr)))*10**(-snr/20)/math.sqrt(2)
            for j in range(draws):
                gen=torch.Generator().manual_seed(base+int(snr)*1000000+int(ix[0])*100+j)
                second=sym+torch.randn(sym.shape,generator=gen)*sd
                reconstructed=net.dec(.5*(y+second),cond)
                loss[ix,j]=nmse_db_per(reconstructed,x)
                sgcs[ix,j]=sgcs_per(reconstructed,x)
        print(f'FUTURES snr={snr} rows={len(inds)} draws={draws} seconds={time.perf_counter()-tick:.1f}',flush=True)
    return loss,sgcs


def cluster_ci(h,values,seed=900):
    ids=h.unique();a=np.array([float(values[h==i].mean()) for i in ids])
    rng=np.random.default_rng(seed)
    boots=a[rng.integers(0,len(a),(1500,len(a)))].mean(1)
    return {'mean':float(a.mean()),'CI95':np.quantile(boots,[.025,.975]).tolist()}


def evaluate(rows,predictions,future,sgcs_future,quality_predictions,hint_energies,fraction=.2):
    out={}
    for snr in rows['snr'].unique().tolist():
        mask=rows['snr']==snr;current=rows['target'][mask]
        fs=future[mask];mid=fs.shape[1]//2
        g=current-fs[:,mid:].mean(1)
        gs=sgcs_future[mask,mid:].mean(1)-rows['sgcs'][mask]
        oracle=current-fs[:,:mid].mean(1)
        n=len(g);k_base=round(n*fraction);k_hint=math.floor((64*k_base-n)/64)
        assert k_hint>=0
        budgets={'bs_symbols':64*n+64*k_base,'fusion_symbols':65*n+64*k_hint,
                 'bs_retransmissions':k_base,'fusion_retransmissions':k_hint,
                 'fusion_unspent_symbols':64*k_base-n-64*k_hint,'n':n}
        strategies={};main_energy=rows['symbols'][mask].square().sum(1)
        selectors={}
        for name,score_all in predictions.items():
            k=k_base if name.startswith('bs_') else k_hint
            score=score_all[mask]
            order=torch.argsort(score,descending=True,stable=True)[:k]
            flag=torch.zeros(n);flag[order]=1;selectors[name]=flag
            strategies[name]={'mean_nmse_db':float(current.mean()-(flag*g).mean()),
                'nmse_improvement_db':float((flag*g).mean()),'mean_sgcs':float(rows['sgcs'][mask].mean()+(flag*gs).mean()),
                'gain_score_spearman':spearman(score.numpy(),g.numpy()),'retransmit_count':k,
                'negative_expected_gain_fraction_selected':float((g[order]<0).float().mean())}
            extra=0. if name.startswith('bs_') else float(hint_energies[4 if 'noisy4' in name else 2][mask].sum())
            strategies[name]['total_actual_tx_energy']=float(main_energy.sum()+(flag*main_energy).sum())+extra
        # Equal-action-count diagnostic, without charging the hint. Separates
        # better ranking from the lost retransmission opportunities.
        gross={}
        for name,score_all in predictions.items():
            score=score_all[mask];flag=torch.zeros(n)
            flag[torch.argsort(score,descending=True,stable=True)[:k_base]]=1
            gross[name]={'nmse_improvement_db':float((flag*g).mean()),
                'paired_minus_bs':cluster_ci(rows['h_index'][mask],(flag-selectors['bs_gain'])*g)}
        # Quality-based actions: currently worst predicted reconstruction first.
        for name,score_all in quality_predictions.items():
            k=k_base if name.startswith('bs_') else k_hint
            score=score_all[mask];flag=torch.zeros(n)
            flag[torch.argsort(score,descending=True,stable=True)[:k]]=1
            strategies['quality_'+name]={'nmse_improvement_db':float((flag*g).mean()),
                'mean_nmse_db':float(current.mean()-(flag*g).mean()),
                'mean_sgcs':float(rows['sgcs'][mask].mean()+(flag*gs).mean()),'retransmit_count':k}
        for kname,k in [('random_base',k_base),('random_pays_hint',k_hint)]:
            strategies[kname]={'nmse_improvement_db':float(g.mean()*k/n),
                'mean_nmse_db':float(current.mean()-g.mean()*k/n),
                'mean_sgcs':float(rows['sgcs'][mask].mean()+gs.mean()*k/n),'retransmit_count':k}
        for name,k in [('reference_base',k_base),('reference_pays_hint',k_hint)]:
            flag=torch.zeros(n);flag[torch.argsort(oracle,descending=True,stable=True)[:k]]=1
            strategies[name]={'nmse_improvement_db':float((flag*g).mean()),'retransmit_count':k,
                              'note':'privileged H and independent future draws; empirical reference, not mathematical bound'}
        pair={}
        for name,flag in selectors.items():
            if name=='bs_gain':continue
            delta=(flag-selectors['bs_gain'])*g
            pair[name+'_minus_bs_gain']=cluster_ci(rows['h_index'][mask],delta)
        out[str(int(snr))]={'budgets':budgets,'all_retransmit_gain_db':float(g.mean()),
                           'equal_retransmit_count_DIAGNOSTIC_hint_not_charged':gross,
                           'policies':strategies,'paired_net_gain_db':pair}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--train-future',type=int,default=8);p.add_argument('--test-future',type=int,default=32)
    p.add_argument('--epochs',type=int,default=65);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads);tr,va,te=data()
    protocol=__import__('json').loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for seed in a.seeds:
        began=time.perf_counter();net,source=build(seed,tr);original=copy.deepcopy(net.state_dict())
        rows={split:torch.load(OUT/f'cache_s{seed}_{split}.pt',weights_only=True) for split in ['train','val','test']}
        truth={'train':tr[protocol['args']['nue']:],'val':va,'test':te}
        for split,r in rows.items():
            fpath=OUT/f'future_s{seed}_{split}.pt'
            if fpath.exists():
                z=torch.load(fpath,weights_only=True)
                assert len(z['loss'])==len(r['target'])
            else:
                draws=a.test_future if split=='test' else a.train_future
                loss,sgcs=add_future(net,r,truth[split],draws,600000000+seed*10000000+{'train':1000000,'val':2000000,'test':3000000}[split])
                z={'loss':loss,'sgcs':sgcs};torch.save(z,fpath)
            r['future']=z['loss'];r['future_sgcs']=z['sgcs']
        target={k:r['target']-r['future'].mean(1) for k,r in rows.items()}
        predictions={};quality_predictions={};info={}
        modes=[('bs_gain',None),('fusion_float_gain','ue_score'),('fusion_clean2_gain','hint_clean2'),
               ('fusion_noisy2_gain','hint_noisy2'),('fusion_noisy4_gain','hint_noisy4'),('ue_noisy2_gain','ue_only2')]
        for name,hint in modes:
            inputs={}
            for split,r in rows.items():
                if hint=='ue_only2':X=torch.cat([r['condition'],r['hint_noisy2'][:,None]],1)
                else:
                    X=bs_inputs(r)
                    if hint:X=torch.cat([X,r[hint][:,None]],1)
                inputs[split]=X
            head,bundle,meta=fit_arrays(inputs['train'],target['train'],inputs['val'],target['val'],930000+seed,25000,a.epochs)
            predictions[name]=predict(head,bundle,inputs['test'])
            info[name]=meta;torch.save(bundle,OUT/f'action_{name}_s{seed}.pt')
            print(f'ACTION HEAD seed={seed} name={name} val_mse={meta["val_mse_db2"]:.5f}',flush=True)
        for mode,hint in [('bs_only',None),('fusion_noisy2','hint_noisy2'),('fusion_noisy4','hint_noisy4')]:
            bundle=torch.load(OUT/f'quality_{mode}_s{seed}.pt',weights_only=True)
            from q2_quality_networks import Monitor
            model=Monitor(bundle['input_dim'],bundle['budget']);model.load_state_dict(bundle['state_dict']);model.eval()
            X=bs_inputs(rows['test'])
            if hint:X=torch.cat([X,rows['test'][hint][:,None]],1)
            quality_predictions[mode]=predict(model,bundle,X)
        quality_result=__import__('json').loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
        hint_energies={}
        for bits in (2,4):
            energy=torch.empty(len(rows['test']['target']));pts=constellation(bits)
            for snr in rows['test']['snr'].unique().tolist():
                ix=rows['test']['snr']==snr;entry=quality_result['hint'][str(bits)]
                cuts=torch.tensor(entry['table'][str(snr)]['cuts'])
                classes=torch.bucketize(rows['test']['ue_score'][ix].contiguous(),cuts)
                scale=entry['scales'][str(snr)]
                energy[ix]=pts[classes].square().sum(1)*(scale**2)*float(net.p_ref_of(64,snr))
            hint_energies[bits]=energy
        result={'seed':seed,'source':source,'args':vars(a),'script_sha256':sha(Path(__file__)),
                'heads':info,'test':evaluate(rows['test'],predictions,rows['test']['future'],rows['test']['future_sgcs'],quality_predictions,hint_energies),
                'scope':'same QAM symbol repetition, average observations, original SNR decoder input; no aging or full control protocol',
                'resource':'one hint per sample charged even to those not retransmitted; fusion_float is information diagnostic charged one symbol, not realizable signaling',
                'duration_s':time.perf_counter()-began}
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in original.items())
        np.savez_compressed(OUT/f'action_predictions_s{seed}.npz',**{k:v.numpy() for k,v in predictions.items()})
        dump(OUT/f'action_results_s{seed}.json',result)
        for snr,c in result['test'].items():
            print('ACTION RESULT seed='+str(seed)+' snr='+snr+' '+str({name:round(x['nmse_improvement_db'],5) for name,x in c['policies'].items()}),flush=True)
        print(f'ACTION SEED DONE {seed} seconds={result["duration_s"]:.1f}',flush=True)

if __name__=='__main__':main()
