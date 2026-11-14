"""Independent audits of optional-refiner outcomes and frozen validation gates."""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_refinement import OUT,PREV,SNRS,Refiner,data,build,score_head,sha,dump,model_input,nmse_db_per,sgcs_per,bs_inputs,predict

torch.set_num_threads(2)
tr,va,te=data();checks=[]
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
assert protocol['script_sha256']==sha(OUT/'run_refinement.py')

with torch.inference_mode():
    for seed in (0,1,2):
        result=json.loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
        assert all(result['audit'].values())
        assert sha(Path(result['source']['checkpoint']))==result['source']['sha256']
        assert sha(Path(__file__).resolve().parents[2] / 'simulation/model_rdae.py')==result['source']['source_model_sha256']
        assert all(sha(Path(p))==v for p,v in result['source_hashes'].items())
        z=np.load(OUT/f'outcomes_s{seed}.npz')
        bundle=torch.load(OUT/f'refiner_s{seed}.pt',weights_only=True)
        model=Refiner(bundle['width']);model.load_state_dict(bundle['state_dict']);model.eval()
        assert result['training']==bundle['meta']
        trace=bundle['meta']['trace']
        selected=bundle['meta']['selected_epoch']
        assert selected>0
        assert trace[selected-1]['val_mean_nmse_db']==bundle['meta']['best_val_nmse_db']
        net,_=build(seed,tr);initial={k:v.clone() for k,v in net.state_dict().items()}
        rows=torch.load(PREV/f'cache_s{seed}_test.pt',weights_only=True)
        head,hb=score_head(seed)
        bs_pred=predict(head,hb,bs_inputs(rows)).numpy()
        assert np.max(np.abs(bs_pred-z['quality']))<1e-5
        maxerr=0.;features={}
        tap=net.dec.blocks[-1].af2.register_forward_hook(lambda m,a,o:features.update(map=o))
        for s in SNRS:
            ix=torch.where(rows['snr']==s)[0][:16]
            hh=te[rows['h_index'][ix]]
            base=net.dec(rows['y'][ix],torch.full((len(ix),1),10**(-s/20)))
            maps=torch.cat([base,features['map']],1)
            refined=base+model(model_input(maps,rows['snr'][ix],bundle['mean'],bundle['sd']))
            err=float((nmse_db_per(refined,hh)-torch.from_numpy(z['refined_loss'][ix])).abs().max())
            assert err<1e-4,err
            maxerr=max(maxerr,err)
            np.testing.assert_allclose(sgcs_per(refined,hh).numpy(),z['refined_sgcs'][ix],rtol=1e-5,atol=1e-5)
        tap.remove()
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in initial.items())
        for s in SNRS:
            ti=z['snr']==s
            vi=(z['val_snr']==s)&(z['val_h_index']>=250)
            gain=z['base_loss'][ti]-z['refined_loss'][ti]
            cell=result['conditions'][str(s)]
            assert abs(float(gain.mean())-cell['all_gain_db'])<1e-7
            for q in (.2,.5):
                g=cell['gates'][str(q)]
                cut=float(torch.quantile(torch.from_numpy(z['val_quality'][vi]),1-q))
                assert cut==g['threshold']
                decision=z['quality'][ti]>=cut
                assert np.array_equal(decision,z[f'gate_{s}_{q}'])
                assert decision.sum()==g['selected_observations']
                np.testing.assert_allclose((gain*decision).mean(),g['base_to_policy_gain_db'],atol=1e-7)
                ratio=result['cost']['monitor_conv_linear_MACs']/result['cost']['refiner_conv_linear_MACs']
                rq=min(1.,decision.mean()+ratio)
                np.testing.assert_allclose(rq,g['random_same_conv_linear_compute_call_fraction'],atol=1e-12)
                np.testing.assert_allclose(gain.mean()*rq,g['random_same_conv_linear_compute_gain_db'],atol=1e-7)
                count=int(decision.sum())
                oracle=np.sort(gain)[-count:].sum()/len(gain) if count else 0.
                assert abs(float(oracle)-g['hindsight_same_realization_oracle_gain_db'])<1e-7
        c=result['cost'];w=bundle['width']
        assert c['refiner_conv_linear_MACs']==1024*(20*w*9+w*2*9)
        assert c['refiner_parameters']==sum(p.numel() for p in model.parameters())==1594
        checks.append({'seed':seed,'source_hashes_unchanged':True,'base_decoder_frozen':True,
                       'validation_selection_recomputed':True,'gate_predictions_replayed':True,
                       'gate_and_policy_gains_recomputed':True,'same_compute_random_recomputed':True,
                       'refined_forward_loss_max_abs_error':maxerr,'parameter_MAC_counts_verified':True,
                       'no_true_H_in_gate_or_refiner_inputs':True})
dump(OUT/'audit.json',{'status':'PASS','checks':checks,'script_sha256':sha(Path(__file__)),
                     'scope':'feasibility only, one architecture/hyperparameter; no official diffusion or system throughput test'})
print(json.dumps({'status':'PASS','checks':checks},ensure_ascii=False,indent=2))
