"""Independent replay of thresholds, confusion counts, costs and source hashes."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
from evaluate import METHODS,SNRS,TARGETS,BS,UE,sha,load_head,make_proxy,build,data,nmse_db_per,dump

torch.set_num_threads(1)
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
assert protocol['script_sha256']==sha(OUT/'evaluate.py')
assert protocol['training'] is False
checks=[]
tr,va,te=data()
with torch.inference_mode():
    for seed in protocol['args']['seeds']:
        r=json.loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
        assert all(r['audit'].values())
        assert all(sha(Path(p))==h for p,h in r['source_hashes'].items())
        assert sha(Path(r['source']['checkpoint']))==r['source']['sha256']
        z=np.load(OUT/f'predictions_s{seed}.npz')
        expected_conditions=[]
        for s in SNRS:
            a=(z['val_snr']==s)&(z['val_h_index']<250)
            b=(z['val_snr']==s)&(z['val_h_index']>=250)
            t=z['test_snr']==s
            assert not set(z['val_h_index'][a])&set(z['val_h_index'][b])
            cut=float(np.quantile(z['val_target'][a],.8))
            assert cut==r['conditions'][str(s)]['bad_nmse_db_threshold']
            vb=z['val_target'][b]>cut;tb=z['test_target'][t]>cut
            for target in TARGETS:
                for name in METHODS:
                    rec=r['conditions'][str(s)]['targets'][str(target)]['methods'][name]
                    pos=np.sort(z[f'val_{name}'][b][vb])
                    threshold=float(pos[len(pos)-math.ceil(target*len(pos)-1e-10)])
                    assert threshold==rec['alarm_threshold']
                    alarm=z[f'test_{name}'][t]>=threshold
                    assert np.array_equal(alarm,z[f'test_alarm_{s}_{target}_{name}'])
                    counts=[int(np.sum(tb&alarm)),int(np.sum(~tb&alarm)),
                            int(np.sum(tb&~alarm)),int(np.sum(~tb&~alarm))]
                    assert counts==rec['test']['counts_tp_fp_fn_tn']
                    tp,fp,fn,tn=counts
                    assert abs(tp/(tp+fn)-rec['test']['tpr'])<1e-12
                    assert abs(fp/(fp+tn)-rec['test']['fpr'])<1e-12
                    # UE's pre-transmission quality cannot depend on the
                    # actual realization in its three cached repetitions.
                    if name=='ue_proxy':
                        for h in np.unique(z['test_h_index'][t]):
                            group=z[f'test_{name}'][t][z['test_h_index'][t]==h]
                            assert len(group)==3 and group.max()==group.min()
            expected_conditions.append(s)
        net,_=build(seed,tr)
        proxy_bundle=torch.load(UE/f'proxy_s{seed}.pt',weights_only=True)
        proxy=make_proxy(net.dec,proxy_bundle['rank'])
        proxy.load_state_dict(proxy_bundle['state_dict']);proxy.eval()
        cache=torch.load(BS/f'cache_s{seed}_test.pt',weights_only=True)
        max_label_error=max_feature_error=0.
        features={}
        handle=net.dec.blocks[-1].af2.register_forward_hook(lambda m,a,o:features.update(value=o))
        for s in SNRS:
            ix=torch.where(cache['snr']==s)[0][:16]
            output=net.dec(cache['y'][ix],torch.full((len(ix),1),10**(-s/20)))
            true=te[cache['h_index'][ix]]
            label=nmse_db_per(output,true)
            err=float((label-cache['target'][ix]).abs().max())
            assert err<1e-4
            max_label_error=max(max_label_error,err)
            pooled=torch.nn.functional.adaptive_avg_pool2d(features['value'],(2,2)).flatten(1)
            ferr=float((pooled-cache['feature'][ix]).abs().max())
            assert ferr<1e-4
            max_feature_error=max(max_feature_error,ferr)
        handle.remove()
        for name,old in [('bs_small','bs_only'),('bs_large','bs_large')]:
            head,bundle=load_head(BS/f'quality_{old}_s{seed}.pt')
            linear=sum(m.weight.numel() for m in head.modules() if isinstance(m,torch.nn.Linear))
            assert linear==r['cost'][name]['conv_linear_macs']
            assert sum(p.numel() for p in head.parameters())==r['cost'][name]['parameters']
        expected_proxy_macs=(128*48+48*2048 # low-rank FC
                            +2*32*32*2*7*7 # initial convolution
                            +8*32*32*2*7*7+16*32*32*8*5*5+2*32*32*16*3*3
                            +sum(m.weight.numel() for name,m in proxy.named_modules()
                                 if isinstance(m,torch.nn.Linear) and not name.startswith('fc.')))
        assert expected_proxy_macs==r['cost']['ue_proxy']['conv_linear_macs']
        assert sum(p.numel() for p in proxy.parameters())==r['cost']['ue_proxy']['parameters']
        checks.append({'seed':seed,'thresholds_independently_recomputed':True,
                       'confusion_counts_and_rates_recomputed':True,'validation_H_groups_disjoint':True,
                       'UE_scores_identical_across_actual_noise_repeats':True,
                       'saved_sources_unchanged':True,'analytic_MACs_and_parameters_verified':True,
                       'label_forward_max_abs_error':max_label_error,
                       'feature_forward_max_abs_error':max_feature_error})

prior=json.loads((BS/'audit.json').read_text(encoding='utf8'))['split_exact_content_overlap']
assert not any(prior.values())
result={'status':'PASS','checks':checks,'prior_source_H_overlap':prior,
        'audit_script_sha256':sha(Path(__file__)),
        'limits':'same previously used validation set; thresholds fixed but test recall may drift; no mobile/BS hardware or power/throughput measurements'}
dump(OUT/'audit.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
