import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
O=Path(__file__).resolve().parent;P=O.parent/'two_sided_experiments_20261113';G=O.parent/'guo_proxy_experiments_20261113'
sys.path.insert(0,str(G))
from run_proxy import apply_cal,fit_arrays,predict,sha
from q2_quality_networks import Monitor
from run_ue_policy import by_H,fixed_threshold
torch.set_num_threads(4);checks=[]
for seed in [0,1,2]:
    result=json.loads((O/f'results_s{seed}.json').read_text(encoding='utf8'))
    assert result['script_sha256']==sha(O/'run_ue_policy.py')
    assert result['frozen_proxy_SHA256']==sha(G/f'proxy_s{seed}.pt')
    assert result['source_single_pass_SHA256']==sha(G/f'single_pass_s{seed}.json')
    a=np.load(O/f'decisions_s{seed}.npz')
    cache=torch.load(P/f'cache_s{seed}_test.pt',weights_only=True)
    np.testing.assert_array_equal(a['current_loss'],cache['target'].numpy())
    np.testing.assert_array_equal(a['h_index'],cache['h_index'].numpy())
    single=json.loads((G/f'single_pass_s{seed}.json').read_text(encoding='utf8'))
    calibration={float(k):v for k,v in single['calibration'].items()}
    q=torch.load(G/f'ue_scores_s{seed}.pt',weights_only=True)
    value=apply_cal(cache,q['aligned']['test']['proxy_clean'],calibration)
    torch.testing.assert_close(value,torch.from_numpy(a['score_ue_quality']),rtol=0,atol=0)
    bundle=torch.load(O/f'ue_gain_head_s{seed}.pt',weights_only=True)
    model=Monitor(bundle['input_dim'],bundle['budget']);model.load_state_dict(bundle['state_dict']);model.eval()
    X=torch.cat([cache['condition'],value[:,None]],1)
    prediction=predict(model,bundle,X).numpy()
    np.testing.assert_allclose(prediction,a['score_ue_gain'],rtol=0,atol=0)
    # Input to UE head contains no actual y/decoder feature/current loss.
    assert X.shape[1]==3
    for s in [0,10,20]:
        ix=a['snr']==s;h=a['h_index'][ix]
        for i in np.unique(h):
            for name in ['ue_quality','ue_gain','ue_quality_best','threshold_ue_gain','threshold_ue_quality']:
                f=a['flag_'+name][ix][h==i];assert np.all(f==f[0])
            for name in ['ue_quality','ue_gain']:
                v=a['score_'+name][ix][h==i];assert np.all(v==v[0])
        for name in ['ue_quality','ue_gain','ue_quality_best','old_ue_quality','bs_gain','bs_quality']:
            assert a['flag_'+name][ix].sum()==360
        assert a['flag_threshold_ue_gain'][ix].sum()==a['flag_bs_matched_threshold_ue_gain'][ix].sum()
        fs=a['future_loss'][ix];gain=a['current_loss'][ix]-fs[:,fs.shape[1]//2:].mean(1)
        for name in ['ue_quality','ue_gain','bs_gain','ue_quality_best']:
            val=float((a['flag_'+name][ix]*gain).mean())
            assert abs(val-result['test'][str(s)]['policies'][name]['nmse_improvement_db'])<2e-7
        for name in ['ue_quality','ue_gain']:
            flag=a['flag_'+name][ix]
            uses=flag.astype(bool)
            score=a['score_'+name][ix]
            # UE chosen set agrees with its own score order, no outcome sorting.
            assert score[uses].min()>=score[~uses].max()-1e-7
    checks.append({'seed':seed,'source_hashes_verified':True,'head_predictions_reproduced':True,
        'UE_inputs_only_local_score_and_condition':True,'UE_decisions_invariant_to_current_noise':True,
        'all_fixed_quota_strategies_select_360_of_1800':True,
        'threshold_BS_baseline_matches_actual_UE_usage':True,'policy_gains_independently_recomputed':True})
out={'checks':checks,'source_split_audit':json.loads((P/'audit.json').read_text(encoding='utf8'))['split_exact_content_overlap'],
     'limitations':'top20% offline pool, control overhead omitted, validation-fixed threshold usage is not exactly20%'}
(O/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
