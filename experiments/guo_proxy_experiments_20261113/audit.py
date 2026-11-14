"""Verify frozen sources, score independence, distillation and resource accounting."""
import json
from pathlib import Path
import numpy as np
import torch
from run_proxy import OUT,PREV,data,build,make_proxy,ue_scores,sha

torch.set_num_threads(4);tr,_,te=data();checks=[]
for seed in [0,1,2]:
    p=OUT/f'results_s{seed}.json'
    if not p.exists():continue
    result=json.loads(p.read_text(encoding='utf8'))
    assert result['script_sha256']==sha(OUT/'run_proxy.py')
    assert result['source']['sha256']==sha(result['source']['checkpoint'])
    net,_=build(seed,tr);initial={k:v.clone() for k,v in net.state_dict().items()}
    b=torch.load(OUT/f'proxy_s{seed}.pt',weights_only=True)
    proxy=make_proxy(net.dec,b['rank']);proxy.load_state_dict(b['state_dict']);proxy.eval()
    assert b['meta']['val_mse']<=b['meta']['initial_val_mse']
    saved=torch.load(OUT/f'ue_scores_s{seed}.pt',weights_only=True)
    # Recompute prefix with EXACT independent local noise seeds. These do not
    # depend on any actual observation in old caches.
    recomputed=ue_scores(net,proxy,te[:16],8,1600000000+seed*10000000+200000,teacher_draws=0,bs=128)
    score_errors={}
    for s in [0.,10.,20.]:
        a=recomputed['proxy_mc'][recomputed['snr']==s]
        bval=saved['scores']['test']['proxy_mc'][saved['scores']['test']['snr']==s][:16]
        # random generator draws for first 16 rows agree despite original batch128
        err=float((a-bval).abs().max());score_errors[str(s)]=err
        torch.testing.assert_close(a,bval,rtol=1e-4,atol=1e-4)
    cache=torch.load(PREV/f'cache_s{seed}_test.pt',weights_only=True)
    max_repeat_diff=0.
    for s in [0.,10.,20.]:
        ix=cache['snr']==s;h=cache['h_index'][ix]
        v=saved['aligned']['test']['proxy_mc'][ix]
        for i in h.unique():max_repeat_diff=max(max_repeat_diff,float(v[h==i].max()-v[h==i].min()))
    assert max_repeat_diff==0
    for s,c in result['actions'].items():
        b=c['budgets'];assert b['fusion_symbols']<=b['bs_symbols']
        assert b['bs_symbols']-b['fusion_symbols']==b['fusion_unspent_symbols']
        n=b['n'];expected=65*n+64*b['fusion_retransmissions'];assert expected==b['fusion_symbols']
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in initial.items())
    single=OUT/f'single_pass_s{seed}.json'
    if single.exists():
        c=json.loads(single.read_text(encoding='utf8'));assert c['script_sha256']==sha(OUT/'run_single_pass.py')
    checks.append({'seed':seed,'source_sha_verified':True,'source_unchanged':True,
        'UE_MC_score_same_for_all_actual_noise_repeats':True,'independent_local_MC_recompute_max_abs':score_errors,
        'proxy_validation_KD_improved':True,'total_symbol_budgets_verified':True,
        'single_pass_finished':single.exists()})
result={'checks':checks,'previous_H_split_audit':json.loads((PREV/'audit.json').read_text(encoding='utf8'))['split_exact_content_overlap'],
        'limitation':'fitted proxy approximates teacher; not exact mapping identity; no official Guo reproduction'}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
