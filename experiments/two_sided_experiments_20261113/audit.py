"""Numerical link parity, information/split, normalization and frozen-source audit."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from run_cooperative import OUT,build,data,cache,seed_all,sha,constellation
from common import nmse_db_per

torch.set_num_threads(2)
tr,va,te=data()
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
sets={}
for label,x in [('ue_train',tr[:3000]),('bs_train',tr[3000:5000]),('val',va[:500]),('test',te[:600])]:
    sets[label]={hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in x}
overlap={a+'_'+b:len(sets[a]&sets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]}
assert not any(overlap.values()),overlap
checks=[]
for seed in (0,1,2):
    result=json.loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
    assert result['source']['sha256']==sha(result['source']['checkpoint'])
    assert result['source']['source_model_sha256']==sha(Path(__file__).resolve().parents[2] / 'simulation/model_rdae.py')
    r=torch.load(OUT/f'cache_s{seed}_test.pt',weights_only=True)
    assert r['target'].numel()==600*3*3
    assert all(int((r['snr']==s).sum())==1800 for s in [0,10,20])
    net,_=build(seed,tr);state={k:v.clone() for k,v in net.state_dict().items()}
    captured={}
    hook=net.dec.register_forward_pre_hook(lambda m,args:captured.update(y=args[0].detach().clone()))
    seed_all(410000000)
    with torch.no_grad():rec=net(te[:128],0.,64)
    hook.remove()
    torch.testing.assert_close(captured['y'],r['y'][:128],rtol=0,atol=0)
    torch.testing.assert_close(nmse_db_per(rec,te[:128]),r['target'][:128],rtol=1e-5,atol=1e-5)
    # UE inputs must be invariant to the sampled realization, unlike y.
    first=cache(net,te[:16],(0,),1,910000000)
    second=cache(net,te[:16],(0,),1,920000000)
    torch.testing.assert_close(first['ue'],second['ue'],rtol=0,atol=0)
    assert not torch.equal(first['y'],second['y'])
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
    checks.append({'seed':seed,'cached_link_y_max_abs_error':float((captured['y']-r['y'][:128]).abs().max()),
                   'cached_loss_max_abs_error':float((nmse_db_per(rec,te[:128])-r['target'][:128]).abs().max()),
                   'UE_input_noise_invariant':True,'backbone_unchanged':True,'source_hash_verified':True})
dump={'split_exact_content_overlap':overlap,'checks':checks,
      'protocol_script_sha_matches':protocol['script_sha256']==sha(OUT/'run_cooperative.py'),
      'constellation_uniform_mean_energy':{str(b):float(constellation(b).square().sum(1).mean()) for b in (2,4)}}
assert dump['protocol_script_sha_matches']
(OUT/'audit.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dump,ensure_ascii=False,indent=2))
