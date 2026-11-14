"""Verify selection and all policy metrics; replay representative unseen conditions."""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import torch
import run_uncertainty as r

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
ap.add_argument('--out',default='audit.json')
args=ap.parse_args()
p=json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
assert r.sha(ROOT/'run_uncertainty.py')==p['script_sha256']
tr,va,te=r.data()
torch.set_num_threads(4)
checks=[]
for seed in args.seeds:
    assert seed in p['args']['seeds']
    result=json.loads((ROOT/f'results_s{seed}.json').read_text(encoding='utf8'))
    lock=json.loads((ROOT/f'locked_s{seed}.json').read_text(encoding='utf8'))
    assert all(result[k]==v for k,v in lock.items())
    for path,h in lock['input_hashes'].items():
        src=Path(path)
        if not src.exists() and 'work' in src.parts:
            i=src.parts.index('work')
            rest=Path(*src.parts[i+1:])
            if rest.parts[0]=='q1_snr_uncertainty_20261114':src=ROOT/Path(*rest.parts[1:])
            else:src=ROOT/'dependency_snapshot'/rest
        assert r.sha(src)==h
    for ns in r.LENGTHS:
        for mode in r.MODES:
            cells=[c for c in lock['validation_trace'] if c['ns']==ns and c['mode']==mode]
            worst={a:max(c['candidate_mean_nmse_db'][str(a)]-c['candidate_mean_nmse_db']['0.0'] for c in cells) for a in r.ALPHAS}
            best=min(worst.values());alpha=max(a for a in r.ALPHAS if worst[a]<=best+1e-7)
            assert alpha==result['selection'][f'{ns}_{mode}']['alpha']
    z=np.load(ROOT/f'outcomes_s{seed}.npz')
    for c in result['cells']:
        prefix=f"ns{c['ns']}_s{c['true_snr']}"
        h=z[prefix+'_h']
        assert c['estimated_snr']==np.clip(c['true_snr']+c['error'],0,20)
        base=z[f"{prefix}_{c['mode']}_e{c['error']}_S0_loss"]
        for method in ('S0','vanilla','conservative'):
            key=f"{prefix}_{c['mode']}_e{c['error']}_{method}"
            loss=z[key+'_loss'];sgcs=z[key+'_sgcs']
            saved=c['methods'][method]
            assert abs(float(loss.mean())-saved['nmse_db'])<1e-7
            assert abs(float(sgcs.mean())-saved['sgcs'])<1e-7
            assert abs(float((base-loss).mean())-saved['gain_vs_input_matched_S0_db'])<1e-7
        if c['alpha']==0:
            np.testing.assert_array_equal(base,z[f"{prefix}_{c['mode']}_e{c['error']}_conservative_loss"])
        if c['error']==0:
            np.testing.assert_array_equal(z[f"{prefix}_{c['mode']}_e0_vanilla_loss"],z[prefix+'_accurate_loss'])
        if c['mode']=='bn_only':
            np.testing.assert_array_equal(base,z[f'{prefix}_bn_only_e0_S0_loss'])
    net,source=r.build(seed,tr);net.requires_grad_(False)
    assert source==result['source']
    initial=copy.deepcopy(net.state_dict());base=r.stats_from_net(net)
    max_error=0.;replayed=0
    for ns in r.LENGTHS:
        bank=r.anchors(seed,ns)
        for snr in (5,15):
            # Full observe reproduces training-noise grouping; first128 is first physical batch.
            obs=r.observe(net,te[600:1000],snr,ns,seed,'test')
            small={k:v[:128] for k,v in obs.items()}
            for mode in r.MODES:
                a=result['selection'][f'{ns}_{mode}']['alpha']
                for error in (-3,3):
                    hat=float(np.clip(snr+error,0,20))
                    decoder_snr=snr if mode=='bn_only' else hat
                    target=r.candidate(*bank,hat)
                    for method,alpha in [('S0',0.),('vanilla',1.),('conservative',a)]:
                        out=r.decode(net,small,te[600:1000],decoder_snr,r.blend(base,target,alpha))
                        key=f'ns{ns}_s{snr}_{mode}_e{error}_{method}'
                        error_value=float(np.max(np.abs(out['loss']-z[key+'_loss'][:128])))
                        assert error_value<1e-5,(seed,key,error_value)
                        max_error=max(max_error,error_value);replayed+=1
    changed=[k for k,v in net.state_dict().items() if not torch.equal(v,initial[k])]
    assert all(k in base for k in changed)
    assert all(p.grad is None for p in net.parameters())
    # Independent moment-mixture formula check, including exact endpoints.
    a,b=r.anchors(seed,16);target=r.candidate(a,b,7.)
    blend=r.blend(base,target,.5)
    for k in base:
        if not k.endswith('running_var'):continue
        mk=k.replace('running_var','running_mean')
        second=.5*(base[k]+base[mk]**2)+.5*(target[k]+target[mk]**2)
        independent=second-blend[mk]**2
        torch.testing.assert_close(blend[k],independent,atol=1e-6,rtol=1e-5)
    checks.append(dict(seed=seed,validation_minimax_selection_recomputed=True,
        all_saved_mean_metrics_recomputed=True,replayed_forward_cases=replayed,
        max_forward_replay_error=max_error,only_decoder_BN_moments_changed=True,
        independent_second_moment_formula=True))
    print('AUDIT PASS',seed,'cases',replayed,'max error',max_error,flush=True)
r.dump(ROOT/args.out,dict(status='PASS',checks=checks,
    scope='All selection/mean metrics and source identities; representative first128-H forward replay for all lengths, both modes, unseen5/15dB and extreme signed biases. Does not certify external validity.'))
