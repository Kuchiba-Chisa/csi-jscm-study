"""Independent full forward replay plus direct NumPy gate/budget calculations."""
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch
import run_joint as r

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
assert r.rf.sha(OUT/'run_joint.py') == protocol['script_sha256']
tr, va, te = r.rf.data()
torch.set_num_threads(4)
records = []
for seed in protocol['args']['seeds']:
    result = json.loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
    z = np.load(OUT/f'outcomes_s{seed}.npz')
    locked = json.loads((OUT/f'locked_s{seed}.json').read_text(encoding='utf8'))
    for k,v in locked.items():
        assert result[k] == v
    for p,h in result['input_hashes'].items():
        source=Path(p)
        if not source.exists():
            prior=source.parts
            if 'work' in prior:
                source=OUT/'dependency_snapshot'/Path(*prior[prior.index('work')+1:])
        assert r.rf.sha(source) == h
    net, meta = r.rf.build(seed,tr)
    net.requires_grad_(False)
    assert meta == result['source']
    initial = copy.deepcopy(net.state_dict())
    checkpoint = torch.load(meta['checkpoint'], weights_only=True)
    assert all(torch.equal(v, net.state_dict()[k]) for k,v in checkpoint.items())
    stored = torch.load(OUT/f'test_observations_s{seed}.pt',weights_only=True)['observations']
    regenerated = r.observations(net,te[600:1000],seed,'test')
    assert all(torch.equal(stored[k],regenerated[k]) for k in stored)
    s0 = r.decoded(net,stored,te[600:1000],seed,False)
    adapted = r.decoded(net,stored,te[600:1000],seed,True)
    changed = [k for k,v in net.state_dict().items() if not torch.equal(v,initial[k])]
    assert all(k.startswith('dec.') and k.endswith(('running_mean','running_var')) for k in changed)
    head,hb=r.rf.score_head(seed)
    nb=torch.load(OUT/f'updated_monitor_s{seed}.pt',weights_only=True)['bundle']
    newhead=copy.deepcopy(head);newhead.load_state_dict(nb['state_dict']);newhead.eval()
    oldref,rb=r.old_refiner(seed)
    ur=torch.load(OUT/f'updated_refiner_s{seed}.pt',weights_only=True)
    newref=copy.deepcopy(oldref);newref.load_state_dict(ur['state_dict']);newref.eval()
    assert sum(p.numel() for p in oldref.parameters())==protocol['cost']['parameters']
    monitor_macs=sum(m.weight.numel() for m in head.modules() if isinstance(m,torch.nn.Linear))
    assert monitor_macs==protocol['cost']['monitor_MACs']
    # Rebuild TRAIN-ONLY moments, which are the 36-scalar no-weight-update variant.
    net.load_state_dict(initial)
    trobs=r.observations(net,tr[:protocol['args']['ntrain']],seed,'train')
    train=r.decoded(net,trobs,tr[:protocol['args']['ntrain']],seed,True)
    moment=copy.deepcopy(rb['norm']); norm=r.rf.normalizers(train)
    moment['mean'],moment['sd']=norm['mean'],norm['sd']
    torch.save(dict(moment_norm=moment),OUT/f'replayed_moment_norm_s{seed}.pt')
    del train
    q0=r.rf.predict(head,hb,r.rf.head_inputs(s0))
    qa=r.rf.predict(head,hb,r.rf.head_inputs(adapted))
    qnew=r.rf.predict(newhead,nb,r.rf.head_inputs(adapted))
    qaff=r.affine(qa,adapted,{int(s):v for s,v in result['affine_params'].items()})
    outputs={
        'S0_refined':r.rf.evaluate(oldref,s0,rb['norm']),
        'A_old_refined':r.rf.evaluate(oldref,adapted,rb['norm']),
        'A_moment_refined':r.rf.evaluate(oldref,adapted,moment),
        'A_updated_refined':r.rf.evaluate(newref,adapted,ur['norm'])}
    replay=dict(S0_base=s0['base_loss'],A_base=adapted['base_loss'],
                q_S0=q0,q_A_old=qa,q_A_affine=qaff,q_A_updated=qnew)
    replay.update({k:v['loss'] for k,v in outputs.items()})
    max_error=0.
    for name,value in replay.items():
        error=float(np.abs(value.numpy()-z[name]).max())
        assert error<1e-5,(name,error)
        max_error=max(max_error,error)
    arms={
        'S0_old_monitor_old_refiner':('S0_base','S0_refined','q_S0','old_gate_thresholds'),
        'A_old_modules_old_gate':('A_base','A_old_refined','q_A_old','old_gate_thresholds'),
        'A_rethreshold_old_modules':('A_base','A_old_refined','q_A_old','recalibrated_old_gate_thresholds'),
        'A_affine_old_modules':('A_base','A_old_refined','q_A_affine','affine_gate_thresholds'),
        'A_updated_monitor_old_refiner':('A_base','A_old_refined','q_A_updated','updated_gate_thresholds'),
        'A_updated_monitor_moment_refiner':('A_base','A_moment_refined','q_A_updated','updated_gate_thresholds'),
        'A_updated_monitor_updated_refiner':('A_base','A_updated_refined','q_A_updated','updated_gate_thresholds')}
    for name,(basekey,refkey,qkey,cutkey) in arms.items():
        for s in r.SNRS:
            key=str(s);ix=z['snr']==s
            call=z[qkey][ix]>=result[cutkey][key]
            np.testing.assert_array_equal(call,z[name+'_call_'+key])
            base,fin=z[basekey][ix],z[refkey][ix]
            output=np.where(call,fin,base)
            bad=result['bad_thresholds'][key]
            random_rate=min(1.,call.mean()+protocol['cost']['monitor_MACs']/protocol['cost']['refiner_MACs'])
            randombad=(1-random_rate)*(base>bad)+random_rate*(fin>bad)
            cell=result['policies'][name][key]
            for actual,recorded in [(call.mean(),cell['refiner_call_rate']),
                  ((output>bad).mean(),cell['final_bad_rate']),
                  (output.mean(),cell['final_nmse_db']),
                  (randombad.mean(),cell['random_same_MAC_bad_rate'])]:
                assert abs(actual-recorded)<1e-6
            assert abs(cell['additional_MACs']-random_rate*protocol['cost']['refiner_MACs'])<1e-6
    assert all(p.grad is None for p in net.parameters())
    records.append(dict(seed=seed,max_full_forward_replay_error=max_error,
        regenerated_received_y_exact=True,only_decoder_BN_buffers_change=True,
        train_only_moments_rebuilt=True,all_policy_gates_and_same_MAC_budgets_recomputed=True))
    print('AUDIT PASS',seed,max_error,flush=True)
r.rf.dump(OUT/'audit.json',dict(status='PASS',records=records,
    scope='Full held-out-observation forward replay, source hashes, train-only moments, model-update boundaries and independent NumPy policy budgets; not external-validity or historical-data-use certification.'))
