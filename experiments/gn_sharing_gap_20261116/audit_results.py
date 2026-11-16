"""Independently verify saved metrics, state provenance and matched controls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
z=np.load(ROOT/'observations.npz')
checks=[]
inits=[]
for name in p['models']:
    ck=torch.load(ROOT/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
    assert len(ck['trace'])==p['args']['epochs']==40
    assert ck['trace'][-1]['epoch']==40
    assert ck['config']==p['args']
    assert not any('running_mean' in k or 'running_var' in k for k in ck['state_dict'])
    assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
    inits.append(ck['initial_state_sha256'])
    count=sum(v.numel() for v in ck['state_dict'].values())
    assert count==553982,count
    for s in p['SNR_grid']:
        row=r['results'][name][str(s)]
        key=f'{name}__snr{s}__'
        ratio=z[key+'ratio'];db=z[key+'db'];sc=z[key+'sgcs']
        assert ratio.shape==db.shape==sc.shape==(3,1000)
        assert np.isfinite(db).all() and np.isfinite(sc).all()
        assert np.allclose(db,10*np.log10(ratio+1e-12))
        assert abs(float(db.mean())-row['mean_nmse_db'])<1e-7
        assert abs(float(sc.mean())-row['sgcs'])<1e-7
        assert abs(float(10*np.log10(ratio.mean()))-row['db_mean_linear_nmse'])<1e-7
        checks.append({'model':name,'snr':s,'arrays_and_metrics':'pass'})
assert len(set(inits))==1 and inits[0]==p['initial_state_sha256']
for s in p['SNR_grid']:
    a=r['results']['shared'][str(s)];b=r['results'][f'specialist_{s}'][str(s)]
    assert a['standard_noise_sha256']==b['standard_noise_sha256']
    d=r['gaps'][str(s)]
    assert abs((a['mean_nmse_db']-b['mean_nmse_db'])-d['nmse_shared_minus_specialist']['mean'])<1e-6
    assert abs((b['sgcs']-a['sgcs'])-d['sgcs_specialist_minus_shared']['mean'])<1e-7
for file,expected in p['source_sha256'].items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
assert hashlib.sha256((ROOT/'run_gn_gap.py').read_bytes()).hexdigest()==p['script_sha256']
out={'status':'pass','checks':checks,'training_seeds':1,'test_H':1000,'noise_draws_per_H':3,
     'all_four_models_same_initialization':True,'all_models_final_40_epochs':True,
     'no_BN_running_statistics':True,'source_and_training_script_unchanged':True,
     'parameters_per_model':553982,'FP32_weight_bytes_per_model':553982*4,
     'limits':'H-paired intervals do not measure retraining uncertainty'}
(ROOT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
