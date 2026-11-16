"""Independent replay of the supplied output-calibration pilot; no model training."""
from pathlib import Path
import sys, os, json, hashlib, time
import numpy as np
import torch

project = Path(__file__).resolve().parents[2] / 'simulation'
sys.path.insert(0,str(project))
os.chdir(project)
from model_jscm import Link, calibrate
from quality_v5 import load_split, forward_op, sigma_of, decode, nmse_db_per, sgcs_per

torch.set_num_threads(4)
outdir=Path(__file__).resolve().parent
original=json.loads((project/'results_affine.json').read_text(encoding='utf8'))
train,val,test=[load_split(n) for n in ['train','val','test']]
def fit(p,t):
    p,t=p.double().flatten(),t.double().flatten()
    pm,tm=p.mean(),t.mean()
    a=((p-pm)*(t-tm)).sum()/((p-pm)**2).sum()
    b=tm-a*pm
    a0=(p*t).sum()/(p*p).sum()
    return float(a),float(b),float(a0)
def metrics(p,t):
    lin=((p-t)**2).sum((1,2,3))/(t*t).sum((1,2,3))
    return {'mean_db':float(nmse_db_per(p,t).mean()),
            'db_mean_linear':float(10*torch.log10(lin.mean())),
            'sgcs2':float(sgcs_per(p,t).mean())}
def load(name,seed):
    net=Link('learned_soft',64,16,backbone='ours')
    ckpt=project/'ckpt_v2'/f'{name}_s{seed}.pt'
    net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True))
    net.eval();calibrate(net,train)
    return net,hashlib.sha256(ckpt.read_bytes()).hexdigest()
def obs(net,tx,snr,nseed):
    g=torch.Generator().manual_seed(nseed)
    return decode(net,tx+sigma_of(net,snr)*torch.randn(tx.shape,generator=g),snr)

start=time.time(); records={}; maxerr=0.; checkpoints={}
with torch.no_grad():
    for seed in range(3):
        net,checkpoints[f'shared_s{seed}']=load('learned_soft',seed)
        xv,xt=forward_op(net,val),forward_op(net,test)
        rows=[]
        for i,snr in enumerate(range(0,21,2)):
            pv=obs(net,xv,snr,1234+snr)
            pt=obs(net,xt,snr,1241+snr)
            a,b,a0=fit(pv,val)
            b0=metrics(pt,test);ab=metrics(a*pt+b,test);aonly=metrics(a0*pt,test)
            saved=original[f's{seed}'][i]
            maxerr=max(maxerr,abs(saved['base']-b0['mean_db']),abs(saved['affine']-ab['mean_db']))
            rows.append({'snr':snr,'a':a,'b':b,'scale_only_a':a0,'base':b0,
                         'affine':ab,'scale_only':aonly,
                         'scale_sgcs_max_abs':float((sgcs_per(a0*pt,test)-sgcs_per(pt,test)).abs().max()),
                         'source_delta':saved['delta']})
        records[f's{seed}']={'shared':rows}
        teacher,checkpoints[f'specialist0_s{seed}']=load('learned_soft_f0',seed)
        pv=obs(teacher,forward_op(teacher,val),0,1234)
        pt=obs(teacher,forward_op(teacher,test),0,1241)
        a,b,a0=fit(pv,val)
        records[f's{seed}']['specialist0']={'a':a,'b':b,'base':metrics(pt,test),
            'affine':metrics(a*pt+b,test),'scale_only':metrics(a0*pt,test)}
        print(f'seed {seed} replay complete',flush=True)

summary=[]
for i,snr in enumerate(range(0,21,2)):
    rows=[records[f's{s}']['shared'][i] for s in range(3)]
    deltas=[r['base']['mean_db']-r['affine']['mean_db'] for r in rows]
    scales=[r['base']['mean_db']-r['scale_only']['mean_db'] for r in rows]
    summary.append({'snr':snr,'base':float(np.mean([r['base']['mean_db'] for r in rows])),
      'affine':float(np.mean([r['affine']['mean_db'] for r in rows])),
      'gain_mean':float(np.mean(deltas)),'gain_seed_sd':float(np.std(deltas,ddof=1)),
      'scale_only_gain':float(np.mean(scales)),
      'a_mean':float(np.mean([r['a'] for r in rows])),
      'affine_sgcs_gain':float(np.mean([r['affine']['sgcs2']-r['base']['sgcs2'] for r in rows])),
      'scale_sgcs_max_abs':max(r['scale_sgcs_max_abs'] for r in rows)})
gap={}
for name in ['base','affine','scale_only']:
    gap[name]=float(np.mean([records[f's{s}']['shared'][0][name]['mean_db']-
                           records[f's{s}']['specialist0'][name]['mean_db'] for s in range(3)]))
report={'source':'results_affine.json and affine_calib.py','source_model':'ours CNN, learned 16-PAM, 64 real symbols',
 'val_H':len(val),'test_H':len(test),'noise_repetitions_per_H':1,'train_seeds':3,
 'fit_objective':'global elementwise MSE, NOT mean(log per-sample NMSE)',
 'max_replay_difference_db':maxerr,'summary':summary,'both_sides_calibrated_gap_0db':gap,
 'checkpoint_sha256':checkpoints,'records':records,'seconds':time.time()-start,
 'limits':['historical synthetic data reused; not external confirmation','no RD-AE/GN/LN performance verified',
 'scalar-only SGCS invariant; output calibration is not confidence probability calibration',
 'test-fit least squares is not an upper bound for mean NMSE dB','no additive decomposition or capacity lower bound']}
(outdir/'affine_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'max_replay_difference_db':maxerr,'gap_0db':gap,'seconds':report['seconds']}),flush=True)
