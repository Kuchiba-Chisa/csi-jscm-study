"""Validation-only output scaling/affine calibration of frozen GN models.

Shared and specialist each receive the same calibration opportunity.
Raw test predictions use precisely the parent experiment's power and noise draws.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
spec=importlib.util.spec_from_file_location('gn_gap_base',BASE/'run_gn_gap.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def predictions(net,x,snr,power,repeats,offset):
    result=[]
    for rep in range(repeats):
        g=torch.Generator().manual_seed(offset+100*snr+rep)
        values=[]
        for xb in x.split(128):
            ss=torch.full((len(xb),),float(snr))
            eps=torch.randn((len(xb),64,2),generator=g)
            pred,_=B.forward(net,xb,ss,eps,power)
            values.append(pred)
        result.append(torch.cat(values))
    return torch.stack(result)

def fit(pred,target):
    p=pred.double().flatten()
    t=target.unsqueeze(0).expand_as(pred).double().flatten()
    pm,tm=p.mean(),t.mean()
    a=((p-pm)*(t-tm)).sum()/((p-pm).square().sum())
    b=tm-a*pm
    a0=(p*t).sum()/p.square().sum()
    # Deployable coefficients rounded to FP32 before evaluation.
    return {'a':float(a.float()),'b':float(b.float()),'scale_a':float(a0.float())}

@torch.no_grad()
def metrics(pred,target):
    ratio=(pred-target).square().sum((-1,-2,-3))/target.square().sum((-1,-2,-3))
    db=10*torch.log10(ratio+1e-12)
    sc=torch.stack([B.sgcs_per(p,target) for p in pred])
    return {'mean_nmse_db':float(db.mean()),'db_mean_linear_nmse':float(10*torch.log10(ratio.mean())),
            'sgcs':float(sc.mean())},{'db':db.numpy(),'ratio':ratio.numpy(),'sgcs':sc.numpy()}

def main():
    start=time.perf_counter();torch.set_num_threads(8)
    source=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
    xval=B.load01('val',root=str(B.PROJECT/'data'))
    xtest=B.load01('test',root=str(B.PROJECT/'data'))
    pairs=[('shared',s) for s in B.SNRS]+[(f'specialist_{s}',s) for s in B.SNRS]
    inputs=[BASE/'results.json',BASE/'observations.npz',BASE/'protocol.json',BASE/'run_gn_gap.py']
    inputs += [BASE/'checkpoints'/f'{name}.pt' for name in B.MODELS]
    inputs += [B.PROJECT/'data'/f'{s}.npz' for s in ('train','val','test')]
    hashes={str(p):sha(p) for p in inputs}
    protocol={'models':pairs,'fit':'one global real affine and scale-only pair per model/SNR; FP32 coefficients',
              'val_fit_H':len(xval),'val_noise_repeats':3,'val_noise_seed_base':150000,
              'test_H':len(xtest),'test_noise_repeats':3,'test_noise_seed_base':50000,
              'power':'parent experiment train-only fixed actual-Ptx references, unchanged',
              'fit_objective':'global elementwise MSE, not mean per-H NMSEdB',
              'comparison':'both shared and specialist calibrated on val separately, no test fitting',
              'source_sha256':hashes,'script_sha256':sha(__file__),
              'limits':'one training seed; same historical synthetic data; scalar scaling does not improve SGCS'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    rows={};arrays={};replay=0.;scale_sc_delta=0.;normal_eq=[]
    parent_arrays=np.load(BASE/'observations.npz')
    for name,snr in pairs:
        ck=torch.load(BASE/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
        net=B.LinkRDAE_GN(cond=False);net.load_state_dict(ck['state_dict']);net.eval()
        before=B.state_hash(net)
        power=source['results'][name][str(snr)]['train_power_reference']
        pv=predictions(net,xval,snr,power,3,150000)
        coeff=fit(pv,xval)
        # Check least-squares residual orthogonality on the validation data.
        pp=pv.double();tt=xval.double().unsqueeze(0)
        scale_residual=tt-coeff['scale_a']*pp
        affine_residual=tt-(coeff['a']*pp+coeff['b'])
        norm={'model':name,'snr':snr,'scale_residual_times_pred_mean':float((scale_residual*pp).mean()),
              'affine_residual_mean':float(affine_residual.mean()),
              'affine_residual_times_pred_mean':float((affine_residual*pp).mean())}
        assert max(abs(v) for k,v in norm.items() if k not in ('model','snr'))<1e-7
        normal_eq.append(norm)
        pt=predictions(net,xtest,snr,power,3,50000)
        row={'coefficients':coeff,'train_power_reference':power}
        for method,pred in [('raw',pt),('scale_only',coeff['scale_a']*pt),('affine',coeff['a']*pt+coeff['b'])]:
            summary,obs=metrics(pred,xtest)
            row[method]=summary
            for k,v in obs.items():arrays[f'{name}__snr{snr}__{method}__{k}']=v
        key=f'{name}__snr{snr}__'
        replay=max(replay,float(np.abs(arrays[key+'raw__db']-parent_arrays[key+'db']).max()))
        assert replay<2e-5,replay
        scdelta=float(np.abs(arrays[key+'raw__sgcs']-arrays[key+'scale_only__sgcs']).max())
        scale_sc_delta=max(scale_sc_delta,scdelta)
        assert scdelta<1e-5
        assert B.state_hash(net)==before
        rows.setdefault(name,{})[str(snr)]=row
        print('CALIBRATED',name,snr,'coeff',coeff,'gain',row['raw']['mean_nmse_db']-row['affine']['mean_nmse_db'],flush=True)
    gaps={}
    for snr in B.SNRS:
        gaps[str(snr)]={}
        for method in ('raw','scale_only','affine'):
            a=f'shared__snr{snr}__{method}__';b=f'specialist_{snr}__snr{snr}__{method}__'
            gaps[str(snr)][method]={'nmse_shared_minus_specialist':B.bootstrap(arrays[a+'db']-arrays[b+'db']),
                                    'sgcs_specialist_minus_shared':B.bootstrap(arrays[b+'sgcs']-arrays[a+'sgcs'])}
        gaps[str(snr)]['gap_reduction_affine']=B.bootstrap(
            (arrays[f'shared__snr{snr}__raw__db']-arrays[f'specialist_{snr}__snr{snr}__raw__db'])-
            (arrays[f'shared__snr{snr}__affine__db']-arrays[f'specialist_{snr}__snr{snr}__affine__db']))
    assert all(sha(p)==v for p,v in hashes.items())
    result={'protocol':protocol,'results':rows,'gaps':gaps,'seconds':time.perf_counter()-start,
            'audit':{'original_files_unchanged':True,'weights_unchanged':True,'raw_max_per_observation_replay_error_db':replay,
                     'scale_only_SGCS_max_absolute_difference':scale_sc_delta,'validation_normal_equations':normal_eq}}
    np.savez_compressed(OUT/'observations.npz',**arrays)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(result)
    print('FINISHED',result['seconds'],'gaps',json.dumps(gaps),flush=True)

def write_report(r):
    lines=['# GN 共享与专用：输出仿射校准','',
           '冻结上一轮四个GN网络，固定Ns=64、16-QAM。每个模型每个SNR分别在验证1000H×3组独立噪声上拟合一个全局实数缩放，及一个全局实数仿射变换。没有主网络训练，没有在test拟合。',
           '共享和对应专用模型均给予同等校准，复用上一轮训练集标定的实际功率，以及相同test1000H×3组噪声。下表差距正为共享较差。','',
           '| SNR | 共享原始 | 共享缩放后 | 共享仿射后 | 专用原始 | 专用仿射后 | 原始共享差距 | 双方仿射后差距 |',
           '|---|---|---|---|---|---|---|---|']
    for s in B.SNRS:
        a=r['results']['shared'][str(s)];b=r['results'][f'specialist_{s}'][str(s)]
        g0=r['gaps'][str(s)]['raw']['nmse_shared_minus_specialist']['mean']
        g1=r['gaps'][str(s)]['affine']['nmse_shared_minus_specialist']['mean']
        vals=[a['raw']['mean_nmse_db'],a['scale_only']['mean_nmse_db'],a['affine']['mean_nmse_db'],
              b['raw']['mean_nmse_db'],b['affine']['mean_nmse_db'],g0,g1]
        lines.append('| '+str(s)+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
    lines+=['','所有NMSE均为平均逐样本NMSE_dB，越低越好。差距减少量需用双方校准前后比较，不能只看共享模型自己改善。','',
            '## 系数与方向指标','',
            '| 模型 | SNR | 仅缩放 a | 仿射 a | 仿射 b | 原始 SGCS | 仿射 SGCS |',
            '|---|---|---|---|---|---|---|']
    for name in B.MODELS:
        for s,row in r['results'].get(name,{}).items():
            c=row['coefficients']
            lines.append(f"| {name} | {s} | {c['scale_a']:.5f} | {c['a']:.5f} | {c['b']:.6f} | {row['raw']['sgcs']:.5f} | {row['affine']['sgcs']:.5f} |")
    lines+=['','## 结论边界','',
            '拟合使用线性MSE最小二乘，报告均值NMSE_dB，两者目标不同，测试效果不保证处处改善。a,b按FP32使用。当前数据先逐H实虚RMS归一化再除3，系数属于这一输入输出坐标系。',
            '纯全局缩放不改变平方SGCS，因此NMSE改善不能直接解释为方向或吞吐改善。完整仿射的平移会改变方向，但需看实际SGCS结果。',
            '只做一个训练种子、合成CSI与AWGN；同一历史验证/测试集已用于项目分析；不把H bootstrap当作重训稳定性。新结果不与前期CNN的0.47dB相加或外推。','',
            '代码affine_gn.py；protocol.json记录源文件和权重哈希；results.json含系数、逐模型指标、双方校准后差距与配对H区间；observations.npz保存逐样本结果。']
    (OUT/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
