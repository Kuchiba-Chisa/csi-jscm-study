"""Prospective single-seed, fixed-Ns RD-AE Q1 supplement.

Original project is read only. All writes go to this script's own directory.
No claim of official full RD-JSCC reproduction or three-seed confirmation.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[2] / 'simulation'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
from model_rdae import LinkRDAE
from model_v4 import normalize_power,to_symbols,from_symbols,qam_map_ste,tx_power
from train_rdae import load01
from quality_v5 import sgcs_per

ANCHORS=(0,10,20)
GRID=(0,5,10,15,20)
NAMES=('teacher_0','teacher_10','teacher_20','shared','weighted','distilled')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(name,data):
    p=OUT/name
    temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,p)

def data():return tuple(load01(s,root=str(PROJECT/'data')) for s in ('train','val','test'))
def scalar_seed(seed,offset):return torch.Generator().manual_seed(seed+offset)

def forward(net,x,snr,eps,power=None):
    feat=(10**(-snr/20)).view(-1,1)
    raw=net.enc(x,feat)
    z=normalize_power(raw,64)
    symbols=qam_map_ste(to_symbols(z))
    actual=tx_power(symbols,64)
    reference=actual.detach().clamp_min(1e-6) if power is None else torch.full_like(actual,float(power))
    sigma=(reference/2).sqrt()*10**(-snr/20)
    y=from_symbols(symbols+eps*sigma[:,None,None])
    return net.dec(y,feat),y,actual

def conditions(seed,epochs,steps,bs):
    """Same schedules/noises are used by all three shared arms."""
    gen=scalar_seed(seed,91001)
    result=[]
    for epoch in range(epochs):
        batch=[]
        for _ in range(steps):
            if torch.rand((),generator=gen).item()<.5:
                a=ANCHORS[int(torch.randint(3,(),generator=gen))]
                batch.append((a,torch.full((bs,),float(a))))
            else:batch.append((None,torch.rand(bs,generator=gen)*20))
        result.append(batch)
    return result

def save_model(path,net,metadata):
    torch.save(dict(state_dict=net.state_dict(),metadata=json.loads(json.dumps(metadata))),path)

def load_model(name):
    bundle=torch.load(OUT/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
    net=LinkRDAE(cond=False)
    net.load_state_dict(bundle['state_dict'])
    return net.eval(),bundle['metadata']

@torch.no_grad()
def validation(net,x,seed):
    net.eval()
    result={}
    for s in ANCHORS:
        values=[]
        g=scalar_seed(seed,92000+s)
        for xb in x[:400].split(128):
            snr=torch.full((len(xb),),float(s))
            eps=torch.randn((len(xb),64,2),generator=g)
            pred,_,_=forward(net,xb,snr,eps)
            r=(pred-xb).square().sum((1,2,3))/xb.square().sum((1,2,3))
            values.append(10*(r+1e-12).log10())
        result[str(s)]=float(torch.cat(values).mean())
    return result

def train(name,args,xtr,xval,schedule):
    path=OUT/'checkpoints'/f'{name}.pt'
    if path.exists():
        net,meta=load_model(name)
        print('RESUME_COMPLETE',name,'sha',sha(path),flush=True)
        return net
    torch.manual_seed(args.seed)
    net=LinkRDAE(cond=False)
    teachers={}
    if name=='distilled':
        for s in ANCHORS:
            teacher,_=load_model(f'teacher_{s}')
            for p in teacher.parameters():p.requires_grad_(False)
            teachers[s]=teacher
    opt=torch.optim.Adam(net.parameters(),lr=.002,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    ordergen=scalar_seed(args.seed,93000)
    noisegen=scalar_seed(args.seed,94000)
    steps=len(xtr)//args.bs
    trace=[]
    started=time.perf_counter()
    for ep in range(args.epochs):
        net.train()
        permutation=torch.randperm(len(xtr),generator=ordergen)
        sums=np.zeros(3)
        for step in range(steps):
            xb=xtr[permutation[step*args.bs:(step+1)*args.bs]]
            anchor,snr=schedule[ep][step]
            if name.startswith('teacher_'):
                snr=torch.full((len(xb),),float(name.split('_')[1]))
            eps=torch.randn((len(xb),64,2),generator=noisegen)
            pred,_,_=forward(net,xb,snr,eps)
            rec=(pred-xb).square().mean()
            # E[w]=1+2*(1/2)*(1/3)=4/3; normalize constant fixes global scale.
            weight=3. if name=='weighted' and anchor==0 else 1.
            objective=rec*weight/(4/3 if name=='weighted' else 1.)
            kd=torch.zeros(())
            if name=='distilled' and anchor is not None:
                # Same H, SNR and standardized Gaussian draw. Each model uses
                # its own transmitted symbols and actual-power noise scale.
                with torch.no_grad():target,_,_=forward(teachers[anchor],xb,snr,eps)
                kd=(pred-target).square().mean()
                objective=objective+args.lam*kd
            opt.zero_grad(set_to_none=True)
            objective.backward()
            opt.step()
            sums+=np.array([float(rec.detach()),float(kd.detach()),float(objective.detach())])
        scheduler.step()
        row=dict(epoch=ep+1,reconstruction_mse=sums[0]/steps,kd_mse=sums[1]/steps,
                 objective=sums[2]/steps,elapsed_seconds=time.perf_counter()-started)
        trace.append(row)
        dump('progress.json',dict(stage='training',model=name,epoch=ep+1,epochs=args.epochs,
                                 elapsed_seconds=row['elapsed_seconds'],last_epoch=row))
        dump(f'trace_{name}.json',trace)
        if (ep+1)%5==0 or ep==0:
            print('TRAIN',name,'epoch',ep+1,'rec',round(row['reconstruction_mse'],6),
                  'kd',round(row['kd_mse'],6),'seconds',round(row['elapsed_seconds'],1),flush=True)
    net.eval()
    meta=dict(name=name,seed=args.seed,epochs=args.epochs,ns=64,
              checkpoint_selection='fixed final epoch, no early stopping/test selection',
              validation_unfrozen_power=validation(net,xval,args.seed),
              seconds=time.perf_counter()-started,trace=trace)
    save_model(path,net,meta)
    print('SAVED',name,'seconds',round(meta['seconds'],1),'val',meta['validation_unfrozen_power'],flush=True)
    return net

@torch.no_grad()
def power_calibration(net,xtr):
    net.eval();result={}
    for s in GRID:
        total=0.;count=0
        for xb in xtr.split(128):
            feat=torch.full((len(xb),1),10**(-s/20))
            symbols=qam_map_ste(to_symbols(normalize_power(net.enc(xb,feat),64)))
            actual=tx_power(symbols,64)
            total+=float(actual.sum());count+=len(xb)
        result[str(s)]=total/count
    return result

def state_hash(net,prefix=None):
    digest=hashlib.sha256()
    for k,v in net.state_dict().items():
        if prefix is None or k.startswith(prefix):
            digest.update(k.encode());digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

@torch.no_grad()
def recalibrate(net,xtr,power,snr,seed):
    before_parameters={n:p.clone() for n,p in net.named_parameters()}
    encoder=state_hash(net,'enc.')
    net.eval()
    bns=[m for m in net.dec.modules() if isinstance(m,nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats();bn.momentum=None;bn.train()
    order=torch.randperm(len(xtr),generator=scalar_seed(seed,95000))
    g=scalar_seed(seed,96000)
    for ids in order.split(128):
        xb=xtr[ids]
        if snr is None:
            # Same mixture family as shared training. A cumulative pass uses
            # continuous SNR-specific training-set power interpolation.
            if torch.rand((),generator=g)<.5:
                ss=torch.full((len(xb),),float(ANCHORS[int(torch.randint(3,(),generator=g))]))
            else:ss=torch.rand(len(xb),generator=g)*20
        else:ss=torch.full((len(xb),),float(snr))
        feat=(10**(-ss/20)).view(-1,1)
        tx=qam_map_ste(to_symbols(normalize_power(net.enc(xb,feat),64)))
        eps=torch.randn(tx.shape,generator=g)
        ref=torch.tensor(np.interp(ss.numpy(),GRID,[power[str(s)] for s in GRID]),dtype=torch.float32)
        sigma=(ref/2).sqrt()*10**(-ss/20)
        y=from_symbols(tx+eps*sigma[:,None,None])
        net.dec(y,feat)
    net.eval()
    for bn in bns:bn.momentum=.1
    assert state_hash(net,'enc.')==encoder
    assert all(torch.equal(p,before_parameters[n]) for n,p in net.named_parameters())

@torch.no_grad()
def observe(net,x,snr,power,seed):
    net.eval()
    ratios=[];sgcs=[];y_hash=hashlib.sha256()
    for repeat in range(3):
        g=scalar_seed(seed,97000+100*snr+repeat)
        r=[];sc=[]
        for xb in x.split(128):
            ss=torch.full((len(xb),),float(snr))
            eps=torch.randn((len(xb),64,2),generator=g)
            pred,y,_=forward(net,xb,ss,eps,power=power[str(snr)])
            y_hash.update(y.numpy().tobytes())
            r.append(((pred-xb).square().sum((1,2,3))/xb.square().sum((1,2,3))).numpy())
            sc.append(sgcs_per(pred,xb).numpy())
        ratios.append(np.concatenate(r));sgcs.append(np.concatenate(sc))
    ratios=np.array(ratios);sc=np.array(sgcs)
    db=10*np.log10(ratios+1e-12)
    return dict(mean_nmse_db=float(db.mean()),db_mean_linear_nmse=float(10*np.log10(ratios.mean())),
                sgcs=float(sc.mean()),y_sha256=y_hash.hexdigest()),dict(db=db,ratio=ratios,sgcs=sc)

def ci(a,b,seed):
    # Improvement relative to b, positive means method a has lower loss.
    differences=(b-a).mean(0)
    g=np.random.default_rng(seed)
    draws=differences[g.integers(0,len(differences),size=(2000,len(differences)))].mean(1)
    return dict(improvement_db=float(differences.mean()),H_paired_CI95=np.quantile(draws,[.025,.975]).tolist(),
                unit='H cluster; 3 noises kept together; one backbone seed only')

def evaluate_all(args,xtr,xval,xtest):
    allrows={};arrays={};power_refs={}
    # Fixed test H600:1000 used previously; new noise does not make a new external data set.
    x=xtest[600:1000]
    for name in NAMES:
        source,_=load_model(name)
        source_hash=state_hash(source)
        power=power_calibration(source,xtr)
        power_refs[name]=power
        allrows[name]={}
        for mode in ('raw','mixed','target'):
            if mode=='mixed' and name.startswith('teacher_'):continue
            for snr in GRID:
                if name.startswith('teacher_') and snr!=int(name.split('_')[1]):continue
                net=copy.deepcopy(source)
                if mode!='raw':recalibrate(net,xtr,power,None if mode=='mixed' else snr,args.seed)
                row,arr=observe(net,x,snr,power,args.seed)
                key=f'{name}/{mode}/{snr}'
                arrays[key]=arr
                allrows[name].setdefault(mode,{})[str(snr)]=row
                if mode!='raw':assert row['y_sha256']==allrows[name]['raw'][str(snr)]['y_sha256']
                print('EVAL',key,'meanNMSE',round(row['mean_nmse_db'],5),flush=True)
                dump('progress.json',dict(stage='evaluating',condition=key))
            assert state_hash(source)==source_hash
        dump('evaluation_partial.json',dict(results=allrows,power=power_refs))
    contrasts={}
    for s in ANCHORS:
        contrasts[str(s)]={}
        original=arrays[f'shared/raw/{s}']['db']
        for method in ('weighted','distilled'):
            contrasts[str(s)][method+'_vs_shared_raw']=ci(arrays[f'{method}/raw/{s}']['db'],original,args.seed+s)
        for mode in ('raw','mixed','target'):
            current=arrays[f'shared/{mode}/{s}']['db']
            # Gap > 0 means shared is worse than a teacher at this exact SNR.
            tmode='raw' if mode=='raw' else 'target'
            teacher=arrays[f'teacher_{s}/{tmode}/{s}']['db']
            gap=ci(teacher,current,args.seed+s)
            contrasts[str(s)]['shared_'+mode+'_minus_specialist_'+tmode]=dict(
                gap_db=gap['improvement_db'],H_paired_CI95=gap['H_paired_CI95'])
        for method in ('shared','weighted','distilled'):
            contrasts[str(s)][method+'_target_vs_raw']=ci(arrays[f'{method}/target/{s}']['db'],arrays[f'{method}/raw/{s}']['db'],args.seed+s)
    flat={k.replace('/','__')+'__'+metric:v for k,row in arrays.items() for metric,v in row.items()}
    np.savez_compressed(OUT/'test_arrays.npz',**flat)
    dump('results.json',dict(results=allrows,power=power_refs,contrasts=contrasts,
        note='fixed64, singleseed, finite16QAM adapted RD-AE; not official full RD-JSCC; teacher gap only at exact 0/10/20'))
    return allrows,contrasts

def summarize(rows,contrasts):
    lines=['# RD-AE Q1 补充：固定 Ns=64 单种子试跑',
    '所有模型从头训练，均含编码端和解码端 AF；16-QAM 固定。主表是最终40轮，不根据测试挑轮次、λ或权重。单种子仅为探索，H聚类区间不代表训练种子稳定性。',
    '## 原始BN状态：专用模型与三种共享训练',
    '| SNR | 专用 | 共享 | 重加权 | 蒸馏 | 共享−专用 |','|---|---|---|---|---|---|']
    for s in ANCHORS:
        vals=[rows[f'teacher_{s}']['raw'][str(s)]['mean_nmse_db']]+[rows[n]['raw'][str(s)]['mean_nmse_db'] for n in ('shared','weighted','distilled')]
        lines.append('| '+str(s)+' | '+' | '.join(f'{v:.4f}' for v in vals)+f' | {vals[1]-vals[0]:+.4f} |')
    lines+=['## BN与训练目标：每种方法独立评价','| SNR | 模型 | 原始 | 混合重统计 | 目标重统计 | 目标−原始改善 |','|---|---|---|---|---|---|']
    for s in GRID:
        for name in ('shared','weighted','distilled'):
            vals=[rows[name][mode][str(s)]['mean_nmse_db'] for mode in ('raw','mixed','target')]
            lines.append(f'| {s} | {name} | '+' | '.join(f'{v:.4f}' for v in vals)+f' | {vals[0]-vals[2]:+.4f} |')
    lines+=['## 读数边界',
    '- 专用教师也有目标BN重统计对照；先看原始差距，再看双方都按目标校准后的剩余差距，不能只改善共享侧就宣称追平。',
    '- 固定Ns64用于隔离跨SNR共享，不包含多长度共享代价；不能与旧多长度权重的曲线直接拼接。',
    '- 重加权目标按期望权重4/3归一化；只加重0dB锚点。蒸馏λ=1，仅锚点batch启用，教师与学生共用标准高斯噪声而各自使用实际发送功率。',
    '- 6000/1000/1000原数据；test取H600:1000，3次新噪声。这些H历史上使用过，不是外部确认集。',
    '- NMSE主表为逐样本dB均值，另存dB(线性NMSE均值)与直接空频域平方SGCS。',
    '- 训练成本包含3个专用教师；部署学生不需要保留教师。BN校准依赖源域训练数据，无权重更新。',
    '- 若教师没有更好，不能预设蒸馏应提高；若某项只有单点改善，不称普遍有效。',
    '- 训练曲线、权重、协议、配对H区间、实际功率和接收y哈希见同目录JSON/NPZ。']
    (OUT/'初步结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--bs',type=int,default=128)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--lam',type=float,default=1.)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.smoke:
        print('SMOKE: in-memory checks only; no model/data/checkpoint writes',flush=True)
        torch.manual_seed(args.seed)
        net=LinkRDAE().eval()
        x=data()[0][:4]
        s=torch.tensor([0.,5.,10.,20.])
        torch.manual_seed(999);original=net(x,s,64)
        torch.manual_seed(999);eps=torch.randn((4,64,2));out,_,_=forward(net,x,s,eps)
        torch.testing.assert_close(out,original,atol=1e-6,rtol=1e-5)
        out.square().mean().backward()
        assert sum(float(p.grad.abs().sum()) for p in net.enc.parameters() if p.grad is not None)>0
        assert sum(float(p.grad.abs().sum()) for p in net.dec.parameters() if p.grad is not None)>0
        print('PASS: supplied-noise forward matches original; both ends receive gradients',flush=True)
        return
    (OUT/'checkpoints').mkdir(exist_ok=True)
    xtr,xval,xtest=data()
    protected=[PROJECT/'model_rdae.py',PROJECT/'model_v4.py',PROJECT/'train_rdae.py']+list((PROJECT/'ckpt_rdae').glob('*.pt'))+[PROJECT/'data'/f'{s}.npz' for s in ('train','val','test')]
    hashes={str(p):sha(p) for p in protected}
    schedule=conditions(args.seed,args.epochs,len(xtr)//args.bs,args.bs)
    protocol=dict(args=vars(args),scope='fixedNs64,adaptedRD-AE,AFbothsides,finite16QAM,AWGN,one_seed',
        training='all six from identical per-seed initialization; Adam2e-3 wd1e-5 cosine; 46 batches/epoch, 40 final epochs; original files read only',
        shared_conditions='half batches anchor0/10/20; rest per-sampleU0,20; fixedNs64; shared arms share order, conditions and eps',
        weighted='0dB anchor weight3,rest1; objective divided by constant expected weight4/3',
        distilled='lambda1 on anchor batches only; teacher frozen eval; same standardized Gaussian draw, per-model actual-power scale; no teacher at deployment',
        evaluation='fixed train-calibrated per-model per-SNR power; 0/5/10/15/20; testH600:1000,3 common Gaussian draws; two NMSE definitions and direct squaredSGCS',
        bn='raw,mixed for students,target; each whole6000 trainingH pass; parameters/encoder/power fixed; teachers raw and target at their ownSNR',
        limits='singleseed pilot, known historical testpopulation; fixedlength cannot characterize multi-lengthsharing; no official fullRDJSCC',
        planned_models=list(NAMES),source_sha256=hashes,script_sha256=sha(__file__))
    if (OUT/'protocol.json').exists():
        old=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        assert old['script_sha256']==protocol['script_sha256'],'Source changed after launch; isolate a new output directory.'
        assert old['args']==protocol['args'],'Different settings require a new output directory.'
    else:dump('protocol.json',protocol)
    started=time.perf_counter()
    for name in NAMES:train(name,args,xtr,xval,schedule)
    rows,contrasts=evaluate_all(args,xtr,xval,xtest)
    summarize(rows,contrasts)
    assert hashes=={str(p):sha(p) for p in protected},'Original project files changed'
    dump('audit.json',dict(status='PASS',original_sources_checkpoints_data_unchanged=True,
                           supplied_noise_forward_smoke='run separately before launch',
                           total_seconds=time.perf_counter()-started,models=list(NAMES)))
    dump('progress.json',dict(stage='complete',seconds=time.perf_counter()-started))
    print('COMPLETE',round(time.perf_counter()-started,1),flush=True)

if __name__=='__main__':main()
