"""Unified evaluation of existing BN-S0 / BN-S2 / GN checkpoints, no training.

No source/checkpoint/data writes. Source train_rdae_gn is NOT imported because
its module-level monkey patch changes train_rdae.LinkRDAE. Its exact grouping
and recursive replacement are reproduced and checked against saved weights.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
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
from model_v4 import normalize_power,qam_map_ste,to_symbols,from_symbols,pair_mask,tx_power
from quality_v5 import load_split,nmse_db_per,sgcs_per

SNRS=(0,10,20)
LENGTHS=(16,32,64)
MODES=('BN_S0','BN_S2','GN')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(name,data):
    p=OUT/name;t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(t,p)
def seedgen(seed):return torch.Generator().manual_seed(seed)

def swap_gn(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.BatchNorm2d):
            groups=next(g for g in (4,2,1) if child.num_features%g==0)
            setattr(module,name,nn.GroupNorm(groups,child.num_features,eps=child.eps))
        else:swap_gn(child)

def build(seed,gn=False):
    net=LinkRDAE(cond=False)
    if gn:
        swap_gn(net)
        path=PROJECT/'ckpt_rdae_gn'/f'rdae_gn_sample_s{seed}.pt'
        assert not any(isinstance(m,nn.BatchNorm2d) for m in net.modules())
    else:path=PROJECT/'ckpt_rdae'/f'rdae_sample_s{seed}.pt'
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    net.eval()
    return net,path

def tensor_hash(net,prefix=None):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():
        if prefix is None or k.startswith(prefix):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def encode(net,x,snr,ns):
    feature=torch.full((len(x),1),10**(-snr/20))
    return qam_map_ste(to_symbols(normalize_power(net.enc(x,feature),ns)))

@torch.no_grad()
def calibrate_power(net,xtr,bs):
    state=tensor_hash(net);net.eval();powers={}
    for ns in LENGTHS:
        powers[str(ns)]={}
        for s in SNRS:
            total=0.;n=0
            for xb in xtr.split(bs):
                symbols=encode(net,xb,s,ns)
                total+=float(tx_power(symbols,ns).sum());n+=len(xb)
            powers[str(ns)][str(s)]=total/n
    assert tensor_hash(net)==state
    return powers

@torch.no_grad()
def observe(net,x,snr,ns,power,eps):
    symbols=encode(net,x,snr,ns)
    mask=pair_mask(torch.full((len(x),),ns,dtype=torch.long)).view(len(x),64,2)
    sigma=math.sqrt(power/2)*10**(-snr/20)
    y=from_symbols(symbols+eps*sigma*mask)*mask.reshape(len(x),128)
    features=torch.full((len(x),1),10**(-snr/20))
    return net.dec(y,features),y,tx_power(symbols,ns)

@torch.no_grad()
def forward_check(net,x,powers):
    net.eval();checks=[]
    # Select exact per-condition power, with no cross-SNR merging approximation.
    net.p_cal={(ns,float(s)):powers[str(ns)][str(s)] for ns in LENGTHS for s in SNRS}
    net.p_cal_mode='ns_snr';net.p_cal_snrs=tuple(float(s) for s in SNRS)
    for ns,s in ((16,0),(32,10),(64,20)):
        torch.manual_seed(102400+ns+s)
        original=net(x,s,ns)
        torch.manual_seed(102400+ns+s)
        eps=torch.randn((len(x),64,2))
        manual,y,_=observe(net,x,s,ns,powers[str(ns)][str(s)],eps)
        error=float((manual-original).abs().max())
        assert error<2e-6,(ns,s,error)
        assert torch.count_nonzero(y[:,2*ns:])==0
        checks.append(dict(ns=ns,snr=s,max_abs_difference=error))
    return checks

@torch.no_grad()
def recalibrate_decoder(source,xtr,snr,ns,power,seed,bs):
    net=copy.deepcopy(source).eval()
    weights={n:p.clone() for n,p in net.named_parameters()}
    enc_hash=tensor_hash(net,'enc.')
    bns=[m for m in net.dec.modules() if isinstance(m,nn.BatchNorm2d)]
    assert len(bns)==16
    for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
    g=seedgen(217000+seed)
    order=torch.randperm(len(xtr),generator=g)
    noise=seedgen(218000+seed+ns*100+snr)
    for ids in order.split(bs):
        xb=xtr[ids]
        eps=torch.randn((len(xb),64,2),generator=noise)
        observe(net,xb,snr,ns,power,eps)
    net.eval()
    assert tensor_hash(net,'enc.')==enc_hash
    assert all(torch.equal(p,weights[n]) for n,p in net.named_parameters())
    changed=[k for k,v in net.state_dict().items() if not torch.equal(v,source.state_dict()[k])]
    assert changed and all(k.startswith('dec.') and k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changed)
    state={k:v.clone() for k,v in net.state_dict().items() if k.startswith('dec.') and k.endswith(('running_mean','running_var','num_batches_tracked'))}
    torch.save(state,OUT/'bn_states'/f's{seed}_ns{ns}_snr{snr}.pt')
    return net,changed

@torch.no_grad()
def evaluate(net,x,snr,ns,power,seed,bs,repeats):
    state=tensor_hash(net);ratios=[];sgcs=[];yhash=hashlib.sha256();epshash=hashlib.sha256();ptx=[]
    for repeat in range(repeats):
        # Same H, order and standard Gaussian draws across BN/GN. Their y may
        # differ due to encoders/power; BN-S0/S2 y must be identical.
        g=seedgen(219000+seed*10000+ns*100+snr*10+repeat)
        r=[];sc=[]
        for xb in x.split(bs):
            eps=torch.randn((len(xb),64,2),generator=g)
            pred,y,p=observe(net,xb,snr,ns,power,eps)
            yhash.update(y.numpy().tobytes());epshash.update(eps.numpy().tobytes())
            linear=((pred-xb).square().sum((1,2,3))/xb.square().sum((1,2,3))).numpy()
            torch.testing.assert_close(nmse_db_per(pred,xb),torch.from_numpy(10*np.log10(linear+1e-12)),atol=3e-6,rtol=1e-5)
            r.append(linear);sc.append(sgcs_per(pred,xb).numpy());ptx.append(p.numpy())
        ratios.append(np.concatenate(r));sgcs.append(np.concatenate(sc))
    ratios=np.asarray(ratios);sc=np.asarray(sgcs);db=10*np.log10(ratios+1e-12)
    assert tensor_hash(net)==state
    metrics=dict(mean_nmse_db=float(db.mean()),db_mean_linear_nmse=float(10*np.log10(ratios.mean())),
                 sgcs=float(sc.mean()),p_ref=power,test_p_tx=float(np.concatenate(ptx).mean()),
                 y_sha256=yhash.hexdigest(),standard_noise_sha256=epshash.hexdigest())
    return metrics,dict(db=db,ratio=ratios,sgcs=sc)

def mean_sd(values):
    v=np.asarray(values,dtype=float)
    return dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),values=v.tolist())

def contrasts(arrays):
    result={}
    # Positive NMSE delta = A worse. Positive SGCS delta = A better.
    for ns in LENGTHS:
        for snr in SNRS:
            key=f'ns{ns}_snr{snr}';cell={}
            for a,b in [('GN','BN_S0'),('GN','BN_S2'),('BN_S0','BN_S2')]:
                differences=[];seeds=[];sg_diffs=[];linear_diffs=[]
                for seed in range(3):
                    ar=arrays[(seed,ns,snr,a)];br=arrays[(seed,ns,snr,b)]
                    by_h=(ar['db']-br['db']).mean(0)
                    differences.append(float(by_h.mean()))
                    sg_diffs.append(float((ar['sgcs']-br['sgcs']).mean()))
                    linear_diffs.append(float(10*np.log10(ar['ratio'].mean())-10*np.log10(br['ratio'].mean())))
                    g=np.random.default_rng(220000+seed+ns+snr)
                    samples=g.integers(0,len(by_h),size=(1500,len(by_h)))
                    interval=np.quantile(by_h[samples].mean(1),[.025,.975])
                    seeds.append(dict(seed=seed,mean_nmse_delta_db=float(by_h.mean()),H_cluster_CI95=interval.tolist()))
                summary=mean_sd(differences)
                radius=4.3026527299*summary['sample_sd']/math.sqrt(3)
                summary['paired_seed_t_CI95']=[summary['mean']-radius,summary['mean']+radius]
                summary['per_seed_H_intervals']=seeds
                summary['sgcs_delta']=mean_sd(sg_diffs)
                summary['db_mean_linear_nmse_delta']=mean_sd(linear_diffs)
                summary['caveat']='3 source model pairs; tCI df2 exploratory, no equivalence claim, no multiple-comparison adjustment; CPU/GPU historical training not bitwise paired'
                cell[a+'_minus_'+b]=summary
            result[key]=cell
    return result

def summarize(rows,diffs):
    summary={}
    for ns in LENGTHS:
        for snr in SNRS:
            key=f'ns{ns}_snr{snr}';summary[key]={}
            for mode in MODES:
                summary[key][mode]={metric:mean_sd([rows[str(seed)][key][mode][metric] for seed in range(3)]) for metric in ('mean_nmse_db','db_mean_linear_nmse','sgcs','p_ref')}
    lines=['# BN原始／条件重标定／GN：统一协议三种子比较','',
      '2026-11-14。现存修正后BN权重与GN权重，有限16-QAM、RD-AE改编模型；只评估及重统计，不训练。主表越负越好，±是3个模型的样本标准差。','',
      '## 统一设置','',
      '- train6000H用于实际发射功率校准与目标BN统计；test全部1000H×3组新标准高斯噪声；SNR0/10/20，Ns16/32/64。',
      '- BN-S0与BN-S2使用完全相同y、冻结功率和编码器/解码器权重；只有接收端BN状态改变。',
      '- BN/GN的编码器不同，不要求y相同；匹配同一H、标准噪声抽样、符号预算和实际功率SNR口径。',
      '- GN将编码端和解码端全部18层BN替换；该比较不单独归因于接收端。',
      '- 当前train_rdae.evaluate本来就是mean(NMSE_dB)，早前“另一种batch口径”的解释不成立。本次另存两种公式以免混淆。','',
      '## 主结果：mean(逐样本NMSE_dB)','',
      '| Ns | SNR | BN-S0 | BN-S2 | GN | GN−S0 | GN−S2 | S0−S2 |',
      '|---|---|---|---|---|---|---|---|']
    for ns in LENGTHS:
        for s in SNRS:
            key=f'ns{ns}_snr{s}';r=summary[key];d=diffs[key]
            vals=[f"{r[m]['mean_nmse_db']['mean']:.4f} ± {r[m]['mean_nmse_db']['sample_sd']:.4f}" for m in MODES]
            deltas=[d[k]['mean'] for k in ('GN_minus_BN_S0','GN_minus_BN_S2','BN_S0_minus_BN_S2')]
            lines.append(f'| {ns} | {s} | '+' | '.join(vals)+' | '+' | '.join(f'{v:+.4f}' for v in deltas)+' |')
    lines+=['','GN−S2为正表示GN误差更大，即重标定BN在该格均值更好。S0−S2为正表示重标定改善。','',
      '## 直接比较GN与BN-S2：训练种子差值区间','',
      '| Ns | SNR | GN−BN-S2均值/dB | 配对三种子t区间95% | 各种子方向 |','|---|---|---|---|---|']
    for ns in LENGTHS:
        for s in SNRS:
            r=diffs[f'ns{ns}_snr{s}']['GN_minus_BN_S2'];lo,hi=r['paired_seed_t_CI95']
            lines.append(f"| {ns} | {s} | {r['mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "+', '.join(f'{v:+.4f}' for v in r['values'])+' |')
    lines+=['','n=3区间很宽；未拒绝零差不等于等价。各测试H的bootstrap区间另存，不能把1000H×3噪声冒充3000个训练种子；未做多重比较校正。','',
      '## 另一种NMSE与平方SGCS','',
      '| Ns | SNR | dB(mean线性NMSE)：S0 / S2 / GN | 平方SGCS：S0 / S2 / GN |','|---|---|---|---|']
    for ns in LENGTHS:
        for s in SNRS:
            r=summary[f'ns{ns}_snr{s}']
            lines.append(f'| {ns} | {s} | '+' / '.join(f"{r[m]['db_mean_linear_nmse']['mean']:.4f}" for m in MODES)+' | '+' / '.join(f"{r[m]['sgcs']['mean']:.4f}" for m in MODES)+' |')
    lines+=['','## 成本与边界','',
      '- BN解码器16层、通道数合计132：一部署点只存mean/var的FP32有效数据为1056字节，9条件为9504字节；不含文件封装。',
      '- GN不依赖BN冻结统计，但不代表整个系统无失配、无需维护。换GN需要重训；本次已复用既有权重。',
      '- 本轮仅比较这一个GN分组设置，不能代表LN/GDN/IN，也没有GN专用模型，不直接度量GN的专用—共享代价。',
      '- 功率及重标定均使用源域训练H；目标SNR/Ns视为已知。没有测真实SNR估计误差、场景变化或吞吐。',
      '- BN与GN历史训练代码/设备的完整逐步来源无法追溯；本次保证评估协议统一，不声称当初训练随机数逐位一致。',
      '- test H此前用于项目分析；新噪声和协议锁定不等于外部独立数据复现。',
      '- 没有根据当前test重选GN分组、训练轮次、BN校准样本数。','',
      '详细逐种子数据、哈希、指标与区间见results.json；逐观测数组见observations.npz；原始文件保持不变审计见audit.json。']
    (OUT/'统一协议比较结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--bs',type=int,default=128);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    (OUT/'bn_states').mkdir(exist_ok=True)
    protected=[PROJECT/f for f in ('model_rdae.py','model_v4.py','train_rdae.py','train_rdae_gn.py','quality_v5.py')]
    protected += [PROJECT/folder/f'{name}_s{s}.pt' for folder,name in [('ckpt_rdae','rdae_sample'),('ckpt_rdae_gn','rdae_gn_sample')] for s in range(3)]
    protected += [PROJECT/'data'/f'{s}.npz' for s in ('train','val','test')]
    hashes={str(p):sha(p) for p in protected}
    protocol=dict(args=vars(args),seeds=[0,1,2],snrs=list(SNRS),lengths=list(LENGTHS),
       source_sha256=hashes,script_sha256=sha(__file__),
       train='original6000 H, realRMS then /3; full set for exact per(ns,SNR) power and each targetBN calibration',
       test='all1000 originaltest H, 3 new standardGaussian draws/H/condition',
       protocol='no train/SGD; existing current BN-S0, targetdecoderBN-S2, all-networkGN; same H/eps and actual-Ptx SNR; BN-S0/S2 y identical',
       metrics='mean NMSE_dB and dB(mean ratio), direct spatial-frequency squaredSGCS',
       calibration='cumulative batch BN moments, bs128 one source-training pass; not exact global layerwise population moments; no label loss/optimization',
       limits='3 historical checkpoint pairs, no guarantee CPU/GPU training random streams identical; one GN group configuration; prior test population reused')
    if (OUT/'protocol.json').exists():
        previous=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        assert previous==protocol,'New source/settings require a new directory.'
    else:dump('protocol.json',protocol)
    xtr=load_split('train',root=str(PROJECT/'data'))/3
    xtest=load_split('test',root=str(PROJECT/'data'))/3
    rows={};arrays={};audit=[];start=time.perf_counter();powers={}
    for seed in range(3):
        bn,bpath=build(seed);gn,gpath=build(seed,gn=True)
        bn_initial=tensor_hash(bn);gn_initial=tensor_hash(gn)
        bp=calibrate_power(bn,xtr,args.bs);gp=calibrate_power(gn,xtr,args.bs)
        powers[str(seed)]={'BN':bp,'GN':gp}
        checks={'BN':forward_check(bn,xtest[:4],bp),'GN':forward_check(gn,xtest[:4],gp)}
        groups=Counter((m.num_groups,m.num_channels) for m in gn.modules() if isinstance(m,nn.GroupNorm))
        rows[str(seed)]={}
        for ns in LENGTHS:
            for s in SNRS:
                key=f'ns{ns}_snr{s}'
                s2,changed=recalibrate_decoder(bn,xtr,s,ns,bp[str(ns)][str(s)],seed,args.bs)
                rows[str(seed)][key]={}
                for mode,net,reference in [('BN_S0',bn,bp),('BN_S2',s2,bp),('GN',gn,gp)]:
                    metrics,arr=evaluate(net,xtest,s,ns,reference[str(ns)][str(s)],seed,args.bs,args.repeats)
                    rows[str(seed)][key][mode]=metrics;arrays[(seed,ns,s,mode)]=arr
                current=rows[str(seed)][key]
                assert current['BN_S0']['y_sha256']==current['BN_S2']['y_sha256']
                assert len({r['standard_noise_sha256'] for r in current.values()})==1
                print('CELL',seed,ns,s,{m:round(v['mean_nmse_db'],5) for m,v in current.items()},'seconds',round(time.perf_counter()-start,1),flush=True)
                dump('progress.json',dict(stage='evaluation',seed=seed,ns=ns,snr=s,completed_cells=seed*9+LENGTHS.index(ns)*3+SNRS.index(s)+1,total_cells=27,seconds=time.perf_counter()-start))
                dump('partial_results.json',dict(results=rows,powers=powers))
                audit.append(dict(seed=seed,ns=ns,snr=s,decoder_stats_only_changed=changed,matching_BN_y=True,matching_standard_noise=True))
        assert tensor_hash(bn)==bn_initial and tensor_hash(gn)==gn_initial
        audit.append(dict(seed=seed,gn_groups={str(k):v for k,v in groups.items()},source_state_unchanged=True,forward_equivalence=checks))
    differences=contrasts(arrays);summary=summarize(rows,differences)
    dump('results.json',dict(results=rows,power=powers,summary=summary,contrasts=differences))
    flat={f's{seed}_ns{ns}_snr{s}_{mode}_{metric}':v for (seed,ns,s,mode),values in arrays.items() for metric,v in values.items()}
    np.savez_compressed(OUT/'observations.npz',**flat)
    assert hashes=={str(p):sha(p) for p in protected},'Original file changed.'
    dump('audit.json',dict(status='PASS',original_files_unchanged=True,checks=audit,seconds=time.perf_counter()-start))
    dump('progress.json',dict(stage='complete',seconds=time.perf_counter()-start))
    print('COMPLETE',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
