"""Matched single-seed GN sharing-cost experiment; fixed Ns=64.

Only this new directory is writable. Source models and historical results stay intact.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[2] / 'simulation'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from train_rdae_gn import LinkRDAE_GN
from train_rdae import load01
from model_v4 import normalize_power, to_symbols, from_symbols, qam_map_ste, tx_power
from quality_v5 import sgcs_per

SNRS = (0, 10, 20)
MODELS = ('shared', 'specialist_0', 'specialist_10', 'specialist_20')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def state_hash(net):
    d = hashlib.sha256()
    for k, v in net.state_dict().items():
        d.update(k.encode()); d.update(v.detach().cpu().numpy().tobytes())
    return d.hexdigest()

def dump(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)

def gen(seed, offset):
    return torch.Generator().manual_seed(seed + offset)

def forward(net, x, snr, eps, power=None):
    feat = (10 ** (-snr / 20)).view(-1, 1)
    z = normalize_power(net.enc(x, feat), 64)
    symbols = qam_map_ste(to_symbols(z))
    actual = tx_power(symbols, 64)
    reference = actual.detach().clamp_min(1e-6) if power is None else torch.full_like(actual, float(power))
    sigma = (reference / 2).sqrt() * 10 ** (-snr / 20)
    y = from_symbols(symbols + eps * sigma[:, None, None])
    return net.dec(y, feat), actual

def validate_forward(net, x):
    """Explicit paired-noise path must reproduce the original complete link."""
    snr = torch.linspace(0, 20, len(x))
    seed = 81721
    torch.manual_seed(seed)
    with torch.no_grad():
        original = net(x, snr, 64)
        eps = torch.randn((len(x), 64, 2), generator=torch.Generator().manual_seed(seed))
        explicit, _ = forward(net, x, snr, eps)
    delta = float((original-explicit).abs().max())
    assert delta < 1e-5, delta
    assert sum(isinstance(m, nn.GroupNorm) for m in net.modules()) == 18
    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    return {'max_absolute_forward_difference': delta, 'GN_layers': 18, 'BN_layers': 0}

def train_one(name, args, xtr, initial):
    folder = OUT / 'checkpoints'
    final = folder / f'{name}.pt'
    if final.exists():
        saved = torch.load(final, map_location='cpu', weights_only=True)
        assert saved['config'] == vars(args)
        print('ALREADY_COMPLETE', name, flush=True)
        net = LinkRDAE_GN(cond=False)
        net.load_state_dict(saved['state_dict'])
        return net.eval()
    net = LinkRDAE_GN(cond=False)
    net.load_state_dict(initial)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    initial_hash = state_hash(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ordergen, snrgen, noisegen = (gen(args.seed, k) for k in (31000, 32000, 33000))
    steps = len(xtr)//args.bs
    trace = []
    first_epoch = 0
    previous_elapsed = 0.
    resume = folder / f'{name}_resume.pt'
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=True)
        assert ck['config'] == vars(args)
        net.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        for g, key in ((ordergen, 'order_rng'), (snrgen, 'snr_rng'), (noisegen, 'noise_rng')):
            g.set_state(ck[key])
        trace = ck['trace']; first_epoch = ck['epoch']; previous_elapsed = ck['elapsed_seconds']
    start = time.perf_counter()
    for ep in range(first_epoch, args.epochs):
        net.train()
        permutation = torch.randperm(len(xtr), generator=ordergen)
        total = 0.
        for step in range(steps):
            xb = xtr[permutation[step*args.bs:(step+1)*args.bs]]
            drawn_snr = 20 * torch.rand(len(xb), generator=snrgen)
            snr = drawn_snr if name == 'shared' else torch.full_like(drawn_snr, float(name.split('_')[1]))
            eps = torch.randn((len(xb),64,2), generator=noisegen)
            prediction, _ = forward(net, xb, snr, eps)
            loss = (prediction-xb).square().mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        elapsed = previous_elapsed + time.perf_counter()-start
        row = {'epoch': ep+1, 'train_mse': total/steps, 'elapsed_seconds': elapsed}
        trace.append(row)
        dump(f'trace_{name}.json', trace)
        dump('progress.json', {'stage':'training','model':name,'model_index':MODELS.index(name)+1,
                              'models_total':len(MODELS),'epoch':ep+1,'epochs':args.epochs,
                              'elapsed_seconds_model':elapsed,'last_train_mse':total/steps})
        if ep == 0 or (ep+1)%5 == 0:
            print('TRAIN',name,'epoch',ep+1,'mse',round(total/steps,6),'seconds',round(elapsed,1),flush=True)
        if (ep+1)%5 == 0:
            torch.save({'config':vars(args),'state_dict':net.state_dict(),'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),'epoch':ep+1,'trace':trace,'elapsed_seconds':elapsed,
                        'order_rng':ordergen.get_state(),'snr_rng':snrgen.get_state(),'noise_rng':noisegen.get_state()}, resume)
    net.eval()
    torch.save({'config':vars(args),'state_dict':net.state_dict(),'initial_state_sha256':initial_hash,
                'trace':trace,'selection':'fixed final epoch; no test/val checkpoint selection'}, final)
    print('COMPLETE',name,'seconds',round(trace[-1]['elapsed_seconds'],1),flush=True)
    return net

@torch.no_grad()
def calibrate_power(net, xtr, snr):
    net.eval()
    powers=[]
    for xb in xtr.split(128):
        feat=torch.full((len(xb),1),10**(-snr/20))
        symbols=qam_map_ste(to_symbols(normalize_power(net.enc(xb,feat),64)))
        powers.append(tx_power(symbols,64))
    return float(torch.cat(powers).mean())

@torch.no_grad()
def observe(net, x, snr, power, repeats):
    net.eval(); ratios=[]; scores=[]; eps_hash=hashlib.sha256()
    for repeat in range(repeats):
        g=gen(0, 50000+100*snr+repeat)
        r=[]; sc=[]
        for xb in x.split(128):
            eps=torch.randn((len(xb),64,2),generator=g)
            eps_hash.update(eps.numpy().tobytes())
            ss=torch.full((len(xb),),float(snr))
            pred,_=forward(net,xb,ss,eps,power)
            r.append(((pred-xb).square().sum((1,2,3))/xb.square().sum((1,2,3))).numpy())
            sc.append(sgcs_per(pred,xb).numpy())
        ratios.append(np.concatenate(r));scores.append(np.concatenate(sc))
    ratio=np.array(ratios); sgcs=np.array(scores); db=10*np.log10(ratio+1e-12)
    return {'mean_nmse_db':float(db.mean()),'db_mean_linear_nmse':float(10*np.log10(ratio.mean())),
            'sgcs':float(sgcs.mean()),'standard_noise_sha256':eps_hash.hexdigest()}, {'db':db,'sgcs':sgcs,'ratio':ratio}

def bootstrap(differences,seed=321):
    per_h=differences.mean(0)
    rng=np.random.default_rng(seed)
    b=per_h[rng.integers(0,len(per_h),size=(3000,len(per_h)))].mean(1)
    return {'mean':float(per_h.mean()),'H_cluster_CI95':np.quantile(b,[.025,.975]).tolist(),
            'uncertainty':'test H resampling only; not retraining-seed uncertainty'}

def evaluate(args,xtr,xval,xtest):
    rows={}; arrays={}
    for name in MODELS:
        ck=torch.load(OUT/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
        net=LinkRDAE_GN(cond=False);net.load_state_dict(ck['state_dict']);net.eval()
        rows[name]={}
        for snr in SNRS:
            power=calibrate_power(net,xtr,snr)
            row, raw=observe(net,xtest,snr,power,args.repeats)
            val, _=observe(net,xval,snr,power,1)
            row['train_power_reference']=power;row['validation']=val
            rows[name][str(snr)]=row
            for key,v in raw.items():arrays[f'{name}__snr{snr}__{key}']=v
            print('EVAL',name,snr,'nmse',round(row['mean_nmse_db'],5),'sgcs',round(row['sgcs'],5),flush=True)
        dump('partial_results.json',rows)
    gaps={}
    for s in SNRS:
        a=f'shared__snr{s}__';b=f'specialist_{s}__snr{s}__'
        assert rows['shared'][str(s)]['standard_noise_sha256']==rows[f'specialist_{s}'][str(s)]['standard_noise_sha256']
        gaps[str(s)]={'nmse_shared_minus_specialist':bootstrap(arrays[a+'db']-arrays[b+'db']),
                      'sgcs_specialist_minus_shared':bootstrap(arrays[b+'sgcs']-arrays[a+'sgcs'])}
    result={'config':vars(args),'scope':'fixedNs64,pure_per_sample_SNR_shared,GN,one_training_seed',
            'results':rows,'gaps':gaps,'gap_direction':'positive means shared is worse',
            'limits':['one seed','synthetic CSI and AWGN','historically reused test H',
                      'GN group setting fixed; no hyperparameter search','not official full RD-JSCC',
                      'not a multi-length sharing-cost experiment','not a matched BN-vs-GN sharing-cost comparison']}
    np.savez_compressed(OUT/'observations.npz',**arrays)
    dump('results.json',result)
    write_report(result)
    return result

def write_report(result):
    lines=['# GN 共享代价：固定 Ns=64 的单种子对照','',
           '## 问题与协议','',
           '检验去除 BN 冻结统计量后，跨 SNR 共享与专用是否仍有性能差距。固定 Ns=64、16-QAM、AWGN，RD-AE 改编骨干的全部 18 层 BN 换为现有分组规则的 GN。',
           '四个模型同初始化、同 H 顺序、同标准噪声、同 Adam 与 40 轮。共享模型每个样本独立抽 U(0,20) dB；三个专用模型分别固定 0/10/20 dB。最终轮次预先固定，不按 val/test 选择。',
           '训练集 6000H 分别标定每个模型在各 SNR 的实际发射功率；test1000H×3组相同标准噪声。正差距表示共享较差。','',
           '## 主结果','',
           '| SNR | GN共享 NMSE/dB | GN专用 NMSE/dB | 共享−专用/dB | 共享 SGCS | 专用 SGCS |',
           '|---|---|---|---|---|---|']
    for s in SNRS:
        a=result['results']['shared'][str(s)];b=result['results'][f'specialist_{s}'][str(s)]
        gap=result['gaps'][str(s)]['nmse_shared_minus_specialist']['mean']
        lines.append(f"| {s} | {a['mean_nmse_db']:.4f} | {b['mean_nmse_db']:.4f} | {gap:+.4f} | {a['sgcs']:.4f} | {b['sgcs']:.4f} |")
    lines+=['','## 不确定性','',
            '下列区间按测试 H 聚类重采样，同一个 H 的三次噪声一起抽样。只有一个训练种子，不能当作训练重复的置信区间。','']
    for s in SNRS:
        d=result['gaps'][str(s)]['nmse_shared_minus_specialist'];lo,hi=d['H_cluster_CI95']
        lines.append(f"- {s} dB：共享减专用 {d['mean']:+.4f} dB，H区间 [{lo:+.4f}, {hi:+.4f}]。")
    lines+=['','## 结果怎样解释','',
            '只在实际差距为正的条件下说“当前 GN 共享模型较专用差”；负值如实写共享更好。即使有差距，也只能排除 BN 冻结统计量是其必要条件，不能把余量归因于容量或不可避免的共享代价。',
            '与历史 BN 数字的训练 SNR 协议不同，不做 BN→GN 共享代价下降量的因果比较。此次不测多长度共享、吞吐或多种子稳定性。','',
            '## 文件','',
            '- run_gn_gap.py：训练、评估与报告代码。',
            '- protocol.json：事先固定协议与源文件哈希。',
            '- checkpoints/：四个最终模型及可恢复训练状态。',
            '- trace_*.json、run.log、progress.json：训练过程。',
            '- results.json、observations.npz：均值、配对区间与逐H/噪声结果。',
            '- audit.json：原链路一致性、相同初始化/输入/噪声和原文件未改动核对。']
    (OUT/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--epochs',type=int,default=40)
    pa.add_argument('--seed',type=int,default=0);pa.add_argument('--threads',type=int,default=8)
    pa.add_argument('--bs',type=int,default=128);pa.add_argument('--repeats',type=int,default=3)
    pa.add_argument('--evaluate-only',action='store_true');args=pa.parse_args()
    torch.set_num_threads(args.threads)
    (OUT/'checkpoints').mkdir(exist_ok=True)
    xtr,xval,xtest=(load01(s,root=str(PROJECT/'data')) for s in ('train','val','test'))
    files=[PROJECT/n for n in ('model_rdae.py','model_v4.py','train_rdae.py','train_rdae_gn.py','quality_v5.py')]
    files += [PROJECT/'data'/f'{s}.npz' for s in ('train','val','test')]
    hashes={str(f):sha(f) for f in files}
    torch.manual_seed(args.seed);net=LinkRDAE_GN(cond=False)
    initial=copy.deepcopy(net.state_dict())
    checks=validate_forward(net,xtr[:8])
    protocol={'args':vars(args),'models':MODELS,'SNR_grid':SNRS,'ns':64,
              'shared_training':'pure per-sample continuous uniform(0,20) dB',
              'specialists':'fixed 0/10/20 dB; same exact GN architecture as shared',
              'training':'all four identical init, H order, standard Gaussian draws; Adam2e-3 wd1e-5 cosine40; final40',
              'epochs_per_model':args.epochs,'batch_size':args.bs,'batches_per_epoch':len(xtr)//args.bs,
              'data_shapes':{'train':list(xtr.shape),'val':list(xval.shape),'test':list(xtest.shape)},
              'evaluation':'train-only actual-Ptx calibration; test1000H x3pairedGaussian; mean per-H NMSEdB and squaredSGCS',
              'initial_state_sha256':state_hash(net),'source_sha256':hashes,'script_sha256':sha(__file__),
              'scope':'single seed; fixed length; no official VQ/diffusion reproduction; reused test H'}
    if not args.evaluate_only:
        old=OUT/'protocol.json'
        if old.exists():
            previous=json.loads(old.read_text(encoding='utf-8'))
            assert previous['args']==vars(args) and previous['source_sha256']==hashes
        else:dump('protocol.json',protocol)
        started=time.perf_counter()
        for name in MODELS:train_one(name,args,xtr,initial)
        print('ALL_TRAINING_COMPLETE',round(time.perf_counter()-started,1),flush=True)
    dump('progress.json',{'stage':'evaluation','models_complete':4})
    result=evaluate(args,xtr,xval,xtest)
    assert all(sha(f)==v for f,v in hashes.items())
    init_hashes=[]
    for name in MODELS:
        ck=torch.load(OUT/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
        init_hashes.append(ck['initial_state_sha256'])
    assert len(set(init_hashes))==1
    dump('audit.json',{'forward_equivalence':checks,'identical_initial_state_sha256':init_hashes[0],
                       'source_hashes_unchanged':True,'paired_eval_noise_hashes_verified':True,
                       'all_results_finite':all(np.isfinite(result['results'][m][str(s)]['mean_nmse_db']) for m in MODELS for s in SNRS)})
    dump('progress.json',{'stage':'complete','results':'results.json'})
    print('FINAL_GAPS',json.dumps(result['gaps']),flush=True)

if __name__=='__main__':main()
