"""Compare similarly sized quality regressors on a FROZEN RD-AE.

All regressors use identical source CSI, observations, targets and early-stopping
data. Decoder features are deterministic functions of y and the receiver state;
they provide an inductive bias, not extra information or additional feedback.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import OUT, build, data, dump, nmse_db_per, seed_all, summarize_predictions


class Monitor(nn.Module):
    def __init__(self, dim, budget=25000):
        super().__init__()
        width = min(range(4, 257), key=lambda h: abs((dim*h + h*h + 3*h + 1) - budget))
        self.net = nn.Sequential(nn.Linear(dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def cache(net, x, snrs, lengths=(16,32,64), repeats=1, noise_seed=20000, bs=128):
    net.eval()
    buffers, rows = {}, {k: [] for k in ('y','feature','condition','target','snr','ns','h_index')}
    def input_hook(module, arguments):
        buffers['y'] = arguments[0].detach()
    def feature_hook(module, arguments, value):
        buffers['feature'] = F.adaptive_avg_pool2d(value.detach(), (2, 2)).flatten(1)
    handles = [net.dec.register_forward_pre_hook(input_hook),
               net.dec.blocks[-1].af2.register_forward_hook(feature_hook)]
    try:
        for ns in lengths:
            for snr in snrs:
                for repeat in range(repeats):
                    for i in range(0, len(x), bs):
                        seed_all(noise_seed + ns*100 + int(snr)*10000 + repeat*1000 + i)
                        xb = x[i:i+bs]
                        xhat = net(xb, float(snr), int(ns))
                        rows['y'].append(buffers['y'])
                        rows['feature'].append(buffers['feature'])
                        rows['condition'].append(torch.tensor([[float(snr)/20, ns/64]]).repeat(len(xb),1))
                        rows['target'].append(nmse_db_per(xhat, xb))
                        rows['snr'].append(torch.full((len(xb),), float(snr)))
                        rows['ns'].append(torch.full((len(xb),), ns))
                        rows['h_index'].append(torch.arange(i,i+len(xb)))
    finally:
        for handle in handles:
            handle.remove()
    return {k: torch.cat(v).cpu() for k,v in rows.items()}


def inputs(rows, mode):
    keys = {'y':['y','condition'], 'feature':['feature','condition'],
            'y_feature':['y','feature','condition']}[mode]
    return torch.cat([rows[k] for k in keys],1)


def fit(train, val, mode, seed, epochs=50, bs=512):
    seed_all(150000 + seed)
    X, V = inputs(train,mode), inputs(val,mode)
    mean, sd = X.mean(0), X.std(0).clamp_min(1e-4)
    X, V = (X-mean)/sd, (V-mean)/sd
    tm, ts = train['target'].mean(), train['target'].std().clamp_min(1e-4)
    Y, Yv = (train['target']-tm)/ts, (val['target']-tm)/ts
    model = Monitor(X.shape[1])
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    best, state, stale, selected = float('inf'), None, 0, 0
    trace = []
    generator = torch.Generator().manual_seed(160000 + seed)
    for ep in range(epochs):
        order = torch.randperm(len(X),generator=generator)
        model.train()
        for i in range(0,len(X),bs):
            idx = order[i:i+bs]
            loss = (model(X[idx])-Y[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            score = float((model(V)-Yv).square().mean())
        trace.append(score)
        if score < best-1e-5:
            best, state, stale, selected = score, copy.deepcopy(model.state_dict()), 0, ep+1
        else:
            stale += 1
            if stale >= 7:
                break
    model.load_state_dict(state); model.eval()
    bundle = {'state_dict':state,'mean':mean,'sd':sd,'target_mean':tm,'target_sd':ts,
              'input_dim':X.shape[1],'mode':mode}
    meta = {'parameters':sum(p.numel() for p in model.parameters()),'selected_epoch':selected,
            'val_mse_db2':best*float(ts**2),'val_trace_standardized_mse':trace}
    return model,bundle,meta


@torch.no_grad()
def predict(model,bundle,rows):
    x = (inputs(rows,bundle['mode'])-bundle['mean'])/bundle['sd']
    return (model(x)*bundle['target_sd']+bundle['target_mean']).numpy()


def metrics(rows,pred,val=None):
    result = {}
    for ns in sorted(rows['ns'].unique().tolist()):
        for snr in sorted(rows['snr'].unique().tolist()):
            mask = ((rows['ns']==ns)&(rows['snr']==snr)).numpy()
            target = rows['target'].numpy()[mask]
            cell = summarize_predictions(pred[mask], target)
            cell['n_observations'] = len(target)
            cell['n_unique_H'] = int(rows['h_index'][mask].unique().numel())
            cell['mean_nmse_db'] = float(target.mean())
            result[f'{int(snr)}dB_ns{int(ns)}'] = cell
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--ntrain',type=int,default=2000)
    parser.add_argument('--nval',type=int,default=400)
    parser.add_argument('--ntest',type=int,default=400)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--train-repeats',type=int,default=2)
    parser.add_argument('--test-repeats',type=int,default=3)
    parser.add_argument('--out',default='q2_results.json')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    xtr,xval,xte=data()
    result={'protocol':vars(args),'models':[],
            'feature':'AdaptiveAvgPool2d(2,2) of last residual block af2 output (16 channels, 64 floats)',
            'labels':'AUROC uses worst 20% within each test condition, an evaluation label, not a deployable threshold',
            'generalization':'5/15 dB unseen by monitor train/val, but covered by original backbone training',
            'seed_unit':'3 independent pretrained backbone seeds; one monitor initialization per backbone/mode'}
    for seed in args.seeds:
        started=time.perf_counter()
        net,source=build(seed,xtr)
        original={k:v.clone() for k,v in net.state_dict().items()}
        train=cache(net,xtr[:args.ntrain],(0,10,20),repeats=args.train_repeats,noise_seed=20000)
        print(f'Q2 seed={seed} training cache {len(train["target"])} observations',flush=True)
        val=cache(net,xval[:args.nval],(0,10,20),noise_seed=30000)
        test=cache(net,xte[:args.ntest],(0,5,10,15,20),repeats=args.test_repeats,noise_seed=40000)
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in original.items()),'Frozen backbone changed'
        row={'seed':seed,'source':source,'heads':{},'transfer':{}}
        learned={}
        for mode in ('y','feature','y_feature'):
            head,bundle,meta=fit(train,val,mode,seed,args.epochs)
            pred=predict(head,bundle,test)
            meta['test']=metrics(test,pred)
            row['heads'][mode]=meta
            learned[mode]=(head,bundle)
            torch.save(bundle,OUT/f'q2_head_s{seed}_{mode}.pt')
            print(f'Q2 seed={seed} {mode}: params={meta["parameters"]}, val_mse={meta["val_mse_db2"]:.4f}, '+
                  str({k:round(v['auroc'],3) for k,v in meta['test'].items() if k.startswith('20dB')}),flush=True)
        # Transfer is evaluated only if matching Q1 receiver states already exist.
        for ns in (16,32):
            state_path=OUT/'q1_states'/f's{seed}_ns{ns}_S2.pt'
            if not state_path.exists():
                continue
            net.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True))
            transferred=cache(net,xte[:args.ntest],(20,),lengths=(ns,),repeats=args.test_repeats,noise_seed=40000)
            selector=(test['snr']==20)&(test['ns']==ns)
            assert torch.equal(transferred['y'],test['y'][selector]),'Transmitter/noise changed during receiver calibration'
            key=f'20dB_ns{ns}'
            row['transfer'][key]={mode:metrics(transferred,predict(head,bundle,transferred))[key]
                                   for mode,(head,bundle) in learned.items()}
            net.load_state_dict(original)
        row['duration_s']=time.perf_counter()-started
        result['models'].append(row)
        dump(OUT/args.out,result)
        print(f'Q2 seed={seed} completed {row["duration_s"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
