"""Post-result matched S0 continuation control. No tuning or replacing main arms."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import run_joint as r

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'continuation_control'
OUT.mkdir(exist_ok=True)
r.OUT=OUT
protocol=dict(label='post-result supplementary control, not prespecified primary arm',
    task='Same data, initialization, optimizer, lr1e-4, max12epochs and earlystop as adapted refiner update, but receiver remains S0',
    args=dict(ntrain=2000,epochs=12),main_script_sha256=r.rf.sha(ROOT/'run_joint.py'))
r.rf.dump(OUT/'protocol.json',protocol)
tr,va,te=r.rf.data()
torch.set_num_threads(4)
cells=[]
for seed in (0,1,2):
    net,source=r.rf.build(seed,tr)
    net.requires_grad_(False)
    obs=r.observations(net,tr[:2000],seed,'train')
    train=r.decoded(net,obs,tr[:2000],seed,False)
    vobs=r.observations(net,va[:500],seed,'val')
    val=r.decoded(net,vobs,va[:500],seed,False)
    fit=r.rf.subset(val,vobs['h_index']<250)
    model,bundle=r.old_refiner(seed)
    updated,meta=r.update_refiner(model,bundle,train,fit,seed,12)
    r.rf.dump(OUT/f'locked_s{seed}.json',dict(seed=seed,meta=meta,source=source,locked_before_test_forward=True))
    del train
    stored=torch.load(ROOT/f'test_observations_s{seed}.pt',weights_only=True)['observations']
    test=r.decoded(net,stored,te[600:1000],seed,False)
    out=r.rf.evaluate(updated,test,bundle['norm'])
    z=np.load(ROOT/f'outcomes_s{seed}.npz')
    improvement=z['S0_refined']-out['loss'].numpy()
    adapted_improvement=z['A_old_refined']-z['A_updated_refined']
    np.savez_compressed(OUT/f'outcomes_s{seed}.npz',refined_loss=out['loss'].numpy(),improvement=improvement)
    row=dict(seed=seed,meta=meta,cells={})
    for s in r.SNRS:
        ix=z['snr']==s;h=z['h_index'][ix]
        difference=adapted_improvement[ix]-improvement[ix]
        row['cells'][str(s)]=dict(original_continuation_improvement_db=float(improvement[ix].mean()),
            adapted_continuation_improvement_db=float(adapted_improvement[ix].mean()),
            adaptation_extra_update_benefit_db=float(difference.mean()),
            adaptation_extra_update_benefit_H_CI95=r.rf.cluster_ci(difference,h,881000+seed*100+s))
    r.rf.dump(OUT/f'results_s{seed}.json',row)
    cells.append(row)
    print('CONTROL DONE',seed,meta['selected_epoch'],flush=True)
lines=['# 事后补充：相同预算继续训练对照','','此对照在看到主实验结果后补充；不改变原方案、门限、测试H或主实验结果，不按结果调参。',
    '原接收机精修器与适配后精修器从同一旧权重开始，使用同一训练H/SNR/噪声、12轮上限、学习率和各自验证数据选轮次。',
    '表中是全部启用时，新旧精修器损失之差，越大表示更新收益越大；额外维护收益是适配接收机更新收益减去原接收机继续训练收益。','',
    '| SNR | 原接收机继续训练收益/dB | 适配后更新收益/dB | 差值/dB | 支持差值为正的种子数 |',
    '|---|---|---|---|---|']
for s in ('0','10','20'):
    cc=[row['cells'][s] for row in cells]
    lines.append(f"| {s} | {np.mean([v['original_continuation_improvement_db'] for v in cc]):+.4f} | {np.mean([v['adapted_continuation_improvement_db'] for v in cc]):+.4f} | {np.mean([v['adaptation_extra_update_benefit_db'] for v in cc]):+.4f} | {sum(v['adaptation_extra_update_benefit_H_CI95'][0]>0 for v in cc)}/3 |")
(ROOT/'继续训练对照.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
r.rf.dump(OUT/'summary.json',dict(protocol=protocol,models=cells))
