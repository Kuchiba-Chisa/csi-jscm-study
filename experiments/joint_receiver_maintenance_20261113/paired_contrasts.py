"""Direct paired comparisons among prespecified policies; no additional fitting."""
from pathlib import Path
import json
import numpy as np
import run_joint as r

OUT=Path(__file__).resolve().parent
contrasts={
 'same_gate_update_refiner':('A_updated_monitor_updated_refiner','A_updated_monitor_old_refiner','policies'),
 'equal20_update_monitor':('A_updated_monitor_old_refiner','A_rethreshold_old_modules','exact20_diagnostic'),
 'equal20_full_vs_original':('A_updated_monitor_updated_refiner','S0_old_monitor_old_refiner','exact20_diagnostic'),
 'equal20_full_vs_adapted_base':('A_updated_monitor_updated_refiner','A_base','exact20_diagnostic')}

def losses(z,row,name,mode,s):
    ix=z['snr']==s
    if name=='A_base':return z['A_base'][ix]
    if name.startswith('S0'):
        base,ref,q='S0_base','S0_refined','q_S0'
    else:
        base='A_base'
        ref='A_updated_refined' if name.endswith('updated_refiner') else 'A_old_refined'
        q='q_A_updated' if 'updated_monitor' in name else 'q_A_old'
    if mode=='policies':call=z[name+'_call_'+str(s)]
    else:
        call=np.zeros(ix.sum(),dtype=bool)
        call[np.argsort(z[q][ix],kind='stable')[-round(.2*ix.sum()):]]=True
    return np.where(call,z[ref][ix],z[base][ix])

out={}
for seed in (0,1,2):
    row=json.loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
    z=np.load(OUT/f'outcomes_s{seed}.npz')
    for name,(a,b,mode) in contrasts.items():
        for s in r.SNRS:
            la,lb=losses(z,row,a,mode,s),losses(z,row,b,mode,s)
            h=z['h_index'][z['snr']==s]
            bad=row['bad_thresholds'][str(s)]
            improvement=lb-la
            risk=(la>bad).astype(float)-(lb>bad).astype(float)
            cell=dict(seed=seed,mean_nmse_improvement_db=float(improvement.mean()),
                mean_nmse_improvement_H_CI95=r.rf.cluster_ci(improvement,h,871000+seed*100+s),
                bad_rate_change=float(risk.mean()),bad_rate_change_H_CI95=r.rf.cluster_ci(risk,h,872000+seed*100+s))
            out.setdefault(name,{}).setdefault(str(s),[]).append(cell)
r.rf.dump(OUT/'paired_contrasts.json',out)
lines=['# 预设策略的配对差补充','','正的NMSE改善表示前一方案更好；负的坏比例变化表示前一方案更好。每个种子按同一H聚类，不重新训练或选择门限。','',
 '| 比较 | SNR | NMSE改善/dB（三种子均值） | 支持改善的种子数 | 坏比例变化/百分点 | 支持坏比例下降的种子数 |',
 '|---|---|---|---|---|---|']
for name,ss in out.items():
    for s,c in ss.items():
        lines.append(f"| {name} | {s} | {np.mean([x['mean_nmse_improvement_db'] for x in c]):+.4f} | {sum(x['mean_nmse_improvement_H_CI95'][0]>0 for x in c)}/3 | {100*np.mean([x['bad_rate_change'] for x in c]):+.3f} | {sum(x['bad_rate_change_H_CI95'][1]<0 for x in c)}/3 |")
lines+=['','same_gate_update_refiner：相同更新后监测器、相同门限与动作集合，只更换精修器权重。',
 'equal20_update_monitor：原精修器相同、精确20%选择，只比较监测排序变化。',
 'equal20_full_vs_original：两边均精确20%选择、相同监测器/精修器架构MAC，比较整套接收流程。',
 'equal20_full_vs_adapted_base：与不启用监测/精修的适配接收机比较；额外计算预算不同，此列用于检查精修是否真的值得启动。',
 '这些是探索性成组比较，未作多重比较校正。']
(OUT/'配对比较.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('\n'.join(lines))
