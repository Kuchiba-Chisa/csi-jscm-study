"""All prespecified arms, seed variation and paired diagnostics; no winner selection."""
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parent
rows=[json.loads((OUT/f'results_s{s}.json').read_text(encoding='utf8')) for s in (0,1,2)]
names=list(rows[0]['policies'])
labels={
 'S0_old_monitor_old_refiner':'原接收机＋旧模块',
 'A_old_modules_old_gate':'适配＋旧模块/旧门限',
 'A_rethreshold_old_modules':'适配＋只重设门限',
 'A_affine_old_modules':'适配＋分数仿射校准',
 'A_updated_monitor_old_refiner':'适配＋更新监测器/旧精修器',
 'A_updated_monitor_moment_refiner':'适配＋更新监测器/精修输入重统计',
 'A_updated_monitor_updated_refiner':'适配＋更新监测器/更新精修器'}

def mean(xs):return float(np.mean(xs))
def ms(xs,d=3):return f'{np.mean(xs):.{d}f} ± {np.std(xs,ddof=1):.{d}f}'
def vec(path):
    v=[]
    for row in rows:
        q=row
        for k in path:q=q[k]
        v.append(q)
    return v

summary=dict(base={},quality={},policies={},refiners={},cost={})
lines=['# 接收机适配—质量监测—按需精修：联合验证',
 '', '这轮固定问题：接收机BN适配后，旧监测器/精修器能否直接沿用；若不能，门限校准、输入重统计或小模块离线更新各能恢复多少。',
 '', '不是官方RD-JSCC端到端复现。冻结RD-AE改编链路，16-QAM、AWGN、Ns=64，三个既有源模型种子。精修器只在此长度训练，因此本轮不扩大到Ns16/32。',
 '', '## 口径与数据',
 '', '- 接收端适配：复用Q1已有0/20dB BN统计量，10dB线性插值；不更新编解码权重、不重新标定发射功率。',
 '- 所有策略复用完全相同y；坏门限固定为原接收机在前250验证H各SNR的NMSE80%分位数。适配后不重新定义坏样本。',
 '- 模块更新：2000训练H，每H随机分配一个SNR；前250验证H选择轮次/拟合仿射，后250H确定20%触发门限。',
 '- 确认评估：testH600:1000，每SNR三组新噪声。H在此前项目中使用过，不能称整个项目全新独立测试集。',
 '- 主表使用验证集固定门限；随机对照匹配实际计算MAC，包括监测器成本。另报告离线池精确20%选择，分离门限漂移与排序能力。',
 '- ±为三种子样本标准差；配对95%区间按H聚类，重复噪声不是独立模型种子。多项比较属于探索性评估。',
 '', '## 1. 接收机适配本身', '',
 '| SNR | 原始NMSE/dB | 适配NMSE/dB | 改善/dB | 原始坏比例 | 适配坏比例 |',
 '|---|---|---|---|---|---|']
for s in ('0','10','20'):
    c=[r['base_comparison'][s] for r in rows]
    summary['base'][s]={k:mean([x[k] for x in c]) for k in ('S0_nmse_db','A_nmse_db','gain_db','S0_bad_rate','A_bad_rate')}
    lines.append(f"| {s} | {ms([x['S0_nmse_db'] for x in c])} | {ms([x['A_nmse_db'] for x in c])} | {ms([x['gain_db'] for x in c])} | {100*mean([x['S0_bad_rate'] for x in c]):.2f}% | {100*mean([x['A_bad_rate'] for x in c]):.2f}% |")
lines += ['', 'Ns64是此前Q1中改善较小的部署长度；不能用这张表替代Ns16/32已有结果。',
 '', '## 2. 监测器迁移：预测校准与排序分开看', '',
 '| SNR | 监测器状态 | MAE/dB | 偏差/dB | AUROC（共同固定坏门限） | Spearman |',
 '|---|---|---|---|---|---|']
for s in ('0','10','20'):
    summary['quality'][s]={}
    for state in ('S0','A_old','A_affine','A_updated'):
        c=[r['quality'][state][s] for r in rows]
        summary['quality'][s][state]={k:mean([x[k] for x in c]) for k in ('mae_db','bias_db','auroc','spearman')}
        lines.append(f"| {s} | {state} | {ms([x['mae_db'] for x in c])} | {ms([x['bias_db'] for x in c])} | {ms([x['auroc'] for x in c])} | {ms([x['spearman'] for x in c])} |")
lines += ['', '正斜率仿射不会改变同条件排序；仿射后重设20%门限与直接重设门限动作相同，不能将其称为排序恢复。',
 '', '## 3. 运行时固定门限下的完整策略', '',
 '| SNR | 策略 | 触发比例 | 最终NMSE/dB | 坏比例 | 同MAC随机坏比例 | 比随机少/百分点 | 支持坏比例下降的种子数* |',
 '|---|---|---|---|---|---|---|---|']
for s in ('0','10','20'):
    summary['policies'][s]={}
    for name in names:
        c=[r['policies'][name][s] for r in rows]
        support=sum(x['bad_rate_minus_random_H_CI95'][1]<0 for x in c)
        summary['policies'][s][name]={k:mean([x[k] for x in c]) for k in ('refiner_call_rate','final_nmse_db','final_bad_rate','random_same_MAC_bad_rate','gain_vs_random_db','bad_rate_minus_random','additional_MACs','final_sgcs')}
        summary['policies'][s][name]['support_bad_rate_seeds']=support
        lines.append(f"| {s} | {labels[name]} | {100*mean([x['refiner_call_rate'] for x in c]):.2f}% | {ms([x['final_nmse_db'] for x in c])} | {100*mean([x['final_bad_rate'] for x in c]):.2f}% | {100*mean([x['random_same_MAC_bad_rate'] for x in c]):.2f}% | {-100*mean([x['bad_rate_minus_random'] for x in c]):+.2f} | {support}/3 |")
lines += ['', '*支持指该种子的H聚类配对95%区间完全低于0，不代表已做多重比较校正。随机基线随精修器和实际触发率变化，不能跨行只看“比随机少”。',
 '', '## 4. 精确20%离线选择诊断', '',
 '| SNR | 策略 | 最终NMSE/dB | 坏比例 | 相对随机NMSE收益/dB | 比随机少/百分点 |',
 '|---|---|---|---|---|---|']
for s in ('0','10','20'):
    for name in names:
        if name=='A_affine_old_modules':continue
        c=[r['exact20_diagnostic'][name][s] for r in rows]
        lines.append(f"| {s} | {labels[name]} | {mean([x['final_nmse_db'] for x in c]):.4f} | {100*mean([x['final_bad_rate'] for x in c]):.2f}% | {mean([x['gain_vs_random_db'] for x in c]):+.4f} | {-100*mean([x['bad_rate_minus_random'] for x in c]):+.2f} |")
lines += ['', '这张表按评估池排序选择20%，只诊断排序，不能当作已实现在线调度。',
 '', '## 5. 精修器本身是否失配（全部启用的诊断）', '',
 '| SNR | 精修器状态 | 相对各自基础重建的改善/dB | 变差样本比例 |',
 '|---|---|---|---|']
for s in ('0','10','20'):
    summary['refiners'][s]={}
    for name in ('S0_old','A_old','A_moment','A_updated'):
        c=[r['refiner_comparison'][s][name] for r in rows]
        summary['refiners'][s][name]={k:mean([x[k] for x in c]) for k in ('always_refine_gain_db','harmed_fraction')}
        lines.append(f"| {s} | {name} | {ms([x['always_refine_gain_db'] for x in c])} | {100*mean([x['harmed_fraction'] for x in c]):.2f}% |")
lines += ['', '基础接收机输出不同，相对改善变小本身不足以证明精修器失配；还需结合适配后更新精修器的收益及验证选择。',
 '', '## 6. 维护成本与边界', '',
 '| 维护方式 | 更新量 | 是否需标签 | 实际CPU更新时间（三种子均值） |',
 '|---|---|---|---|',
 '| 重设门限 | 每SNR一个门限 | 否，仅需验证分数；不保证质量校准 | 无梯度 |',
 '| 分数仿射校准 | 6个标量 | 是，250验证H的真值误差 | 闭式拟合，无梯度 |',
 '| 精修输入重统计 | 36个标量 | 否，仅需接收特征 | 前向统计，无梯度 |']
for key,name in [('monitor_update','监测器离线更新'),('refiner_update','精修器离线更新')]:
    c=[r[key] for r in rows]
    summary['cost'][key]=dict(parameters=c[0]['parameters'],mean_seconds=mean([x['seconds'] for x in c]),selected_epochs=[x['selected_epoch'] for x in c])
    lines.append(f"| {name} | {c[0]['parameters']}参数 | 是，2000训练H＋250验证H | {mean([x['seconds'] for x in c]):.2f}s，选中轮次{[x['selected_epoch'] for x in c]} |")
lines += ['', '- 时间仅为小模块优化，不含获取标签、生成缓存、部署通信和标定锚点；总实验时间另记。',
 '- 标签来自本地有真值H的离线数据；不能假设实际BS在线免费获得真值。',
 '- 更新后的部署只需保留选定的一套监测器/精修器；评估保存多套版本是为了比较。',
 '- 未验证新场景、未知反馈长度、估计SNR误差、真实业务门限或吞吐收益；不声称硬件PoC。',
 '- 所有预设方案均报告，不按测试结果挑选方法或种子。完整逐种子区间、SGCS和MAC见results_s*.json。',
 '', '## 复现', '',
 '先阅读README.md。脚本依赖本项目旧实验模块与既有权重，归档不是独立可运行的官方复现包。',
 '', '```powershell',
 "& 'simulation\\.venv\\Scripts\\python.exe' -u work/joint_receiver_maintenance_20261113/run_joint.py --seeds 0 1 2 --threads 4",
 "& 'simulation\\.venv\\Scripts\\python.exe' work/joint_receiver_maintenance_20261113/audit_joint.py",
 '```']
(OUT/'联合验证结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
