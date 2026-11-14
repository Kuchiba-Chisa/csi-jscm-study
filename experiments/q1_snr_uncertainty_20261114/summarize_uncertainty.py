"""Report all locked methods, all source seeds and both causal interventions."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
rows=[json.loads((ROOT/f'results_s{s}.json').read_text(encoding='utf8')) for s in (0,1,2)]
MODES=('bn_only','all_receiver')
LENGTHS=(16,32,64)
ERRORS=(-3,-1,0,1,3)

def mean(a):return float(np.mean(a))
def ms(a):return f'{np.mean(a):+.4f} ± {np.std(a,ddof=1):.4f}'
def cells(row,mode,ns=None,errors=None,interior=True):
    return [c for c in row['cells'] if c['mode']==mode and (ns is None or c['ns']==ns)
        and (errors is None or c['error'] in errors)
        and (not interior or c['true_snr'] in (5,10,15))]

summary=dict(selection={},aggregate={},aggregate_all={},by_bias={},worst_points={},accurate_reference={},estimation_penalty={},risk_attribution={})
lines=['# Q1补充：SNR信息不准确时，条件适配是否仍值得部署？','',
 '本轮复用RD-AE改编模型与已有0/20dB BN锚点；不重新训练编码器、解码器或星座，不改真实发射功率和物理噪声。不是官方完整RD-JSCC复现。','',
 '## 实验设计','',
 '- 三个既有源模型种子；Ns16/32/64；真实测试SNR0/5/10/15/20dB；BS估计偏差−3/−1/0/+1/+3dB。',
 '- `bn_only`：只让BN统计量选择使用错误SNR，接收端AF仍用准确SNR。',
 '- `all_receiver`：BN选择及接收端AF都用估计SNR。UE编码器始终用准确SNR；本轮不测双侧条件同时失配。',
 '- 各组使用同一真实含噪y。估计值截到[0,20]dB，不外推锚点；主表只汇总5/10/15dB，边界点另报。',
 '- S0基线使用与相应模式相同的接收端AF信息，避免把AF误差归因给BN。准确条件适配单列为参照。',
 '- 保守适配在原始BN矩与条件BN矩间收缩；方差包含均值之间的混合项，不只平均方差。',
 '- 每种子、已知Ns、模式选一个固定α，候选0/.25/.5/.75/1；只用前250验证H、SNR0/10/20、偏差−3/0/+3，最小化相对同输入S0的最坏验证部署点平均损失差。α=0即不适配，允许被选中。',
 '- 选择锁定后才生成testH600:1000的三组新噪声；5/15dB和±1dB未用于选α，但验证/测试来自同一信源总体，H在历史项目中使用过。',
 '- 固定偏差是敏感性干预，不是测得的估计误差分布。±是三个模型种子的样本标准差；同H多噪声不当独立种子。','',
 '## 1. 测试前锁定的适配强度','',
 '| Ns | 模式 | seed0 | seed1 | seed2 |','|---|---|---|---|---|']
for ns in LENGTHS:
    for mode in MODES:
        a=[r['selection'][f'{ns}_{mode}']['alpha'] for r in rows]
        summary['selection'][f'{ns}_{mode}']=a
        lines.append(f'| {ns} | {mode} | {a[0]} | {a[1]} | {a[2]} |')
lines += ['', 'α=1为完整条件适配；α=0回退原始BN。不同模式允许分别配置，但不能在测试时按真值或损失挑α。','',
 '## 2. 内部SNR点的总体收益及负迁移','',
 '每个种子×模式×Ns含15个部署点：真实SNR5/10/15×五种偏差。正收益表示优于同输入S0。','',
 '| Ns | 模式 | 方法 | 平均NMSE改善/dB | 负收益点数（三种子共45） | 95%配对区间支持负收益点数 | 最坏部署点收益/dB* |','|---|---|---|---|---|---|---|']
for ns in LENGTHS:
    for mode in MODES:
        key=f'{ns}_{mode}';summary['aggregate'][key]={}
        for method in ('vanilla','conservative'):
            cc=[cells(r,mode,ns) for r in rows]
            gain=[[c['methods'][method]['gain_vs_input_matched_S0_db'] for c in rr] for rr in cc]
            bad=sum(v<0 for rr in gain for v in rr)
            sigbad=sum(c['methods'][method]['gain_vs_input_matched_S0_H_CI95'][1]<0 for rr in cc for c in rr)
            minbyseed=[min(rr) for rr in gain]
            pooled=[c for rr in cc for c in rr]
            worst=min(pooled,key=lambda c:c['methods'][method]['gain_vs_input_matched_S0_db'])
            item=dict(mean_gain_db=mean([mean(rr) for rr in gain]),seed_mean_gains=[mean(rr) for rr in gain],
                negative_cells=bad,significant_negative_cells=sigbad,total_cells=45,
                mean_worst_cell_gain_db=mean(minbyseed),worst_observed_cell_gain_db=min(minbyseed),
                mean_sgcs=mean([c['methods'][method]['sgcs'] for c in pooled]),
                mean_gain_CI_scope='source-seed SD and individual H-cluster CIs; no pooled independence assumption')
            summary['aggregate'][key][method]=item
            lines.append(f"| {ns} | {mode} | {method} | {ms([mean(rr) for rr in gain])} | {bad}/45 | {sigbad}/45 | {min(minbyseed):+.4f} |")
        # Keep biggest observed losses as diagnostics, not as tuning targets.
        flat=[dict(seed=r['seed'],**c) for r in rows for c in cells(r,mode,ns)]
        summary['worst_points'][key]=sorted(flat,key=lambda c:c['methods']['vanilla']['gain_vs_input_matched_S0_db'])[:3]
lines += ['', '*最坏列为所有种子×内部部署点中的最小平均收益，是观测诊断，不是数学最坏情形保证。负收益计数对极小数值也计数，须结合配对区间判断。','',
 '### 全区间部署网格（包括截断边界）','',
 '| Ns | 模式 | 方法 | 平均改善/dB | 负收益点数（三种子共75） | 95%区间支持负收益点数 | 最坏收益/dB |','|---|---|---|---|---|---|---|']
for ns in LENGTHS:
    for mode in MODES:
        key=f'{ns}_{mode}';summary['aggregate_all'][key]={}
        for method in ('vanilla','conservative'):
            cc=[cells(r,mode,ns,interior=False) for r in rows]
            gains=[[c['methods'][method]['gain_vs_input_matched_S0_db'] for c in rr] for rr in cc]
            bad=sum(v<0 for rr in gains for v in rr)
            sigbad=sum(c['methods'][method]['gain_vs_input_matched_S0_H_CI95'][1]<0 for rr in cc for c in rr)
            mins=[min(rr) for rr in gains]
            item=dict(mean_gain_db=mean([mean(rr) for rr in gains]),seed_mean_gains=[mean(rr) for rr in gains],
                negative_cells=bad,significant_negative_cells=sigbad,total_cells=75,worst_observed_cell_gain_db=min(mins))
            summary['aggregate_all'][key][method]=item
            lines.append(f"| {ns} | {mode} | {method} | {ms([mean(rr) for rr in gains])} | {bad}/75 | {sigbad}/75 | {min(mins):+.4f} |")
lines += ['', '全区间网格等权只是一项敏感性汇总，不代表真实SNR/估计误差的出现概率。边界截断会造成部分重复有效条件。','',
 '### 负收益是否真由估计偏差引起？','',
 '| Ns | 模式 | 方法 | 准确信息下负收益点数/15 | 有效非零偏差下负收益点数 | 原本非负、引入偏差后转负的点数 |','|---|---|---|---|---|---|']
for ns in LENGTHS:
    for mode in MODES:
        for method in ('vanilla','conservative'):
            cleanbad=0;errorbad=0;newbad=0;error_total=0
            for row in rows:
                cc=cells(row,mode,ns,interior=False)
                clean={c['true_snr']:c['methods'][method]['gain_vs_input_matched_S0_db'] for c in cc if c['error']==0}
                cleanbad+=sum(v<0 for v in clean.values())
                for c in cc:
                    if c['effective_error']==0:continue
                    error_total+=1
                    negative=c['methods'][method]['gain_vs_input_matched_S0_db']<0
                    errorbad+=negative
                    newbad+=negative and clean[c['true_snr']]>=0
            summary['risk_attribution'][f'{ns}_{mode}_{method}']=dict(clean_negative=cleanbad,
                nonzero_effective_error_negative=errorbad,nonzero_effective_error_total=error_total,newly_negative=newbad)
            lines.append(f'| {ns} | {mode} | {method} | {cleanbad}/15 | {errorbad}/{error_total} | {newbad} |')
lines += ['', '这是观测均值符号的配对诊断，不能把准确信息时已有的负收益归因给SNR误差。新增转负也需结合各点区间，不等于统计显著的因果断言。','',
 '## 3. 按估计偏差分组','',
 '对Ns16/32/64及真实SNR5/10/15等权，先求每种子均值再报告±。','',
 '| 模式 | 偏差/dB | 完整适配相对S0改善/dB | 保守适配相对S0改善/dB | 保守相对完整改善/dB |','|---|---|---|---|---|']
for mode in MODES:
    for e in ERRORS:
        cc=[cells(r,mode,errors=(e,)) for r in rows]
        v=[mean([c['methods']['vanilla']['gain_vs_input_matched_S0_db'] for c in rr]) for rr in cc]
        c=[mean([q['methods']['conservative']['gain_vs_input_matched_S0_db'] for q in rr]) for rr in cc]
        d=[a-b for a,b in zip(c,v)]
        summary['by_bias'][f'{mode}_{e}']=dict(vanilla_gain_db=mean(v),conservative_gain_db=mean(c),
            conservative_minus_vanilla_db=mean(d),per_seed_delta=d)
        lines.append(f'| {mode} | {e:+d} | {ms(v)} | {ms(c)} | {ms(d)} |')
lines += ['', '### 估计偏差本身造成多少变化？','',
 '各方法与自身零偏差版本比较，正值是变差，负值是偶然改善；与上一表的“相对S0收益”不是同一指标。内部SNR、三长度等权。','',
 '| 模式 | 偏差/dB | S0误差代价/dB | 完整适配误差代价/dB | 保守适配误差代价/dB |','|---|---|---|---|---|']
for mode in MODES:
    for e in ERRORS:
        methods={}
        for method in ('S0','vanilla','conservative'):
            values=[]
            for row in rows:
                clean={(c['ns'],c['true_snr']):c for c in cells(row,mode,errors=(0,))}
                cc=cells(row,mode,errors=(e,))
                values.append(mean([c['methods'][method]['nmse_db']-clean[(c['ns'],c['true_snr'])]['methods'][method]['nmse_db'] for c in cc]))
            methods[method]=dict(mean_db=mean(values),per_seed=values)
        summary['estimation_penalty'][f'{mode}_{e}']=methods
        lines.append(f"| {mode} | {e:+d} | {ms(methods['S0']['per_seed'])} | {ms(methods['vanilla']['per_seed'])} | {ms(methods['conservative']['per_seed'])} |")
lines += ['', '## 4. 边界点与截断','',
 '| 真实SNR | 模式 | 方法 | 平均改善/dB | 负收益点数（三种子×三长度×五偏差） |','|---|---|---|---|---|']
for s in (0,20):
    for mode in MODES:
        for method in ('vanilla','conservative'):
            cc=[[c for c in cells(r,mode,interior=False) if c['true_snr']==s] for r in rows]
            values=[[c['methods'][method]['gain_vs_input_matched_S0_db'] for c in rr] for rr in cc]
            lines.append(f'| {s} | {mode} | {method} | {ms([mean(rr) for rr in values])} | {sum(v<0 for rr in values for v in rr)}/45 |')
lines += ['', '例如真实0dB、偏差−3dB截断后仍为0dB，不算3dB外推能力。','',
 '## 5. 准确信息下的原始适配参照','',
 '| Ns | SNR | 完整适配改善/dB |','|---|---|---|']
for ns in LENGTHS:
    for s in (0,5,10,15,20):
        vv=[next(c['gain_db'] for c in r['accurate_reference'] if c['ns']==ns and c['true_snr']==s) for r in rows]
        summary['accurate_reference'][f'{ns}_{s}']=dict(gain_db=mean(vv),per_seed=vv)
        lines.append(f'| {ns} | {s} | {ms(vv)} |')
lines += ['', '## 结论边界','',
 '- 对所有预设条件和三个种子报告，不因测试结果选点、扩大偏差范围或重新调α。',
 '- 选择α用有真值的离线验证数据；部署不需H真值、梯度或在线重标定。不能把离线数据成本省略。',
 '- 该评估检验当前BN架构的适用范围，不能证明LayerNorm/GDN等架构同样存在问题。',
 '- 使用旧统计量收缩是已有思想，不能仅凭本实验主张算法首创。测试结果不提供误差分布改变后的保证。',
 '- 准确SNR下的适配只是信息准确参照，不是数学最优上界；若错误条件偶尔更好，仍应如实保留。保守方法在零偏差时的收益/损失亦单列，不能将一般收缩效果全部归因于抗估计误差。',
 '- 所有逐点NMSE/SGCS、配对区间、有效估计偏差见results_s*.json；逐观测数组见outcomes_s*.npz。',
 '- 数据总体未换场景，未模拟SNR估计器、PUCCH、下行吞吐或CSI老化。','']
(ROOT/'SNR估计误差与保守适配_结果.md').write_text('\n'.join(lines),encoding='utf8')
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(selection=summary['selection'],aggregate=summary['aggregate'],by_bias=summary['by_bias']),ensure_ascii=False,indent=2))
