from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def ms(values):
    a=np.asarray(values,dtype=float)
    return f'{a.mean():.3f} ± {a.std(ddof=1):.3f}' if len(a)>1 else f'{a.mean():.3f} (n={len(a)})'


def main():
    a=json.loads((ROOT/'q1_anchor_results.json').read_text(encoding='utf8'))
    b=json.loads((ROOT/'q2_action_results.json').read_text(encoding='utf8'))
    methods=['S0','nearest','linear_variance','second_moment','target_reference']
    complete=len(a['cells'])==27 and len(b['models'])==3
    gaps=[]
    for ns in (16,32,64):
        for snr in (5,10,15):
            matching=[c for c in a['cells'] if c['ns']==ns and c['snr_db']==snr]
            if matching:
                gaps.append(float(np.mean([c['methods']['linear_variance']-c['methods']['target_reference'] for c in matching])))
    low20=[r['test']['0']['policies']['0.2'] for r in b['models']]
    low30=[r['test']['0']['policies']['0.3'] for r in b['models']]
    rows=['# 后续实验结果：少锚点部署适配与重传收益预测','',
          '本轮基于三个既有 RD-AE 改编模型，冻结原始权重；不是官方完整 RD-JSCC 的复现或评估。',
          '数值为源模型种子间均值 ± 样本标准差；重复噪声不作为独立种子。',
          f'完成度：Q1 {len(a["cells"])}/27 个种子×部署点；Q2 {len(b["models"])}/3 个源模型种子。','',
          '## 关键读数','',
          f'- Q1：在线性均值/方差插值方案下，各部署点三种子平均 NMSE 与逐点校准参照的最大绝对差约 {max(abs(v) for v in gaps):.3f} dB。只测已知长度、SNR 区间内插值；不是外推保证。',
          f'- Q2：0 dB、20% 重传时，预测收益相对随机额外改善 {ms([e["learned_gain_minus_random_db"] for e in low20])} dB；30% 时为 {ms([e["learned_gain_minus_random_db"] for e in low30])} dB。',
          '- Q2：10 dB 的额外收益更小；20 dB 没有跨三个种子都稳定优于随机的证据。',
          '- 两项均有部署边界：Q1 仍需离线锚点校准；Q2 仍需离线带标签训练，且额外重传资源和控制开销不能忽略。','',
          '## Q1：0/20 dB 两锚点能否覆盖中间 SNR','',
          '均为平均逐样本 NMSE_dB，越低越好。锚点方法不使用目标 SNR 校准集，不在线训练。目标校准参照额外使用该目标条件训练数据。', '',
          '| Ns | 目标 SNR | 原始 | 最近锚点 | 均值/方差插值 | 均值/二阶矩插值 | 逐目标校准参照 |',
          '|---|---|---|---|---|---|---|']
    anchor_aggregate=[]
    for ns in (16,32,64):
        for snr in (5,10,15):
            cells=[c for c in a['cells'] if c['ns']==ns and c['snr_db']==snr]
            if not cells:continue
            rows.append(f'| {ns} | {snr} | '+' | '.join(ms([c['methods'][m] for c in cells]) for m in methods)+' |')
            anchor_aggregate.append({'ns':ns,'snr':snr,
                    'linear_minus_target_db':[c['methods']['linear_variance']-c['methods']['target_reference'] for c in cells],
                    'nearest_minus_linear_db':[c['methods']['nearest']-c['methods']['linear_variance'] for c in cells]})
    rows+=['','插值的统计依据有边界：二阶矩线性插值对应一种矩构造，不能证明中间 SNR 的非线性特征分布满足该公式。两个规则事先固定、同时报告。',
           '没有测 SNR 外推、未见反馈长度或场景漂移；SNR 条件估计误差与切换控制开销未建模。','',
           '## Q2：相同重传预算，预测收益能否优于预测质量','',
           '动作：重复相同 64 个复 QAM 符号，对两次观测平均，保持解码器原 SNR 输入。重传 20%/30% 样本，额外符号资源相同；不计往返控制开销。',
           '收益是整体平均逐样本 NMSE_dB 的下降，正为好。两预测器输入/规模/样本完全相同，区别仅为监督目标。','',
           '| SNR | 重传比例 | 随机选择期望收益 | 按预测质量挑最差 | 按预测收益挑最大 | 收益预测−随机 | 收益预测−质量预测 | 特权独立未来噪声参照 |',
           '|---|---|---|---|---|---|---|---|']
    for snr in (0,10,20):
        for p in (.2,.3):
            entries=[r['test'][str(snr)]['policies'][str(p)] for r in b['models']]
            cols=[[e['random_expected_gain_db'] for e in entries],
                  [e['predicted_quality']['gain_db'] for e in entries],
                  [e['predicted_gain']['gain_db'] for e in entries],
                  [e['learned_gain_minus_random_db'] for e in entries],
                  [e['learned_gain_minus_quality_db'] for e in entries],
                  [e['gain_reference_independent_future']['gain_db'] for e in entries]]
            rows.append(f'| {snr} | {p:.0%} | '+' | '.join(ms(c) for c in cols)+' |')
    rows+=['','### 分数到底在排什么','',
           '| SNR | 预测收益与实际 MC 收益的 Spearman | 预测质量与实际 MC 收益的 Spearman | 全部重传的平均收益 / dB |',
           '|---|---|---|---|']
    for snr in (0,10,20):
        records=[r['test'][str(snr)] for r in b['models']]
        rows.append(f'| {snr} | '+' | '.join(ms([r[k] for r in records]) for k in
                   ('gain_score_spearman','quality_score_spearman_with_gain','all_retransmit_gain_db'))+' |')
    rows+=['','## 怎样解释','',
           '- 比较“收益预测−随机”的配对差，并同时看三个种子和按 H 聚类的区间；不能只看其中一个种子或均值为正。',
           '- 若只优于质量预测但未稳定优于随机，只能说预测质量不是好的重传依据，不能说新策略有效。',
           '- 高 SNR 即便排序相关，也可能由于动作本身收益很小而没有实际价值。',
           '- 特权参照需真实 H，并用前 16 次未来噪声排序、后 16 次独立噪声评估；它是不可部署经验参照，不是数学性能上界。',
           '- JSON 中另含“按评估的同一组未来噪声选最大”的 hindsight 参照，该列不可用于论证可预测空间。',
           '- 本次没有优化重传后的接收器，也没有证明任何可用观测下的最优动作策略。负结果只能约束本次表示、监督和训练预算。',
           '- 标准差来自三个已训练重建模型及配套监测器，按 H 的 bootstrap 区间只覆盖固定决策集合的测试样本波动。',
           '- 统计插值、收益回归及 MLP 都不主张首创；意义是检验它们是否解决当前模型里明确观察到的问题。','',
           '## 文件','',
           '- q1_anchor_results.json：Q1 完整条件表、checkpoint SHA256、固定功率标定。',
           '- q2_action_results.json：Q2 策略收益、配对差、H 聚类区间、训练选择记录。',
           '- action_test_s*.npz：每个 H/当前观测的未来损失、预测分数，可独立重算策略收益。',
           '- README.md：实验设计、信息边界与复现命令。','']
    (ROOT/'后续实验结果.md').write_text('\n'.join(rows),encoding='utf8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=False,layout='constrained')
    colors={'S0':'#8d969e','nearest':'#86a7bd','linear_variance':'#d77e43','second_moment':'#75a492','target_reference':'#8c70a6'}
    labels={'S0':'Original','nearest':'Nearest anchor','linear_variance':'Linear mean/variance',
            'second_moment':'Linear raw moment','target_reference':'Target calibration'}
    for ax,ns in zip(axes,(16,32,64)):
        for m in methods:
            matrix=np.array([[c['methods'][m] for c in a['cells'] if c['ns']==ns and c['snr_db']==s] for s in (5,10,15)])
            ax.errorbar([5,10,15],matrix.mean(1),yerr=matrix.std(1,ddof=1),color=colors[m],label=labels[m],marker='o',capsize=2,lw=1.4)
        ax.set_title(f'Ns={ns}');ax.set_xlabel('Uncalibrated target SNR (dB)');ax.set_xticks([5,10,15]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean sample NMSE (dB), lower is better');axes[-1].legend(fontsize=7)
    fig.savefig(ROOT/'q1_anchor_interpolation.png',dpi=180);fig.savefig(ROOT/'q1_anchor_interpolation.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained',sharey=True)
    for ax,snr in zip(axes,(0,10,20)):
        for i,p in enumerate((.2,.3)):
            records=[r['test'][str(snr)]['policies'][str(p)] for r in b['models']]
            for j,(label,key,color) in enumerate([('Random',None,'#9ba6b0'),('Quality','predicted_quality','#77a4bd'),('Value','predicted_gain','#d88145')]):
                vals=np.array([e['random_expected_gain_db'] if key is None else e[key]['gain_db'] for e in records])
                ax.bar(i+(j-1)*.23,vals.mean(),width=.22,yerr=vals.std(ddof=1),color=color,capsize=2,label=label if i==0 else None)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks([0,1],['20%','30%']);ax.set_xlabel('Retransmission fraction')
        ax.set_title(f'{snr} dB');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean sample NMSE reduction (dB)');axes[-1].legend(fontsize=8)
    fig.savefig(ROOT/'q2_action_value.png',dpi=180);fig.savefig(ROOT/'q2_action_value.pdf');plt.close(fig)
    print('Saved follow-up report and figures')


if __name__=='__main__':main()
