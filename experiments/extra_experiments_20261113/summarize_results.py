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
    return f'{a.mean():.3f} ± {a.std(ddof=1):.3f}' if len(a)>1 else f'{a.mean():.3f} (n=1)'


def main():
    q1=json.loads((ROOT/'q1_results.json').read_text(encoding='utf8'))
    q2=json.loads((ROOT/'q2_results.json').read_text(encoding='utf8'))
    lines=['# 今日补充实验结果：接收端适配范围与质量预测网络','',
           '范围：既有 RD-AE 改编模型；不是官方端到端 RD-JSCC 复现。全部旧模型和旧结果文件只读。',
           '', '## Q1：固定发送端，改变接收端的适配范围','',
           '均值 ± 独立源模型种子间样本标准差。每条件测试 400 个 H×3 次配对噪声。',
           '所有可学习适配均从同一 S2 出发，固定全部 BN 运行统计与 p_cal。', '',
           '| 20 dB 部署长度 | S0 | S1 | S2 | 仅 BN 仿射 | 仅 AF | 完整接收端 |',
           '|---|---|---|---|---|---|---|']
    for ns in (16,32):
        cells=[r for r in q1['cells'] if r['ns']==ns]
        lines.append('| '+str(ns)+' | '+' | '.join(ms([r['arms'][m]['test_nmse_db'] for r in cells])
                      for m in ('S0','S1','S2','bn_affine','af','decoder'))+' |')
    lines+=['','| 相对 S2 的额外改善（正为好，dB） | BN 仿射（264 参数） | AF（4966 参数） | 完整接收端（291110 参数） |',
            '|---|---|---|---|']
    for ns in (16,32):
        cells=[r for r in q1['cells'] if r['ns']==ns]
        lines.append('| Ns='+str(ns)+' | '+' | '.join(ms([r['arms'][m]['gain_over_S2_db'] for r in cells])
                      for m in ('bn_affine','af','decoder'))+' |')
    lines+=['','FP32 条件专用参数存储：BN 仿射约 1.03 KiB，AF 约 19.40 KiB，完整接收端约 1137.15 KiB；统计量存储另计。',
            '完整接收端使用两个预设学习率按验证集选择，BN/AF 使用一个预设学习率；并非完全等训练预算。',
            '没有修改发送端、没有从头训练各条件专用模型；本表不能给出完整共享代价或理论上界。','',
            '## Q2：同等规模网络能否从解码器特征判断重建质量','',
            '三个预测器均约 2.5 万参数；重建网络冻结，训练/验证/测试按 H 分离。训练与验证仅用 0/10/20 dB；5/15 dB 是监测器未见条件。',
            '以下 AUROC 按每个 (SNR,Ns) 独立计算最差 20% 标签，不能用作部署阈值。', '',
            '| SNR | Ns | y+条件 | 特征+条件 | y+特征+条件 | 特征相对 y 的配对差 |',
            '|---|---|---|---|---|---|']
    for snr in (0,5,10,15,20):
        for ns in (16,32,64):
            key=f'{snr}dB_ns{ns}'
            values={m:[r['heads'][m]['test'][key]['auroc'] for r in q2['models']]
                    for m in ('y','feature','y_feature')}
            delta=np.array(values['feature'])-np.array(values['y'])
            lines.append(f'| {snr} | {ns} | '+' | '.join(ms(values[m]) for m in values)+' | '+ms(delta)+' |')
    lines+=['','### MAE 与选择性风险（20 dB）','',
            '| Ns | 预测器 | MAE / dB | 保留 70% 的平均 NMSE / dB | 全部样本平均 NMSE / dB |',
            '|---|---|---|---|---|']
    for ns in (16,32,64):
        key=f'20dB_ns{ns}'
        for mode in ('y','feature','y_feature'):
            records=[r['heads'][mode]['test'][key] for r in q2['models']]
            lines.append(f'| {ns} | {mode} | '+ms([r['mae_db'] for r in records])+' | '+
                         ms([r['risk']['0.70'] for r in records])+' | '+ms([r['mean_nmse_db'] for r in records])+' |')
    lines+=['','### 接收端 S2 校准后，不重训质量头','',
            '同一个 H 和噪声下，接收 y 逐元素不变；重建目标和解码器特征随接收端状态改变。',
            '监测器旧/新状态间 AUROC 的变化同时包含新误差排序的影响，不能直接归因于单一特征分布变化。', '',
            '| Ns | 预测器 | S0 AUROC | S2 AUROC | S0 MAE | S2 MAE | 验证集单标量偏差校正后 MAE |',
            '|---|---|---|---|---|---|---|']
    for ns in (16,32):
        key=f'20dB_ns{ns}'
        for mode in ('y','feature','y_feature'):
            ready=[r for r in q2['models'] if key in r.get('transfer',{})]
            if not ready:
                continue
            groups=[[r['heads'][mode]['test'][key][metric] for r in ready] for metric in ('auroc','mae_db')]
            trans=[[r['transfer'][key][mode][metric] for r in ready] for metric in ('auroc','mae_db')]
            corrected=[r.get('transfer_validation_bias_correction',{}).get(key,{}).get(mode,{}).get('mae_db') for r in ready]
            last=ms(corrected) if all(v is not None for v in corrected) else '未完成'
            lines.append(f'| {ns} | {mode} | {ms(groups[0])} | {ms(trans[0])} | {ms(groups[1])} | {ms(trans[1])} | {last} |')
    lines+=['','## 这次新增实验增加了什么认识','',
            '1. Q1 从“校准有没有收益”推进到“同一校准状态之后，增加适配参数还能拿多少收益”。少参数适配与完整接收端微调的边际收益都应和 S2 比，不能把校准收益重复归功于微调。',
            '2. Q2 直接利用已有解码器特征，并保持头部规模相近。更高 AUROC 不代表特征含有超过 y 的信息，而表明相同训练预算下更适合当前质量预测任务。',
            '3. Q1 与 Q2 的连接点是模型维护：接收端校准改变了误差标签，也改变了特征监测器的输入。旧质量头在新接收端上的排序、偏差和绝对误差必须重测，不能只验证重建变好就认为监测仍有效。',
            '4. 验证集单标量偏差修正不改变样本排序，因此不可能改善 AUROC；它只检验质量头失准中有多少能用一个偏移量解释，并需要带真实质量标签的验证样本。']
    lines+=['','## 结论边界','',
            '- 这是针对已有结果的跟进实验，两个 Q1 部署点由已知结果选定，不是全条件无偏普查。',
            '- 读取 decoder feature 没有新增信息；它是 y 和模型状态的确定性函数。性能差异反映本次网络和训练预算下表示方式的可学习性。',
            '- 质量预测网络、AF 微调、BN 仿射微调都是已有方法路线，不称为新算法。',
            '- AUROC 的种子标准差来自三个既有重建模型及对应质量头，不能把 400×3 组观察当成 1200 个独立 H。',
            '- 保留较好样本的风险改善不等于吞吐/波束赋形系统收益，没有实现完整回退协议。',
            '- 绝对 NMSE 与旧表可能因功率标定网格和评估噪声不同而有差异，本次组间比较完全配对。',
            '- 正式脚本、随机种子、验证选择轨迹、源文件 SHA256 和所有数值保存在同目录 JSON 中。','']
    (ROOT/'今日补充实验结果.md').write_text('\n'.join(lines),encoding='utf8')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    modes=('S0','S1','S2','bn_affine','af','decoder')
    labels=('Original','Mixed BN','Target BN','BN affine','AF only','Full receiver')
    colors=('#8996a5','#709fc0','#43a6a1','#f0b44d','#db8249','#9b69af')
    for ax,ns in zip(axes,(16,32)):
        rows=[r for r in q1['cells'] if r['ns']==ns]
        matrix=np.array([[r['arms'][m]['test_nmse_db'] for m in modes] for r in rows])
        ax.plot(np.arange(6),matrix.mean(0),color='#b6bdc5',lw=1,zorder=1)
        for i,color in enumerate(colors):
            ax.errorbar(i,matrix.mean(0)[i],yerr=matrix.std(0,ddof=1)[i],
                        fmt='o',color=color,capsize=4,markersize=7,zorder=2)
        ax.set_xticks(np.arange(6),labels,rotation=28,ha='right')
        ax.set_title(f'20 dB, Ns={ns}');ax.set_ylabel('Mean sample NMSE (dB), lower is better')
        ax.set_ylim(float(matrix.min()-0.25),float(matrix.max()+0.25))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(ROOT/'q1_adaptation.png',dpi=180);fig.savefig(ROOT/'q1_adaptation.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained',sharey=True)
    for ax,ns in zip(axes,(16,32,64)):
        for mode,color,label in [('y','#6d91b3','Received y'),('feature','#d47a40','Decoder feature'),('y_feature','#65a18a','y + feature')]:
            a=np.array([[r['heads'][mode]['test'][f'{s}dB_ns{ns}']['auroc'] for s in (0,5,10,15,20)] for r in q2['models']])
            ax.errorbar([0,5,10,15,20],a.mean(0),yerr=a.std(0,ddof=1),label=label,color=color,marker='o',capsize=2)
        ax.axhline(.5,color='gray',ls='--',lw=1);ax.set_ylim(.45,1.0);ax.set_title(f'Ns={ns}')
        ax.set_xlabel('Feedback SNR (dB)');ax.set_xticks([0,5,10,15,20]);ax.grid(alpha=.2)
    axes[0].set_ylabel('AUROC, higher is better');axes[-1].legend(fontsize=8,loc='lower right')
    fig.savefig(ROOT/'q2_quality_heads.png',dpi=180);fig.savefig(ROOT/'q2_quality_heads.pdf');plt.close(fig)
    print('Saved report and figures',flush=True)


if __name__=='__main__':
    main()
