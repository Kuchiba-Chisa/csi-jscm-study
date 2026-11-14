"""Build a comparison report from finished seeds, without selecting on test."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;P=O.parent/'two_sided_experiments_20261113'
new=[json.loads(x.read_text(encoding='utf8')) for x in sorted(O.glob('results_s*.json'))]
single={int(p.stem.split('s')[-1]):json.loads(p.read_text(encoding='utf8')) for p in O.glob('single_pass_s*.json')}
old={s:json.loads((P/f'results_s{s}.json').read_text(encoding='utf8')) for s in [0,1,2]}
oa={s:json.loads((P/f'action_results_s{s}.json').read_text(encoding='utf8')) for s in [0,1,2]}
def ms(v,d=3):
    a=np.array(v,float);return f'{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}' if len(a)>1 else f'{a[0]:.{d}f}'
labels={
 'old_ue':'旧 UE 小质量头（未校准）',
 'old_ue_MLP_calibrated':'旧 UE 小质量头（验证集仿射校准）',
 'proxy_clean':'UE 干净码字过代理，直接与 H 比较',
 'proxy_mc':'UE 本地独立噪声 MC8 代理，未校准',
 'proxy_mc_calibrated':'UE MC8 代理，验证集仿射校准',
 'teacher_mc':'UE 完整 BS 解码器 MC32（高成本参照）',
 'bs_only':'BS 特征头（原强基线）',
 'old_fusion_noisy2':'旧小头 + BS，2 bit QPSK',
 'proxy_fusion_float':'代理 + BS，浮点提示（诊断）',
 'proxy_fusion_clean2':'代理 + BS，2 bit 无误码（诊断）',
 'proxy_fusion_noisy2':'代理 + BS，2 bit QPSK',
 'proxy_fusion_noisy4':'代理 + BS，4 bit 16-QAM',
 'proxy_fusion_shuffled':'代理浮点提示打乱（诊断）'}
def value(r,key,s,metric):
    if key in ['old_ue','bs_only','old_fusion_noisy2']:
        k={'old_ue':'ue_only','bs_only':'bs_only','old_fusion_noisy2':'fusion_noisy2'}[key]
        return old[r['seed']]['quality'][k]['test'][s][metric]
    return r['quality'][key]['test'][s][metric]

lines=['# Guo 式 UE 代理解码器对照：阶段性结果','',
       f'已完成独立源模型种子：{[x["seed"] for x in new]}。三条件为反馈 SNR=0/10/20 dB，Ns=64，有限 16-QAM，AWGN，合成 CSI。',
       '这是 Guo 启发的“输出蒸馏代理 + 与 UE 真值比较”在当前链路上的实现；不是 Guo 的 Transformer、implicit precoding vector 数据或官方复现，不能据此判定原论文优劣。',
       '全部旧主网络冻结。代理训练使用 3000 个 H，融合头另用不重合的 2000 个 H；验证/测试分别 500/600 个独立 H，测试每条件 3 次实际反馈噪声。均值 ± 样本标准差按三个源模型种子计算。','',
       '**主要读数：代理与 UE 真值比较显著强于原小质量头；原先“UE 小头不如 BS”的结果不能代表 UE 侧监测的信息能力。单次代理 AUROC 约为 0.916/0.956/0.954，BS 单侧约为 0.864/0.890/0.887。**',
       '单次代理经过验证集仿射校准后 MAE 约为 0.501/0.568/0.617 dB；单次代理浮点提示 + BS 为 0.442/0.496/0.524 dB，提示经 2 bit QPSK 信道后为 0.578/0.648/0.691 dB。浮点融合支持信息互补，真实有限提示有量化与误码损失。',
       '同总符号预算下，单次代理 2 bit 融合相对 BS 收益头的净改善仅约 +0.0006/+0.0019/+0.0001 dB，均缺少各种子一致的配对区间支持。不能宣称已取得稳定重传净收益。',
       '单次代理 conv/linear MAC 约为完整 BS 解码器的 21%；MC8 的质量监测未显示值得其八倍计算量的稳定优势。代理拟合保真度仍有限，不能把本结果解释成 Guo 原论文或该方法族的性能极限。','',
       '## 方法与信息边界','',
       '1. 学生代理输入带噪码字和 SNR，监督目标为同输入下冻结 BS 解码器的输出，只优化输出 MSE，不以真实 H 为训练目标。',
       '2. 代理为当前 AE 的单残差块解码器（原 BS 五块），全连接层用 rank=48 因子化；兼容层从自身教师初始化，FC 用 SVD 初始化。此处使用自有教师权重，不验证跨厂商黑盒蒸馏。',
       '3. UE 持有 H 与确定性发送码字 x，但看不到本次 y。分别以干净 x 输入代理，或自行生成 8 组独立 AWGN 运行代理，与 H 比较并平均逐样本 NMSE_dB。后者估计发送前条件平均质量，不是本次实际损失。',
       '4. 验证集上按已知 SNR 拟合一个非负斜率仿射校正，补偿代理分数的尺度/偏差；旧 UE 小头也报告同类校正对照。部署不在线训练，但离线校正依赖带标签验证数据。',
       '5. 量化与提示链路沿用前轮：2 bit QPSK/4 bit 16-QAM，各一个复符号，提示噪声在训练、验证、测试均启用。BS 融合头输入已有 decoder feature、条件和提示，约 2.5 万参数。',
       '6. 高成本参照使用完整 BS 解码器和 32 组独立本地噪声；不使用本次实际噪声。它用于观察代理能力损失，不是信息论上界，也非免费方案。','']
for metric,title in [('mae_db','质量预测 MAE（dB，越低越好）'),('auroc_worst20','坏样本 AUROC（越高越好）')]:
    lines+=['## '+title,'','| 方法 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
    for key,label in labels.items():
        if new:lines.append('| '+label+' | '+' | '.join(ms([value(r,key,s,metric) for r in new]) for s in ['0','10','20'])+' |')
    lines+=['','坏样本是测试各条件最差 20% 的诊断标签，不是部署门限。MAE 改善表示监测更准确，不代表重建网络本身更好。','']
lines+=['## 代理拟合保真度与计算成本','','| 条件 | 代理相对教师输出的相对 MSE（先平均再取 dB） |','|---|---|']
for s in ['0','10','20']:
    if new:lines.append('| '+s+' dB | '+ms([r['proxy_fidelity'][s]['dB_of_mean_relative_MSE'] for r in new])+' |')
if new:
    c=new[0]['cost'];pc=c['proxy'];tc=c['BS_decoder']
    lines+=['',f'代理参数 {pc["parameters"]:,}；BS 解码器 {tc["parameters"]:,}；FP32 权重约为 {pc["parameters"]*4/1024:.1f} 与 {tc["parameters"]*4/1024:.1f} KiB。',
            f'单次代理 conv/linear MAC 约 {pc["conv_linear_MACs_per_sample"]/1e6:.2f}M；单次 BS 解码器约 {tc["conv_linear_MACs_per_sample"]/1e6:.2f}M；MC8 代理约 {c["proxy_MC8_MACs"]/1e6:.2f}M。',
            '**轻量代理单次计算较少，不代表 MC8 总计算仍较少。** 上述只计 conv/linear，未计编码器、归一化、激活、质量比较及 UE 硬件功耗；CPU batch32 时间只供本机参考。',
            '','| 种子 | 选中 epoch | 初始验证 KD MSE | 最佳验证 KD MSE | 训练时间 / 秒 |','|---|---|---|---|---|']
    for r in new:
        m=r['proxy'];lines.append(f'| {r["seed"]} | {m["selected_epoch"]} | {m["initial_val_mse"]:.6f} | {m["val_mse"]:.6f} | {m["seconds"]:.1f} |')
lines+=['','## 同总符号预算的重传收益','',
        '每份主 CSI 为 64 个复符号。BS-only 选 20% 重传；双侧为全部样本增加 1 个提示符号后，重传比例降至约 18.44%，总符号数不超过 BS-only。动作、未来噪声及测试 H 与前轮相同。',
        '下表为全部样本平均逐样本 NMSE_dB 的下降，正为好。无老化、完整 ACK/NACK/调度时延或吞吐仿真；只是固定动作的离线选择评估。','',
        '| 策略 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
actions={'old_bs':'BS 单侧收益头','old_fusion':'旧小头 + BS，2 bit',
         'proxy_float_gain':'代理浮点 + BS（提示诊断）','proxy_clean2_gain':'代理无误码 2 bit + BS（诊断）',
         'proxy_noisy2_gain':'代理有噪 2 bit + BS','proxy_noisy4_gain':'代理有噪 4 bit + BS',
         'proxy_ue_noisy2_gain':'代理 UE 单侧，2 bit 提示',
         'quality_proxy_fusion_noisy2':'按双侧预测质量挑最差（非收益预测）'}
def act(r,key,s):
    if key=='old_bs':return oa[r['seed']]['test'][s]['policies']['bs_gain']['nmse_improvement_db']
    if key=='old_fusion':return oa[r['seed']]['test'][s]['policies']['fusion_noisy2_gain']['nmse_improvement_db']
    return r['actions'][s]['policies'][key]['nmse_improvement_db']
for key,label in actions.items():
    if new:lines.append('| '+label+' | '+' | '.join(ms([act(r,key,s) for r in new],4) for s in ['0','10','20'])+' |')
lines+=['','## 双侧代理相对 BS 单侧的配对净改善','',
        '正为双侧更好。区间按 H 聚类，为固定拟合头部下的测试样本波动；跨源模型种子波动另列。','',
        '| 方案 | SNR | 三种子均值 ± SD | 各种子及 H 聚类 CI95 |','|---|---|---|---|']
for name in ['proxy_noisy2_gain','proxy_noisy4_gain']:
    for s in ['0','10','20']:
        vals=[r['actions'][s]['paired_net_gain_db'][name+'_minus_bs_gain'] for r in new]
        if vals:
            details='; '.join(f's{r["seed"]}: {v["mean"]:+.4f} [{v["CI95"][0]:+.4f},{v["CI95"][1]:+.4f}]' for r,v in zip(new,vals))
            lines.append('| '+name+' | '+s+' | '+ms([v['mean'] for v in vals],4)+' | '+details+' |')
lines+=['','## 结论边界与复现','',
        '- 本实验不具备评价“Guo 原方法不如我们”的条件。论文的源表示、骨干、反馈信道、指标与本实验不同。',
        '- 代理输出与教师仍可能有显著差异，特别是低 SNR；应结合完整 BS 解码器参照，区分代理拟合不足与发送前监测的限制。',
        '- 代理的质量提示未必最适合预测重传收益；本轮不另训练任务专用提示编码器，不把质量监测与动作价值混同。',
        '- 三个源模型种子、一个代理初始化/模型；未做全架构超参数搜索。重建主网络未更新。',
        '- 原先的现有小头/BS结果全部复用同样的源模型、测试 H 和当前/未来噪声；并非重跑后挑更有利的旧结果。',
        '- 主结论仅针对 Ns=64 的三个已知 SNR；未扩展接收端重标定、SNR 失准、信道老化与场景变化。',
        '- 资源配平采用平均符号预算；不同选择样本的实际发射能量可能有小幅差异，JSON 保留实际能量，不能称严格逐样本等能量比较。','',
        '```powershell',
        "& 'simulation/.venv/Scripts/python.exe' -u run_proxy.py --seeds 0 1 2 --epochs 45",
        "& 'simulation/.venv/Scripts/python.exe' -u run_single_pass.py --seeds 0 1 2",
        "& 'simulation/.venv/Scripts/python.exe' -X utf8 summarize.py",'```','',
        '需在脚本目录运行，依赖旁边的 two_sided_experiments_20261113 及已有 simulation 源文件。所有旧权重、代码与结果保持只读。']
if single:
    insert=['## 单次代理对照：不运行本地 MC','',
            '复用同一已训练代理，只对干净发送码字推理一次、与 UE 的 H 比较，再作同类验证集仿射校准。单次计算约为 MC8 的八分之一。',
            '这是完成 MC 主实验时补充的计算成本对照，所有配置和负结果同时保留，不在测试集上选择主结论。','',
            '| 方法 / 指标 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
    for name,label in [('single_proxy_calibrated','UE 单次代理，MAE'),('single_proxy_fusion_float','单次代理 + BS 浮点，MAE'),
                       ('single_proxy_fusion_noisy2','单次代理 + BS 2 bit，MAE')]:
        vals=[r for r in new if r['seed'] in single]
        insert.append('| '+label+' | '+' | '.join(ms([single[r['seed']]['quality'][name]['test'][s]['mae_db'] for r in vals]) for s in ['0','10','20'])+' |')
    for name,label in [('single_proxy_calibrated','UE 单次代理，AUROC'),('single_proxy_fusion_noisy2','单次代理 + BS 2 bit，AUROC')]:
        insert.append('| '+label+' | '+' | '.join(ms([single[r['seed']]['quality'][name]['test'][s]['auroc_worst20'] for r in vals]) for s in ['0','10','20'])+' |')
    insert+=['','| 同总符号预算的重传策略 | 0 dB 改善 | 10 dB 改善 | 20 dB 改善 |','|---|---|---|---|']
    for name,label in [('single_proxy_float_gain','单次代理浮点 + BS（诊断）'),('single_proxy_noisy2_gain','单次代理 2 bit + BS')]:
        insert.append('| '+label+' | '+' | '.join(ms([single[r['seed']]['actions'][s]['policies'][name]['nmse_improvement_db'] for r in vals],4) for s in ['0','10','20'])+' |')
    insert+=['','| 单次代理 2 bit 相对 BS 的净配对差 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|',
        '| 均值 ± SD | '+' | '.join(ms([single[r['seed']]['actions'][s]['paired_net_gain_db']['single_proxy_noisy2_gain_minus_bs_gain']['mean'] for r in vals],4) for s in ['0','10','20'])+' |','']
    pos=lines.index('## 结论边界与复现');lines[pos:pos]=insert
(O/'Guo式代理对照结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(O/'Guo式代理对照结果.md')
if new:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13.5,3.8));snrs=[0,10,20]
    plotnames=[('old_ue_MLP_calibrated','UE small head calibrated'),('proxy_mc_calibrated','UE proxy MC8'),('bs_only','BS only'),('proxy_fusion_noisy2','Proxy MC8 + BS 2-bit')]
    if len(single)==len(new):
        plotnames=[('old_ue_MLP_calibrated','UE small head calibrated'),('single_proxy_calibrated','UE proxy single-pass'),('bs_only','BS only'),('single_proxy_fusion_noisy2','Proxy single-pass + BS 2-bit')]
    for name,label in plotnames:
        for ax,key in zip(axs[:2],['mae_db','auroc_worst20']):
            v=np.array([[(single[r['seed']]['quality'][name]['test'][str(s)][key] if name.startswith('single_') else value(r,name,str(s),key)) for s in snrs] for r in new])
            ax.errorbar(snrs,v.mean(0),yerr=v.std(0,ddof=1) if len(v)>1 else None,label=label,marker='o',capsize=3)
    for name,label,color in [('old_bs','BS only','tab:green'),('old_fusion','Small head + BS','tab:purple'),('single_proxy_noisy2_gain' if len(single)==len(new) else 'proxy_noisy2_gain','Proxy + BS 2-bit','tab:red')]:
        v=np.array([[(single[r['seed']]['actions'][str(s)]['policies'][name]['nmse_improvement_db'] if name.startswith('single_') else act(r,name,str(s))) for s in snrs] for r in new])
        axs[2].errorbar(snrs,v.mean(0),yerr=v.std(0,ddof=1) if len(v)>1 else None,label=label,marker='o',capsize=3,color=color)
    for ax in axs:ax.grid(alpha=.25);ax.set_xticks(snrs);ax.set_xlabel('Feedback SNR (dB)')
    axs[0].set_ylabel('Quality prediction MAE (dB)');axs[1].set_ylabel('AUROC (worst 20%)')
    axs[2].set_ylabel('NMSE improvement, matched symbols (dB)');axs[0].legend(fontsize=8);axs[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(O/'guo_proxy_comparison.png',dpi=170);plt.close(fig)
