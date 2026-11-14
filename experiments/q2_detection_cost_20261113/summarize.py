"""Summarize all predeclared operating points; no test threshold optimization."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
rows=[json.loads((OUT/f'results_s{s}.json').read_text(encoding='utf8')) for s in (0,1,2)]
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
audit=json.loads((OUT/'audit.json').read_text(encoding='utf8'))
assert audit['status']=='PASS'
methods=('bs_small','bs_large','ue_proxy')
names={'bs_small':'BS 小质量头','bs_large':'BS 大质量头','ue_proxy':'UE 单次代理'}
en={'bs_small':'BS small head','bs_large':'BS larger head','ue_proxy':'UE single proxy'}
colors={'bs_small':'#2874A6','bs_large':'#7F8C8D','ue_proxy':'#D35400'}
snrs=(0,10,20)

def stats(values):
    return float(np.mean(values)),float(np.std(values,ddof=1))

def pm(values,scale=1,digits=2):
    m,s=stats(np.array(values)*scale)
    return f'{m:.{digits}f} ± {s:.{digits}f}'

def vals(s,t,m,k):
    return [r['conditions'][str(s)]['targets'][str(t)]['methods'][m]['test'][k] for r in rows]

summary={'primary_target':.9,'detection':{},'cost':rows[0]['cost'],'timing':{},
         'note':'mean and sample SD across 3 source-model seeds; all target thresholds fitted on validation only'}
for s in snrs:
    summary['detection'][str(s)]={}
    for t in (.8,.9,.95):
        summary['detection'][str(s)][str(t)]={m:{k:dict(zip(('mean','seed_sd'),stats(vals(s,t,m,k))))
                                           for k in ('tpr','fpr','precision','alarm_fraction','bad_prevalence')}
                                            for m in methods}
for m in methods:
    summary['timing'][m]=dict(zip(('mean_seed_median_ms','seed_sd'),
                                stats([r['timing'][m+'_increment']['median_ms'] for r in rows])))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')

lines=[
'# Q2：固定报警门限与新增计算成本评估',
'',
'记录日期：2026-11-13。三种子全部完成；没有新增训练，使用已有 UE 代理与 BS 质量头。',
'',
'**结论：BS 侧监测显著减少终端新增计算，但当前实现的误报明显多于 UE 代理。扩大已有 BS 头没有稳定解决这一问题。因此，目前得到的是明确的精度与部署位置取舍，尚不能认定任何一侧综合更优。**',
'',
'## 1. 本轮究竟补了什么',
'',
'每个已有重建模型和反馈 SNR，验证集前 250 个 H（每 H 两次已有噪声观察）的 NMSE_dB 第 80 百分位用于定义“坏样本”；验证集后 250 个 H 用于固定报警门限。主设置要求门限在后一组验证数据上检出至少 90% 的坏样本，80%/95% 同时作为预设对照。',
'',
'测试时冻结坏样本标准和报警门限，不按测试集重排名额、不重新选择阈值。每条件测试 600 个独立 H × 3 次实际噪声。模型种子 0/1/2；Ns=64、16-QAM、AWGN、合成 CSI、当前冻结 RD-AE 改编骨干。不是 Guo 或 RD-JSCC 官方复现。',
'',
'验证集以前已用于模型选择；本次 A/B 分组避免坏样本定义和门限设置复用相同 H，但不是新收集的独立验证集。“坏”仍是按条件定义的相对重建误差标准，尚不是吞吐或业务失效门限。',
'',
'## 2. 主结果：验证集目标检出率 90%，测试门限冻结',
'',
'下表为三个源模型种子间均值 ± 样本标准差，单位均为 %。检出率 = 坏样本中被报警的比例；误报率 = 正常样本中被报警的比例；报警精确率 = 报警样本中确实坏的比例。',
'',
'| SNR | 方法 | 测试检出率 | 测试误报率 | 报警精确率 | 总报警比例 |',
'|---|---|---|---|---|---|',
]
for s in snrs:
    for m in methods:
        lines.append(f'| {s} dB | {names[m]} | '+ ' | '.join(pm(vals(s,.9,m,k),100) for k in ('tpr','fpr','precision','alarm_fraction'))+' |')
lines += ['',
'验证目标相同，不保证测试集的实际检出率严格相同；所以这不是“测试检出率完全配平”的比较。不得反过来使用测试标签重调门限。',
'',
'| SNR | 冻结标准下测试坏样本占比 | 三种子各自的坏样本 NMSE_dB 门限 |',
'|---|---|---|']
for s in snrs:
    prevalence=vals(s,.9,'bs_small','bad_prevalence')
    cuts=[r['conditions'][str(s)]['bad_nmse_db_threshold'] for r in rows]
    lines.append(f'| {s} dB | {pm(prevalence,100)}% | '+ ' / '.join(f'{x:.3f}' for x in cuts)+' |')
lines += ['',
'对照含义：BS 小头确实可以识别部分风险，但为保持较高检出率，会把较多正常 CSI 也标为可疑。如果后续动作有成本，误报可能削弱部署收益。本轮没有据此计算吞吐、回退或重传收益。',
'',
'## 3. 预设的其他报警目标',
'',
'| 验证目标检出率 | SNR | BS 小头：测试检出率 / 误报率 | BS 大头：测试检出率 / 误报率 | UE 代理：测试检出率 / 误报率 |',
'|---|---|---|---|---|']
for t in (.8,.95):
    for s in snrs:
        line=f'| {t*100:.0f}% | {s} dB | '
        line+=' | '.join(pm(vals(s,t,m,'tpr'),100)+' / '+pm(vals(s,t,m,'fpr'),100) for m in methods)+' |'
        lines.append(line)
lines += ['',
'这些设置按验证集规则同时保留，不从中挑选最有利测试结果作为新主结论。完整每种子混淆矩阵、按 H 聚类的 95% 区间和两侧配对差保存在 results_s*.json。聚类区间只覆盖固定模型及门限下测试 H 的变化，不包括训练或门限拟合的不确定性。',
'',
'## 4. 新增参数、存储和计算',
'',
'只统计监测新增部分。UE 编码器与 BS 主解码器本来就用于反馈重建，不作为监测新增成本。UE 代理用干净发送码字推理一次，再与 UE 持有的 H 比较；BS 头使用解码过程已有的中间特征，不重新运行一次解码器。',
'',
'| 方法 | 计算位置 | 新增参数 | 持久张量 / KiB | 卷积及线性层 MAC/次 | 同 PC 单样本时间 / ms |',
'|---|---|---|---|---|---|']
for m in methods:
    c=rows[0]['cost'][m]
    timing=[r['timing'][m+'_increment']['median_ms'] for r in rows]
    lines.append(f"| {names[m]} | {'UE' if m=='ue_proxy' else 'BS'} | {c['parameters']:,} | {c['persistent_tensor_bytes_including_preprocessing']/1024:.2f} | {c['conv_linear_macs']:,} | {pm(timing,digits=3)} |")
ratio=rows[0]['cost']['ue_proxy']['conv_linear_macs']/rows[0]['cost']['bs_small']['conv_linear_macs']
lines += ['',
f'UE 代理卷积/线性 MAC 约为 BS 小头的 **{ratio:.1f} 倍**；这不等于真实运行时间或功耗相差同样倍数。统计不包含归一化、激活、池化的 MAC；运行时间包含 BS 特征池化、拼接、归一化、头部、报警比较，以及 UE 代理、误差计算和报警比较。',
'',
'持久张量包括参数、BN 缓冲、BS 标准化张量和按三个 SNR 存储的 FP32 报警门限，不包括运行框架、瞬时激活、CSI/码字缓存和网络控制状态。',
'',
f"计时平台：{protocol['hardware']['cpu']}，{protocol['hardware']['os']}，PyTorch {protocol['hardware']['torch']}，CPU 单线程、FP32、batch=1、预热后 {protocol['args']['timing_blocks']} 个随机交错计时块，每块 {protocol['args']['timing_iterations']} 次。表中先取每个源模型各块中位数，再报告三模型的均值±SD。",
'',
'**这些是同一台 PC 的实现时间，不能替代实际 UE 与 BS 硬件上的时延、能耗或多用户负载测量。**',
'',
'### BS 原解码器和集成监测时间（仅作上下文）',
'',
'| 路径 | 同 PC 单样本时间 / ms |',
'|---|---|']
for key,label in [('decoder_tapped_baseline','原解码器＋相同特征读取钩子'),('decoder_plus_bs_small','原解码器＋BS 小头监测'),('decoder_plus_bs_large','原解码器＋BS 大头监测')]:
    lines.append(f"| {label} | {pm([r['integrated_timing'][key]['median_ms'] for r in rows],digits=3)} |")
lines += ['',
'集成时间保留随机计时块原始值。短时间差受 CPU 波动和 Python 调用影响，不把两条总时间相减当成精确的增量硬件时延；上表的直接增量监测时间更便于理解新增工作。',
'',
'## 5. 本轮可以支持的应用故事',
'',
'> BS 每次重建 CSI 后，利用现有解码器特征输出一个风险判断，不等待批量 CSI。我们比较把监测放在 BS 与在 UE 增加代理解码器之间的取舍：前者不要求 UE 新增代理计算，后者利用 UE 持有的真实 CSI，当前检测更准确。',
'',
'当前证据止于“标记风险的精度与新增计算成本”。标记之后如何保守预编码、是否更新反馈、以及实际吞吐收益仍未实现。原离线重复发送实验保留为应用探索，不作为已落地的调度/HARQ 方案。',
'',
'## 6. 对 Q2 的结论更新',
'',
'1. Q2 仍为逐样本质量监测，本次只是补全固定门限和成本证据，没有换研究主线。',
'2. 当前 BS 小头显著节省终端新增网络计算，但误报率高；不能只用“BS 算力强、容易部署”认定综合更优。',
'3. 现有 BS 大头的实际参数为 83,201（此前“10 万预算”是目标预算称呼），三种子没有呈现各条件一致的检测优势，不能认为单纯加宽已解决差距。',
'4. UE 代理不是零成本，但本轮的检测效果更好；不能因为部署在 UE 就把它排除。',
'5. 没有新训练或新网络创新；本轮贡献是把已有方案的部署取舍量化。真正的端到端应用价值仍取决于报警后的动作成本和后果。',
'',
'## 7. 审计与文件',
'',
'audit.json 为 PASS：三种子逐一复算验证阈值、测试混淆矩阵与比率，确认 UE 分数不依赖实际三次噪声实现；复核原始文件哈希、缓存标签/特征前向一致性、参数和解析 MAC。原模型、代理、质量头未训练也未修改。',
'',
'- evaluate.py / protocol.json：冻结规则、计时协议、环境和评估实现。',
'- predictions_s*.npz：验证与测试标签、分数、逐条件固定门限报警结果。',
'- results_s*.json：混淆矩阵、聚类区间、配对差、成本与原始计时块。',
'- audit.py / audit.json：独立复算与审计。',
'- summary.json / q2_detection_cost.png / q2_detection_cost.pdf：汇总和图表。',
'- README.md：复现命令与限制。',
]
(OUT/'Q2固定门限与新增成本评估结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(11,7.6),layout='constrained')
for m in methods:
    for k,axis,title in [('tpr',ax[0,0],'Test detection rate'),('fpr',ax[0,1],'Test false-alarm rate')]:
        ms=np.array([stats(vals(s,.9,m,k)) for s in snrs])*100
        axis.errorbar(snrs,ms[:,0],yerr=ms[:,1],fmt='o-',capsize=4,label=en[m],color=colors[m],lw=1.6)
        axis.set(title=title,xlabel='Feedback SNR (dB)',ylabel='Percent',xticks=snrs)
        axis.grid(axis='y',alpha=.2)
ax[0,0].axhline(90,color='#999999',ls='--',lw=1)
ax[0,0].set_ylim(70,103);ax[0,1].set_ylim(0,75)
ax[0,0].legend(fontsize=8,loc='lower right')
labels=[en[m].replace(' ','\n',1) for m in methods]
macs=[rows[0]['cost'][m]['conv_linear_macs'] for m in methods]
ax[1,0].bar(labels,macs,color=[colors[m] for m in methods],width=.55)
ax[1,0].set(yscale='log',ylabel='Conv/linear MAC per sample',title='Additional monitoring compute')
ax[1,0].set_ylim(1e4,2e7)
for i,val in enumerate(macs):ax[1,0].text(i,val*1.2,f'{val/1e6:.3f}M',ha='center',fontsize=9)
timings=np.array([stats([r['timing'][m+'_increment']['median_ms'] for r in rows]) for m in methods])
ax[1,1].bar(labels,timings[:,0],yerr=timings[:,1],color=[colors[m] for m in methods],capsize=4,width=.55)
ax[1,1].set(ylabel='CPU milliseconds / sample',title='Additional monitoring time (same PC)')
for i,val in enumerate(timings[:,0]):ax[1,1].text(i,val+max(timings[:,0])*.06,f'{val:.3f}',ha='center',fontsize=9)
ax[1,1].set_ylim(0,max(timings[:,0]+timings[:,1])*1.3)
fig.suptitle('CSI quality monitoring: frozen validation thresholds and incremental cost\nValidation target: 90% detection; error bars: SD across 3 source-model seeds',fontsize=12)
fig.savefig(OUT/'q2_detection_cost.png',dpi=180)
fig.savefig(OUT/'q2_detection_cost.pdf')
plt.close(fig)
print('\n'.join(lines[:38]))
print('Cost ratios: UE/BS-small conv-linear MAC =',ratio)
