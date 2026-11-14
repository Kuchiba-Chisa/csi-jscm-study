"""Report every seed/condition, with SNR-only controls and no test tuning."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
rows=[json.loads((OUT/f'results_s{s}.json').read_text(encoding='utf8')) for s in (0,1,2)]
audit=json.loads((OUT/'audit.json').read_text(encoding='utf8'));assert audit['status']=='PASS'
timing=json.loads((OUT/'timing.json').read_text(encoding='utf8'))
snrs=(0,10,20)

def pm(v,d=4,scale=1):
    x=np.asarray(v)*scale
    return f'{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}'

def cell(s,key):return [r['conditions'][str(s)][key] for r in rows]
def gate(s,q,key):return [r['conditions'][str(s)]['gates'][str(q)][key] for r in rows]

def allocate(order,fraction):
    rates={s:0. for s in snrs};remaining=3*fraction
    for s in order:
        rates[s]=min(1.,max(0.,remaining));remaining-=rates[s]
    return rates

pooled=[];risk=[]
for r in rows:
    seed=r['seed'];z=np.load(OUT/f'outcomes_s{seed}.npz')
    gain=z['base_loss']-z['refined_loss']
    val_gate=z['val_h_index']>=250
    vgain=z['val_base_loss']-z['val_refined_loss']
    valmeans={s:float(vgain[val_gate&(z['val_snr']==s)].mean()) for s in snrs}
    order=sorted(snrs,key=lambda s:valmeans[s],reverse=True)
    ratio=r['cost']['monitor_conv_linear_MACs']/r['cost']['refiner_conv_linear_MACs']
    for q in (.2,.5):
        call=np.zeros(len(gain),bool)
        for s in snrs:
            ix=z['snr']==s;call[ix]=z[f'gate_{s}_{q}']
        rate=float(call.mean());budget_rate=min(1.,rate+ratio)
        low=allocate(snrs,budget_rate);best=allocate(order,budget_rate)
        low_weights=np.array([low[int(s)] for s in z['snr']])
        best_weights=np.array([best[int(s)] for s in z['snr']])
        # Both controls allocate the same expected conv/linear monitoring budget;
        # allocation depends on SNR and validation mean gains, never test labels.
        assert abs(float(low_weights.mean())-budget_rate)<1e-12
        assert abs(float(best_weights.mean())-budget_rate)<1e-12
        added=r['cost']['monitor_conv_linear_MACs']+rate*r['cost']['refiner_conv_linear_MACs']
        all_total=r['cost']['base_decoder_MACs']+r['cost']['refiner_conv_linear_MACs']
        full_total=r['cost']['base_decoder_MACs']+added
        p={'seed':seed,'validation_target_call_fraction':q,'actual_call_fraction':rate,
           'quality_gate_gain_db':float((gain*call).mean()),
           'random_same_calls_gain_db':float(gain.mean()*rate),
           'random_same_MAC_gain_db':float(gain.mean()*budget_rate),
           'SNR_low_first_same_MAC_gain_db':float((gain*low_weights).mean()),
           'SNR_validation_gain_order_same_MAC_gain_db':float((gain*best_weights).mean()),
           'SNR_validation_order':order,'SNR_validation_mean_gains':valmeans,
           'low_first_probabilities':low,'validation_order_probabilities':best,
           'mean_added_MACs':added,'total_decoder_MAC_saving_relative_always_refine':1-full_total/all_total,
           'baseline_type':'analytical expected randomized controls matched observed quality-gate compute; not full deployed resource controller'}
        pooled.append(p)
    for s in snrs:
        ix=z['snr']==s
        vi=(z['val_snr']==s)&(z['val_h_index']<250)
        cut=float(np.quantile(z['val_base_loss'][vi],.8))
        call=z[f'gate_{s}_0.2'];q=call.mean()
        base_bad=z['base_loss'][ix]>cut
        refined_bad=z['refined_loss'][ix]>cut
        gate_loss=np.where(call,z['refined_loss'][ix],z['base_loss'][ix])
        random_same_mac_rate=min(1.,q+ratio)
        random_bad=(1-random_same_mac_rate)*base_bad+random_same_mac_rate*refined_bad
        risk_difference=(gate_loss>cut).astype(float)-random_bad
        hh=z['h_index'][ix]
        rng=np.random.default_rng(99100+seed*100+s)
        boot_weights=rng.multinomial(600,np.ones(600)/600,size=1000)
        grouped=np.bincount(hh,weights=risk_difference,minlength=600)
        boot_risk=boot_weights@grouped/len(hh)
        risk.append({'seed':seed,'snr':s,'bad_nmse_db_threshold':cut,
                     'baseline_bad_fraction':float(base_bad.mean()),
                     'all_refined_bad_fraction':float(refined_bad.mean()),
                     'quality_gate_bad_fraction':float((gate_loss>cut).mean()),
                     'random_same_calls_expected_bad_fraction':float(((1-q)*base_bad+q*refined_bad).mean()),
                     'random_same_MAC_expected_bad_fraction':float(random_bad.mean()),
                     'quality_minus_random_same_MAC_bad_fraction':float(risk_difference.mean()),
                     'quality_minus_random_same_MAC_H_CI95':np.quantile(boot_risk,[.025,.975]).tolist()})
summary={'pooled':pooled,'relative_bad_risk':risk,'audit':'PASS',
         'note':'Three source-model seeds, same shared refiner architecture; SNR controls added during analysis, all rules retained.'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')

lines=['# Q2 最小实验：按需启用一个共享残差修正网络','',
'日期：2026-11-13。三个源模型种子全部完成。原编码器、星座映射、功率标定、共享解码器与旧质量头保持不变。',
'',
'**结果：额外修正网络本身有小幅收益；现有质量头按“最差优先”启用它，没有稳定优于同计算预算随机策略。当前尚未证明质量监测可以有效分配精修计算。**',
'',
'## 1. 设置与成本','',
'- 有限 16-QAM、AWGN、Ns=64、SNR=0/10/20 dB、合成 CSI，当前 RD-AE 改编骨干；不是官方 RD-JSCC 或扩散精修复现。',
'- 一个共享修正 CNN：输入初步重建 2 通道、已有 decoder AF 特征 16 通道和广播条件 2 通道；20→8→2 的两层 3×3 卷积，SiLU，无 BN，最后一层零初始化。三个 SNR 使用同一套修正权重。',
'- 1594 参数（FP32 权重约 6.23 KiB），训练输入标准化另占 144 字节；约 1,622,016 卷积 MAC/次。没有新增 UE 网络、反馈符号或按条件的完整解码器库。',
'- 原 BS 质量头 24893 参数、24638 线性 MAC；若正式部署门控仍需计其存储与每样本推理。门控不会释放修正网络存储。',
'- 修正训练：训练集前 1200 H，每 SNR 一次独立噪声；最多 15 epoch，Adam 1e−3。验证前 250 H 选 epoch，允许 epoch0 恒等输出；后 250 H 设门限。既有质量头以前用过整500H选模型，本轮不是全新独立验证。',
'- 测试 600 H×3 次当前噪声，每条件完全配对。原质量头在修正前评分，验证集最可疑20%或50%为门限；测试逐次判断，不按测试池 top-k。实际调用比例单独报告。',
'- 预设停止条件：种子0若选择epoch0，停止其他种子和测试。实际种子0验证改善，故按原计划补三种子。未新增收益门控网络或搜索修正架构。',
'',
'## 2. 修正网络全部启用：它本身有没有用？','',
'均为三个源模型种子均值 ± 样本标准差。NMSE 改善是全部样本平均逐样本 NMSE_dB 的下降，正为好。','',
'| SNR | 基础 NMSE/dB | 全部修正 NMSE/dB | 全部修正改善/dB | 平方 SGCS 改善 | 修正后改善的观察占比 |',
'|---|---|---|---|---|---|']
for s in snrs:
    lines.append(f"| {s} dB | {pm(cell(s,'base_nmse_db'))} | {pm(cell(s,'refined_all_nmse_db'))} | {pm(cell(s,'all_gain_db'))} | {pm(cell(s,'all_sgcs_gain'),5)} | {pm(cell(s,'fraction_refinement_improves'),2,100)}% |")
lines += ['',
'这些改善仅证明本次增加一个小模块能修正部分已有解码误差，不证明引入了新信息、不证明修正所有坏样本，也不证明网络已充分优化。',
'',
'## 3. 验证门限20%：质量头能否选对样本？','',
'随机策略是期望收益：每个同SNR观察以相同概率独立启用。此时只依据 SNR 的策略无法在该条件内区分观察，等价于这种随机对照。',
'同 MAC 随机对照把质量头开销换成额外修正调用：随机调用率 = 质量策略实际调用率 + 24638/1622016，即约多1.52个百分点。只配平卷积/线性MAC，不冒充严格相同时延/能耗。','',
'| SNR | 测试调用率 | 质量门控改善/dB | 同调用次数随机/dB | 同MAC随机/dB | 质量门控−同MAC随机/dB |',
'|---|---|---|---|---|---|']
for s in snrs:
    lines.append(f"| {s} dB | {pm(gate(s,.2,'test_call_fraction'),2,100)}% | {pm(gate(s,.2,'base_to_policy_gain_db'))} | {pm(gate(s,.2,'random_same_calls_gain_db'))} | {pm(gate(s,.2,'random_same_conv_linear_compute_gain_db'))} | {pm(gate(s,.2,'quality_minus_random_compute_db'))} |")
lines += ['',
'### 50%门限对照（同时保留，未据测试结果选择门限）','',
'| SNR | 实际调用率 | 质量门控改善/dB | 同MAC随机/dB | 配对差/dB |',
'|---|---|---|---|---|']
for s in snrs:
    lines.append(f"| {s} dB | {pm(gate(s,.5,'test_call_fraction'),2,100)}% | {pm(gate(s,.5,'base_to_policy_gain_db'))} | {pm(gate(s,.5,'random_same_conv_linear_compute_gain_db'))} | {pm(gate(s,.5,'quality_minus_random_compute_db'))} |")
lines += ['',
'每种子的 H 聚类区间在 results_s*.json；不能把三次噪声当三个独立 H。真实 H 只用于监督与评价，不进入部署门控或修正输入。',
'',
'## 4. 跨SNR资源分配对照','',
'下面把三个 SNR 等权汇总。为比较同计算资源，SNR-only 策略用与质量策略实际成本相等的期望调用预算，不支付质量头开销，在同SNR内随机调用。',
'低SNR优先是最简单的 SNR 门限式基准；另一对照按门限验证集上的修正平均收益排列SNR。两者排序均不读取测试标签，所有结果同时保留。这是本轮普通CNN的对照，不是复现RD-JSCC原扩散的最佳门限。','',
'| 验证调用目标 | 实际质量门控调用率 | 质量门控改善/dB | 随机同MAC/dB | 低SNR优先同MAC/dB | 验证收益选SNR同MAC/dB |',
'|---|---|---|---|---|---|']
for q in (.2,.5):
    pp=[p for p in pooled if p['validation_target_call_fraction']==q]
    lines.append(f"| {q*100:.0f}% | {pm([p['actual_call_fraction'] for p in pp],2,100)}% | "+' | '.join(pm([p[k] for p in pp]) for k in ('quality_gate_gain_db','random_same_MAC_gain_db','SNR_low_first_same_MAC_gain_db','SNR_validation_gain_order_same_MAC_gain_db'))+' |')
lines += ['',
'资源匹配是离线评价对照，不能当作完整部署控制器。质量门控本身使用冻结门限逐次决策，不要求等一批CSI到齐。',
'',
'## 5. 标记为坏的样本有没有减少？','',
'固定相对坏样本标准为前250验证H的基础NMSE第80百分位。它是重建风险诊断，不是实际业务失败门限。','',
'| SNR | 基础坏样本比例 | 全部修正 | 质量门控20% | 同调用率随机期望 |',
'|---|---|---|---|---|']
for s in snrs:
    rr=[v for v in risk if v['snr']==s]
    lines.append(f"| {s} dB | "+' | '.join(pm([v[k] for v in rr],2,100)+'%' for k in ('baseline_bad_fraction','all_refined_bad_fraction','quality_gate_bad_fraction','random_same_calls_expected_bad_fraction'))+' |')
lines += ['',
'质量门控在这个相对门限下，坏样本比例的三种子均值略低于同调用率随机，差约0.17–0.23个百分点；但改用同MAC随机作配对比较，各条件都没有三个种子的H聚类区间一致支持严格优势。完整区间与同MAC风险见summary.json。这个小幅趋势不能解释成已解决坏样本处理。',
'',
'该表与平均NMSE分别解释，不能以其中一个指标掩盖另一指标的不利结果；仍未测吞吐、中断、BLER或真实预编码。',
'',
'## 6. 时间与总计算成本','',
'同一台 Ryzen 7 7735H，CPU单线程、FP32、batch1，7个随机交错计时块。每种子抽取每SNR前32次观察用于计时，不等同于全测试集的调用比例；原始时长与实际计时调用率保存在 timing.json。','',
'| 新增处理路径（不含基础解码器） | 同PC时间/ms |',
'|---|---|']
for name,label in [('monitor_only','质量监测'),('refiner_always','全部执行修正'),('quality_gate20_pipeline','质量监测＋条件分支＋选中时修正')]:
    lines.append(f"| {label} | {pm([v['timing'][name]['median_ms'] for v in timing['results']],4)} |")
pp=[p for p in pooled if p['validation_target_call_fraction']==.2]
lines += ['',
f"相对所有观察都运行修正器，20%门限策略的**基础解码器＋监测＋修正总卷积/线性MAC**减少约 {pm([p['total_decoder_MAC_saving_relative_always_refine'] for p in pp],2,100)}%。不能把少运行约80%的修正器说成整个接收器省80%计算；基础解码器仍约22.34M MAC。",
'',
'相对完全没有修正模块，选择性修正仍然增加计算与存储。计时不代表真实基站时延、UE功耗或多用户并发性能。',
'',
'## 7. 结论与停止位置','',
'1. 一套基础解码器＋一个共享小修正模块可以运行，无需UE重传或完整专用模型库。这个工程形式可行。',
'2. 本次修正器全部运行有小幅平均收益；但原质量分数选择最差样本，不具有跨种子、跨条件稳定的同计算资源优势。因此，暂时没有证明“监测使精修更有效”。',
'3. 相比此前固定重复发送，本轮已验证另一种BS本地动作；这不意味着当前动作已足够强、满足业务可靠性或具有方法新颖性。',
'4. 若要继续，下一步需要检验修正收益是否能从基础接收特征预测，而不是继续扩大质量报警比例。本次按限时最小实验停止，未训练额外收益门控网络。',
'5. 与RD-JSCC的联系是两阶段精修；后者已提出SNR门限切换。小CNN替代扩散、有限星座或逐样本开关本身都不直接构成新颖性证据。',
'',
'## 8. 文件与审计','',
'audit.json = PASS：源权重与代码哈希未变，门控分数复算、门限和策略收益复算、同MAC随机对照复算、修正前向复算最大损失差为0，参数与MAC核对。',
'',
'run_refinement.py / protocol.json 保存事先固定的最小方案；refiner_s*.pt 保存权重和训练标准化；outcomes_s*.npz 保存独立可重算的逐样本结果；results_s*.json 保存验证选择、配对区间与成本；benchmark.py / timing.json 保存实际分支计时。',
'',
'原始样本、已有质量头、主权重不覆盖。SNR分配和相对坏样本风险为完成训练后的补充分析，规则与结果全部保留，不把这些事后分析说成预注册结论。',
]
(OUT/'选择性精修最小实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(10.7,4.4),layout='constrained')
xx=np.arange(3);width=.24
for offset,key,label,color in [(-1,'all_gain_db','Always refine','#2874A6'),(0,'base_to_policy_gain_db','Quality gate (val20%)','#D35400'),(1,'random_same_conv_linear_compute_gain_db','Random, same added MAC','#7F8C8D')]:
    values=[cell(s,key) if key=='all_gain_db' else gate(s,.2,key) for s in snrs]
    mean=np.array([np.mean(v) for v in values]);sd=np.array([np.std(v,ddof=1) for v in values])
    ax[0].bar(xx+offset*width,mean,width,yerr=sd,capsize=3,label=label,color=color)
ax[0].set(xticks=xx,xticklabels=['0','10','20'],xlabel='Feedback SNR (dB)',ylabel='Mean per-sample NMSE improvement (dB)',title='Refinement works; selective benefit is limited')
ax[0].legend(fontsize=8);ax[0].axhline(0,color='black',lw=.7)
values=[gate(s,.2,'quality_minus_random_compute_db') for s in snrs]
means=np.array([np.mean(v) for v in values]);sd=np.array([np.std(v,ddof=1) for v in values])
ax[1].errorbar(snrs,means,yerr=sd,fmt='o-',color='#D35400',capsize=4)
for i,r in enumerate(rows):
    ax[1].scatter(snrs,[r['conditions'][str(s)]['gates']['0.2']['quality_minus_random_compute_db'] for s in snrs],alpha=.45,s=20)
ax[1].axhline(0,color='black',lw=.8,ls='--')
ax[1].set(xticks=snrs,xlabel='Feedback SNR (dB)',ylabel='Quality gate minus random (dB)',title='Same added conv/linear MAC budget')
fig.suptitle('One shared residual CNN: 1,594 parameters, no extra feedback\nThree frozen backbone seeds; error bars are source-model seed SD',fontsize=12)
fig.savefig(OUT/'selective_refinement.png',dpi=180);fig.savefig(OUT/'selective_refinement.pdf');plt.close(fig)
print('\n'.join(lines[20:47]))
print('POOLED',json.dumps(pooled,ensure_ascii=False))
