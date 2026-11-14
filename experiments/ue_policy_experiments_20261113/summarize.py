import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
rs=[json.loads(p.read_text(encoding='utf8')) for p in sorted(O.glob('results_s*.json'))]
def ms(v,d=4):
    a=np.array(v,float);return f'{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}' if len(a)>1 else f'{a[0]:.{d}f}'
def vals(name,s,metric='nmse_improvement_db'):return [r['test'][s]['policies'][name][metric] for r in rs]
lines=['# UE 单侧追加反馈决策：补充实验','',
       f'完成源模型种子：{[r["seed"] for r in rs]}。既有有限 16-QAM、AWGN、Ns=64、RD-AE 改编模型，600 个测试 H，每个条件 3 次配对当前噪声，复用同一组未来重传噪声。',
       '**结论：相同追加比例下，当前 UE 策略优于当前 BS 收益头，0 dB 的三个种子配对区间均支持这一差异。最差质量优先仍不如随机。**',
       '**新增诊断发现：UE 收益头在本设置下几乎等价于按代理预测质量“最好优先”。因此不能将收益归因于新增网络或主张新算法；一个简单排序规则就得到基本相同的结果。**',
       '主结果是离线、同资源池比较，不是已经实现跨用户调度或零信令部署。固定验证门限另测，实际测试资源超出 20%，必须按实际资源评价。','',
       '## 策略与信息边界','',
       '- UE-quality：冻结单次 Guo 式代理，使用干净发送码字和 UE 自身 H 评分，追加预测质量最差的样本。',
       '- UE-gain：轻量收益回归头只输入本地代理分数、SNR 和 Ns；训练标签是平均追加收益。对同 H/条件的训练噪声平均后拟合，部署不读取本次接收 y。',
       '- UE-quality-best：按同一代理分数选择预测质量最好的样本，无额外收益头训练；这是检查复杂模型是否必要的补充诊断。',
       '- BS-gain：沿用已训练 BS 特征收益头，读取本次带噪观测产生的特征。',
       '- 动作为重复完全相同的 64 个复符号、两次观测平均、保持原 SNR 输入解码器。UE 提前决定追加，BS 接收后决定重传；不建模两者控制时序的真实延迟差。',
       '- UE 直接使用本地分数，不向 BS 发送量化质量提示。追加样本标识、预留资源、授权等控制开销对所有策略均未建模，不能称为零控制开销方案。','',
       '## 固定 20% 追加预算：平均逐样本 NMSE_dB 改善','',
       '每个 SNR：UE 从 600 个独立 H 中选择 120 个，同 H 的三个当前噪声重复使用同一决定；BS 从 1800 次当前观测中选择 360 次。初传和追加均共 138240 个复符号，平均 76.8 个符号/次。',
       'UE 的精确 top20% 是离线资源池排序：不表示分散 UE 已能不交换任何信息而完成全局排名。','',
       '| 策略 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
names={'random_20_expected':'随机追加（期望）','old_ue_quality':'旧 UE 小质量头，最差优先',
       'ue_quality':'UE 代理，最差质量优先','ue_gain':'UE 预计收益优先',
       'ue_quality_best':'UE 代理，最好质量优先（简单诊断）','bs_quality':'BS 最差质量优先','bs_gain':'BS 预计收益优先'}
for name,label in names.items():
    if rs:lines.append('| '+label+' | '+' | '.join(ms(vals(name,s)) for s in ['0','10','20'])+' |')
lines+=['','## 配对差：UE 收益优先相对 BS 收益优先','',
        '以下正值为 UE 更好。区间按 H 聚类，对固定拟合模型和选择集合进行配对重采样；跨源模型种子 SD 单独报告。','',
        '| SNR | 三种子均值 ± SD | 各种子与 H 聚类 CI95 |','|---|---|---|']
for s in ['0','10','20']:
    ps=[r['test'][s]['paired']['bs_gain_minus_ue_gain'] for r in rs]
    if ps:
        details='; '.join(f's{r["seed"]}: {-p["mean"]:+.4f} [{-p["CI95"][1]:+.4f},{-p["CI95"][0]:+.4f}]' for r,p in zip(rs,ps))
        lines.append('| '+s+' | '+ms([-p['mean'] for p in ps])+' | '+details+' |')
lines+=['','## SGCS：同预算最终平均值','',
        '| 策略 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
for name in ['random_20_expected','ue_quality','ue_gain','ue_quality_best','bs_gain']:
    if rs:lines.append('| '+names[name]+' | '+' | '.join(ms(vals(name,s,'mean_sgcs')) for s in ['0','10','20'])+' |')
lines+=['','## 简单排序能否替代 UE 收益网络','',
        '收益头约 2.5 万参数，新增离线监督训练；最好质量优先只对已存在代理分数反向排序。比较以下配对差，正值才支持收益头额外有效。',
        '| SNR | UE 收益头 − 最好质量优先 / dB | 相同追加决定占比 |','|---|---|---|']
for s in ['0','10','20']:
    if rs:
        dif=[r['test'][s]['paired']['ue_gain_minus_quality_best']['mean'] for r in rs]
        agreement=[]
        for r in rs:
            a=np.load(O/f'decisions_s{r["seed"]}.npz');ix=a['snr']==int(s)
            agreement.append(float(np.mean(a['flag_ue_gain'][ix]==a['flag_ue_quality_best'][ix])))
        lines.append('| '+s+' | '+ms(dif,6)+' | '+ms(agreement,4)+' |')
lines+=['','## 固定验证门限：实际资源使用','',
        '各 UE 策略的门限只用 500 个验证 H 设置为约 20% 追加，测试时冻结，不根据测试标签重调。实际比例有漂移，所以下表不与固定 20% 的 BS 直接比；额外提供与实际 UE 追加次数相同的 BS 基线。','',
        '| SNR | UE 收益门限实际追加比例 | UE 改善 / dB | 同实际次数的 BS 改善 / dB |','|---|---|---|---|']
for s in ['0','10','20']:
    if rs:lines.append('| '+s+' | '+ms([100*v for v in vals('threshold_ue_gain',s,'actual_append_fraction')],2)+'% | '+ms(vals('threshold_ue_gain',s))+' | '+ms(vals('bs_matched_threshold_ue_gain',s))+' |')
lines+=['','## 实际发射能量对照','',
        '相同符号数不保证选中样本的实际 QAM 能量严格相同；下面为 UE 收益策略相对 BS 收益策略的总能量差。',
        '| SNR | 总能量差 / % |','|---|---|']
for s in ['0','10','20']:
    if rs:lines.append('| '+s+' | '+ms([100*(u/b-1) for u,b in zip(vals('ue_gain',s,'actual_total_tx_energy'),vals('bs_gain',s,'actual_total_tx_energy'))],3)+' |')
lines+=['','## 如何解释','',
        '1. 当前结果不支持“BS 后验观测必然比 UE 发送前判断更适合资源决策”。UE 有真实 H，信息不被 BS 接收观测包含。这里只证明当前实现的比较，不是两侧信息能力的理论排序。',
        '2. 质量最差不等于追加收益最大。本设置下较好的样本反而更值得重复发送；这与“差样本中存在重复发送无法消除的压缩/模型误差”相容，但未做严格误差分解，不能当作已证实机制。',
        '3. UE 收益头几乎只是学会反向质量排序。因此不能把新增网络称为方法创新；简单规则足够解释这轮改善。',
        '4. Guo 的工作强调质量监测和反馈策略选择，不能据此声称“质量驱动追加反馈首次提出”。本轮也没有复现其官方 Transformer/数据/控制协议。',
        '5. 该比较固定重复动作、当前解码条件和分布；新解码器、增量冗余、信道老化、多用户干扰下排序可能改变。',
        '6. 单次代理需约 11.2 万参数、4.68M conv/linear MAC，源模型冻结，不需在线训练。收益头若被简单规则替代，其额外训练与推理可省，但代理本身仍有成本。',
        '7. 验证门限在测试集出现超预算，部署仍需资源跟踪/接入控制。未解决此点前，不能把离线 top20% 当作已实现的 UE 自主协议。',
        '8. 高 SNR 下所有策略的绝对追加收益很小；小数点后几千分之一的差异不能当作明显部署价值。','',
        '## 复现','',
        '```powershell',
        "& 'simulation/.venv/Scripts/python.exe' -u run_ue_policy.py --seeds 0 1 2",
        'python summarize.py','```','',
        '需在脚本目录运行。依赖同工作区的 Guo 代理实验与双侧实验缓存，不是独立代码包。所有旧模型和结果只读。']
(O/'UE单侧决策补充结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(O/'UE单侧决策补充结果.md')
if rs:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.8));ss=['0','10','20']
    for name,label,color in [('random_20_expected','Random','gray'),('ue_quality','UE worst quality','tab:orange'),
            ('bs_gain','BS gain','tab:blue'),('ue_gain','UE gain','tab:green'),('ue_quality_best','UE best quality','tab:red')]:
        v=np.array([vals(name,s) for s in ss]);axs[0].errorbar([0,10,20],v.mean(1),yerr=v.std(1,ddof=1) if len(rs)>1 else None,label=label,color=color,marker='o',capsize=3)
    v=np.array([[-r['test'][s]['paired']['bs_gain_minus_ue_gain']['mean'] for r in rs] for s in ss])
    axs[1].errorbar([0,10,20],v.mean(1),yerr=v.std(1,ddof=1) if len(rs)>1 else None,color='tab:green',marker='o',capsize=3)
    axs[1].axhline(0,color='gray',lw=1)
    for ax in axs:ax.set_xlabel('Feedback SNR (dB)');ax.set_xticks([0,10,20]);ax.grid(alpha=.25)
    axs[0].set_ylabel('NMSE improvement, 20% repeats (dB)');axs[1].set_ylabel('UE gain minus BS gain (dB)');axs[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(O/'ue_policy_comparison.png',dpi=170);plt.close(fig)
