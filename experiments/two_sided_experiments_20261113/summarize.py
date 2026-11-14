from pathlib import Path
import json
import numpy as np

O=Path(__file__).resolve().parent
def ms(v,d=3):
    a=np.asarray(v,float);return f'{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}' if len(a)>1 else f'{a[0]:.{d}f}'

q=[json.loads(p.read_text(encoding='utf8')) for p in sorted(O.glob('results_s*.json'))]
a=[json.loads(p.read_text(encoding='utf8')) for p in sorted(O.glob('action_results_s*.json'))]
protocol=json.loads((O/'protocol.json').read_text(encoding='utf8'))
lines=['# UE—BS 协同监测实验结果','',
       '范围：既有 RD-AE 改编骨干、有限 16-QAM、AWGN、Ns=64；不是官方完整 RD-JSCC 复现。',
       f'质量实验已完成种子：{[x["seed"] for x in q]}。动作实验已完成种子：{[x["seed"] for x in a]}。',
       '数值为独立源模型种子之间的均值 ± 样本标准差。重复噪声不是独立种子。','',
       '本轮结论：当前设置下存在可测的 UE/BS 质量信息互补，但在计入提示误码和符号成本后，未获得稳定的重传净收益。质量监测改善与资源分配收益必须分开。',
       '2 bit QPSK 提示使 MAE 在 0/10/20 dB 分别从约 0.594/0.880/0.937 降至 0.575/0.820/0.886 dB；扣开销后相对 BS 收益头的平均重传净改善为 −0.0090/−0.0010/+0.0002 dB。最后一项幅度很小且跨种子不稳定。',
       '0 dB、相同重传次数而未扣提示成本时，2 bit 融合相对 BS 收益头平均改善约 +0.0057 dB；支付提示后转为约 −0.0090 dB。当前限制来自收益与成本的比例，而非完全没有信息互补。','',
       '## 协议','',
       f'UE 头使用前 {protocol["args"]["nue"]} 个训练 H；BS 头使用随后 {protocol["args"]["nfusion"]} 个 H。验证/测试使用独立源文件，测试 {protocol["args"]["ntest"]} 个 H，每条件重复 {protocol["args"]["test_repeats"]} 次噪声。',
       'UE 头为约 10 万参数 MLP，输入已有编码器 af2 特征与原始 latent、已知 SNR/Ns；不输入当前噪声或实际重建。BS 头约 2.5 万参数，读取已有 decoder feature；另设约 10 万参数 BS-only 容量对照。',
       'UE 训练目标为同一 H/条件下 8 次独立噪声重建损失的均值。UE 头按验证集选择，融合头用与 UE 训练 H 不重合的数据训练。',
       '主网络固定；提示只进入监测器，不修改重建输出。质量表的“增益”是监测精度改善，不是 CSI 重建 NMSE 改善。','',
       '## 质量预测 MAE（dB，越低越好）','',
       '| 方法 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
names={'ue_only':'UE 浮点预测','bs_only':'BS 特征头 25k','bs_large':'BS 特征头 100k',
       'fusion_float':'双侧浮点提示（诊断）','fusion_float_shuffled':'浮点提示按 H 打乱（诊断）',
       'fusion_clean2':'双侧 2 bit，无误码','fusion_clean4':'双侧 4 bit，无误码',
       'fusion_noisy2':'双侧 2 bit，1 个 QPSK 符号','fusion_noisy4':'双侧 4 bit，1 个 16-QAM 符号'}
for name,label in names.items():
    if q:lines.append('| '+label+' | '+' | '.join(ms([x['quality'][name]['test'][s]['mae_db'] for x in q]) for s in ['0','10','20'])+' |')
lines+=['','## 坏样本识别 AUROC（越高越好）','',
        '每个 SNR 内，以测试最差 20% 为诊断标签；不是可部署阈值。',
        '| 方法 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
for name,label in names.items():
    if q:lines.append('| '+label+' | '+' | '.join(ms([x['quality'][name]['test'][s]['auroc_worst20'] for x in q]) for s in ['0','10','20'])+' |')
lines+=['','## 提示信道','',
        '2 bit 用 1 个 QPSK 复符号；4 bit 用 1 个 16-QAM 复符号。按 BS-fit 训练样本的类别频率标定提示平均能量，与主信号 P_ref 对齐。提示误码同时进入融合头的训练、验证和测试。无信道编码保护，未实现完整控制协议。',
        '| 提示 | SNR | 符号错误率 | 测试实测提示能量 / P_ref |','|---|---|---|---|']
for bits in ('2','4'):
    for s in ['0','10','20']:
        if q:lines.append(f'| {bits} bit | {s} | '+ms([x['hint'][bits]['stats']['test'][s]['symbol_error_rate'] for x in q])+' | '+ms([x['hint'][bits]['stats']['test'][s]['measured_energy_relative_to_main_Pref'] for x in q])+' |')
lines+=['','## 融合相对 BS-only 的 MAE 配对差','',
        '差值 = 融合 MAE − BS MAE，负为更好。95% 区间按 H 聚类，覆盖固定拟合模型下的测试样本波动；跨模型种子另报告。',
        '| 方法 | SNR | 种子间均值 ± SD | 各种子配对差及 H 聚类 CI95 |','|---|---|---|---|']
for name in ['fusion_float','fusion_noisy2','fusion_noisy4']:
    for s in ['0','10','20']:
        c=[x['paired'][name+'_minus_bs'][s] for x in q]
        if c:
            details='; '.join(f's{x["seed"]}: {cc["mae_a_minus_b_db"]:+.4f} [{cc["H_cluster_CI95"][0]:+.4f},{cc["H_cluster_CI95"][1]:+.4f}]' for x,cc in zip(q,c))
            lines.append(f'| {name} | {s} | '+ms([cc['mae_a_minus_b_db'] for cc in c],4)+' | '+details+' |')
if a:
    lines+=['','## 相同平均符号预算下的重传选择','',
            '动作固定为重复同一组 64 个复 QAM 符号、两次观测平均、仍使用原始 SNR 解码条件。未优化重传接收器，无 CSI 老化或完整 ACK/授权时延。',
            'BS-only 重传 20%；双侧为每个样本支付 1 个提示符号后，重传约 18.44%。取整使双侧不超过预算，余量不足 64 个符号。各策略主信号功率参考相同；实际总发射能量也在 JSON 中报告，不能将其称为逐样本精确等能量比较。',
            '下表是对全部样本平均逐样本 NMSE_dB 的改善，正为好，不是只在被选样本上平均。未来噪声仅供离线标签/评价；特权参照用独立两组未来噪声分别排序/评价。','',
            '| 策略 | 0 dB 改善 | 10 dB 改善 | 20 dB 改善 |','|---|---|---|---|']
    for name in a[0]['test']['0']['policies']:
        lines.append('| '+name+' | '+' | '.join(ms([x['test'][s]['policies'][name]['nmse_improvement_db'] for x in a],4) for s in ['0','10','20'])+' |')
    lines+=['','### 双侧相对 BS 收益头的净配对改善（正为好）','',
            '| 策略 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
    for name in ['fusion_float_gain','fusion_clean2_gain','fusion_noisy2_gain','fusion_noisy4_gain','ue_noisy2_gain']:
        lines.append('| '+name+' | '+' | '.join(ms([x['test'][s]['paired_net_gain_db'][name+'_minus_bs_gain']['mean'] for x in a],4) for s in ['0','10','20'])+' |')
    if all('equal_retransmit_count_DIAGNOSTIC_hint_not_charged' in x['test']['0'] for x in a):
        lines+=['','### 相同重传次数的排序诊断（未扣提示成本）','',
                '各策略都选择 20% 重传，仅用于分辨信息收益与提示成本；不可当成等资源系统结果。',
                '| 策略相对 BS 收益头 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
        for name in ['fusion_float_gain','fusion_clean2_gain','fusion_noisy2_gain','fusion_noisy4_gain','ue_noisy2_gain']:
            lines.append('| '+name+' | '+' | '.join(ms([x['test'][s]['equal_retransmit_count_DIAGNOSTIC_hint_not_charged'][name]['paired_minus_bs']['mean'] for x in a],4) for s in ['0','10','20'])+' |')
    lines+=['','### 双侧净改善的各种子配对区间','',
            '| 策略 | SNR | 各种子净改善及 H 聚类 CI95 |','|---|---|---|']
    for name in ['fusion_noisy2_gain','fusion_noisy4_gain']:
        for s in ['0','10','20']:
            parts=[]
            for x in a:
                c=x['test'][s]['paired_net_gain_db'][name+'_minus_bs_gain']
                parts.append(f's{x["seed"]}: {c["mean"]:+.4f} [{c["CI95"][0]:+.4f},{c["CI95"][1]:+.4f}]')
            lines.append(f'| {name} | {s} | '+'; '.join(parts)+' |')
    lines+=['','### 实际总发射能量差（双侧相对 BS，百分比）','',
            '平均符号预算配平不保证有限测试集合的实际发射能量完全相同。以下同时计初传、被选重传与提示，采用各自实际符号幅度。',
            '| 策略 | 0 dB | 10 dB | 20 dB |','|---|---|---|---|']
    for name in ['fusion_noisy2_gain','fusion_noisy4_gain']:
        lines.append('| '+name+' | '+' | '.join(ms([100*(x['test'][s]['policies'][name]['total_actual_tx_energy']/x['test'][s]['policies']['bs_gain']['total_actual_tx_energy']-1) for x in a],3) for s in ['0','10','20'])+' |')
lines+=['','## 判断边界','',
        '- MAE/AUROC 更好只支持质量监测更准确，不自动支持重传或吞吐收益。',
        '- 本次 UE-only 是编码器特征上的轻量预测头，未复现 Guo 的代理解码器方案；它未超过 BS 头不能用于推断 UE 侧监测整体不如 BS。',
        '- 打乱提示实验是固定融合头的输入干预，帮助诊断提示依赖，不是严密的信息论证明。',
        '- 浮点提示和无误码量化提示只作为诊断；不能据此宣称真实提示链路有效。',
        '- 目前只有 Ns=64、三个已知 SNR、合成 CSI 分布；未验证新场景、未知 SNR、压缩率泛化或实际蜂窝协议。',
        '- 增量参数、离线监督、UE 部署计算与模型版本同步都有成本；没有在线训练不等于无维护成本。',
        '- 本实验尚未开展 Q1 校准后监测迁移；若净资源收益不成立，先报告限制，不盲目扩展。',
        '- 重传选择是按条件汇总的离线平均资源约束，不是已实现的逐时隙多用户调度器。','',
        '## 复现','',
        '```powershell',
        "& 'simulation/.venv/Scripts/python.exe' -u run_cooperative.py --seeds 0 1 2",
        "& 'simulation/.venv/Scripts/python.exe' -u run_actions.py --seeds 0 1 2",
        'python summarize.py','```','',
        '脚本放在当前目录运行；重建模型和原始数据只读。protocol.json、各 results_s*.json、action_results_s*.json、权重和缓存保存在本目录。']
(O/'双侧协同实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(O/'双侧协同实验结果.md')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    snrs=[0,10,20]
    for name,label in [('bs_only','BS only'),('fusion_float','UE+BS float'),('fusion_noisy2','UE+BS 2-bit QPSK'),('fusion_noisy4','UE+BS 4-bit QAM')]:
        for ax,metric in zip(axs[:2],['mae_db','auroc_worst20']):
            v=np.array([[x['quality'][name]['test'][str(s)][metric] for s in snrs] for x in q])
            ax.errorbar(snrs,v.mean(0),yerr=v.std(0,ddof=1) if len(v)>1 else None,marker='o',label=label,capsize=3)
    axs[0].set_ylabel('Quality prediction MAE (dB)');axs[1].set_ylabel('AUROC (worst 20%)')
    if a:
        for name,label in [('bs_gain','BS only'),('fusion_noisy2_gain','UE+BS 2-bit QPSK'),('fusion_noisy4_gain','UE+BS 4-bit QAM')]:
            v=np.array([[x['test'][str(s)]['policies'][name]['nmse_improvement_db'] for s in snrs] for x in a])
            axs[2].errorbar(snrs,v.mean(0),yerr=v.std(0,ddof=1) if len(v)>1 else None,marker='o',label=label,capsize=3)
        axs[2].set_ylabel('NMSE improvement, matched symbols (dB)')
    else:axs[2].text(.5,.5,'Action evaluation pending',ha='center',va='center',transform=axs[2].transAxes)
    for ax in axs:
        ax.set_xlabel('Feedback SNR (dB)');ax.set_xticks(snrs);ax.grid(alpha=.25)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(O/'cooperative_summary.png',dpi=170);plt.close(fig)
except ImportError as e:print('Plot unavailable:',e)
