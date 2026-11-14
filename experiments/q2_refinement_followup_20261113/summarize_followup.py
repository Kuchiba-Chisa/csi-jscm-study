"""Retain every fixed variant and condition; never choose methods by test score."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_followup import cluster_ci

ROOT = Path(__file__).resolve().parent
rows = [json.loads(p.read_text(encoding='utf8')) for p in sorted(ROOT.glob('results_s*.json'))]
assert rows
audit = json.loads((ROOT / 'audit.json').read_text(encoding='utf8'))
assert audit['status'] == 'PASS'
protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))


def pm(vals, digits=4):
    x = np.asarray(vals)
    suffix = f' ± {x.std(ddof=1):.{digits}f}' if len(x) > 1 else '（单种子）'
    return f'{x.mean():.{digits}f}' + suffix


summary = []
for row in rows:
    z = np.load(ROOT / f"outcomes_s{row['seed']}.npz")
    uniform_gain = z['base_loss'] - z['uniform_refined_loss']
    for variant in ('uniform', 'tail_weighted'):
        cells = row['conditions'][variant]
        cross = []
        paired = np.zeros(len(z['snr']))
        cross_paired = np.zeros_like(paired)
        for s in (0, 10, 20):
            ix = z['snr'] == s
            rate = cells[str(s)]['rate']
            rr = min(1., rate + row['cost']['monitor_MACs'] / row['cost']['refiner_MACs'])
            cross.append(cells[str(s)]['gate_gain_db'] - float(uniform_gain[ix].mean() * rr))
            gain = z['base_loss'][ix] - z[f'{variant}_refined_loss'][ix]
            call = z[f'{variant}_gate_{s}']
            paired[ix] = gain * (call - rr)
            cross_paired[ix] = gain * call - uniform_gain[ix] * rr
        summary.append(dict(seed=row['seed'], variant=variant,
                            selected_on_validation=row['selected_before_test'] == variant,
                            pooled_gate_gain_db=float(np.mean([v['gate_gain_db'] for v in cells.values()])),
                            pooled_random_same_MAC_gain_db=float(np.mean([v['random_same_MAC_gain_db'] for v in cells.values()])),
                            pooled_delta_db=float(np.mean([v['gate_minus_random_same_MAC_db'] for v in cells.values()])),
                            pooled_delta_H_CI95=cluster_ci(paired, z['h_index'], 671000 + row['seed']),
                            pooled_gate_minus_uniform_random_db=float(np.mean(cross)),
                            pooled_gate_minus_uniform_random_H_CI95=cluster_ci(cross_paired, z['h_index'], 672000 + row['seed'])))

(ROOT / 'summary.json').write_text(json.dumps(dict(by_seed=summary), indent=2), encoding='utf8')
lines = ['# Q2 后续：坏样本优先精修与随机分配的比较', '',
'本轮目标是在相同计算预算下，检验“监测判差→BS 本地精修”是否具有收益。不是保证正结果，也不是官方 RD-JSCC 复现。', '',
'## 固定设置', '',
'- 冻结原有限16-QAM RD-AE改编链路和BS质量头；Ns=64、AWGN、SNR=0/10/20 dB。一个跨SNR共享修正器，无额外UE计算、反馈、专用模型库。',
'- 修正器使用基础重建、原解码器16通道特征、SNR，以及真实收到的128维含噪y；y先映射为4×8×8特征，再插值至32×32，和空间特征拼接。三层3×3卷积24→16→16→2，SiLU，无BN，末层零初始化。',
f"- 参数 {rows[0]['cost']['parameters']}，FP32权重 {rows[0]['cost']['weights_bytes']/1024:.2f} KiB；约 {rows[0]['cost']['refiner_MACs']/1e6:.3f}M卷积/线性MAC，另计质量头24638MAC。MAC不含插值、标准化、激活与内存访问。",
'- 比较固定的两个训练方案：uniform（逐样本NMSE_dB平均）与tail_weighted（同一目标中，质量头判为同SNR训练分数最差40%的样本权重为5，其余为1）。两个方案初始化、训练数据与batch顺序相同。',
'- 6000训练H，每个H随机分配一个SNR并生成一次噪声；每SNR约2000H，共享45轮上限、8轮早停、Adam+余弦学习率。不是每H经历三个SNR或无限新噪声训练。',
'- val前250H按各方案自己的训练目标选epoch；val后250H确定最差20%质量门限并在测试前锁定方案；所有方案均保留并报告。已有质量头曾用过这500个valH，因此不是新的独立验证集。',
'- 本次test为H600:1000×3次全新噪声；它与上一轮精修的前600H不重叠，但不保证没被整个项目其他历史实验用过。因此不是外部数据复现。',
'- 主比较是同一修正器的差样本门控 vs 同MAC随机；还比较差样本加权修正器 vs 普通修正器随机分配，防止只是牺牲其他样本制造相对优势。测试真值仅用于评分，不参与门控。',
'- 正数改善表示平均逐样本NMSE_dB下降。随机收益为相同调用概率下的解析期望，不是随机跑一次。三次噪声不当作三个独立训练种子。', '',
'## 逐条件结果', '',
'| 训练 | SNR | 全部精修收益/dB | 最差门控收益/dB | 同MAC随机收益/dB | 门控−随机/dB |',
'|---|---|---|---|---|---|']
for variant in ('uniform', 'tail_weighted'):
    for s in (0, 10, 20):
        values = [r['conditions'][variant][str(s)] for r in rows]
        lines.append(f"| {variant} | {s} | " + ' | '.join(pm([v[key] for v in values]) for key in
                     ('all_refinement_gain_db', 'gate_gain_db', 'random_same_MAC_gain_db', 'gate_minus_random_same_MAC_db')) + ' |')
lines += ['', '## 等权汇总三个SNR', '',
'| 训练 | 门控收益/dB | 同MAC随机收益/dB | 门控−同自身随机/dB | 门控−普通修正器随机/dB |',
'|---|---|---|---|---|']
for variant in ('uniform', 'tail_weighted'):
    vals = [r for r in summary if r['variant'] == variant]
    lines.append(f'| {variant} | ' + ' | '.join(pm([v[k] for v in vals]) for k in
                 ('pooled_gate_gain_db', 'pooled_random_same_MAC_gain_db', 'pooled_delta_db', 'pooled_gate_minus_uniform_random_db')) + ' |')
lines += ['', '这是预设的等权SNR汇总，不表示真实网络流量分布。单独条件仍必须报告，不能用汇总掩盖某个SNR的负结果。', '',
'## SGCS与坏样本风险', '',
'坏样本门限取val前250H的基础重建误差80%分位数，固定后用于test；不是经过业务验证的中断门限。', '',
'| 训练 | SNR | 门控SGCS改善 | 同MAC随机SGCS改善 | 原坏比例 | 门控坏比例 | 同MAC随机坏比例 |',
'|---|---|---|---|---|---|---|']
for variant in ('uniform', 'tail_weighted'):
    for s in (0, 10, 20):
        vals = [r['conditions'][variant][str(s)] for r in rows]
        lines.append(f'| {variant} | {s} | ' + ' | '.join(pm([v[k] for v in vals], 5) for k in
                     ('gate_sgcs_gain', 'random_same_MAC_sgcs_gain', 'base_bad_rate', 'gate_bad_rate', 'random_same_MAC_bad_rate')) + ' |')
lines += ['', '## 逐种子、验证选择与不确定性', '']
for r in rows:
    lines.append(f"- 种子{r['seed']}：测试前验证选择={r['selected_before_test']}；两个训练方案的最佳epoch=" +
                 ', '.join(f"{v}:{r['training'][v]['selected_epoch']}" for v in ('uniform', 'tail_weighted')) + '。')
    for variant in ('uniform', 'tail_weighted'):
        for s in (0, 10, 20):
            c = r['conditions'][variant][str(s)]
            lines.append(f"  - {variant}, {s} dB：调用率{100*c['rate']:.2f}%；门控−随机H聚类95%区间{c['delta_H_CI95']}；选中观察精修后变差比例{100*c['fraction_selected_harmed']:.2f}%。")
lines += ['', '区间按H聚类保留三次噪声相关性；跨种子均值±标准差不能单独视为统计显著性证明。', '',
'## 边界与审计', '',
'1. 加权训练针对某些样本分配模型能力，可能损害其他样本；因此必须同时看全精修和普通训练对照。',
'2. 本轮同时改变了模型容量、y输入、训练样本量、目标和时长；与旧1594参数版本比较不能用于单因素归因。两个新版本之间才控制了结构、数据与优化预算。',
'3. 只试两个事前固定方案；不按测试表现继续修改权重或结构。若优势只在某个SNR或某个种子出现，必须保留该限制。',
'4. 没有测试吞吐、中断、真实预编码、异分布信道，不能宣称已获得系统部署收益或论文级新颖性。',
'5. audit.json=PASS：重建新噪声观察、重跑修正前向、门控、预算和收益；源网络状态/文件保持不变。', '',
'详细逐样本输出见outcomes_s*.npz，事前设置见protocol.json，训练轨迹与验证选择见calibration_s*.json与results_s*.json。']
(ROOT / '坏样本精修后续结果.md').write_text('\n'.join(lines) + '\n', encoding='utf8')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for variant, label in [('uniform', 'Uniform training'), ('tail_weighted', 'Tail-weighted training')]:
    vals = np.array([[r['conditions'][variant][str(s)]['gate_minus_random_same_MAC_db'] for s in (0, 10, 20)] for r in rows])
    axes[0].errorbar([0, 10, 20], vals.mean(0), yerr=vals.std(0, ddof=1) if len(rows) > 1 else None, marker='o', capsize=3, label=label)
    gain = np.array([[r['conditions'][variant][str(s)]['gate_gain_db'] for s in (0, 10, 20)] for r in rows])
    axes[1].plot([0, 10, 20], gain.mean(0), marker='o', label=label)
axes[0].axhline(0, color='black', linewidth=.8)
axes[0].set_ylabel('Quality gate minus same-MAC random (dB)')
axes[1].set_ylabel('Absolute gated refinement gain (dB)')
for ax in axes:
    ax.set_xlabel('Feedback SNR (dB)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / 'refinement_followup.png', dpi=180)
fig.savefig(ROOT / 'refinement_followup.pdf')
fig2, ax = plt.subplots(figsize=(7, 4.5))
for variant, key, label in [
    ('uniform', 'base_bad_rate', 'Base decoder'),
    ('uniform', 'random_same_MAC_bad_rate', 'Uniform refiner: same-MAC random'),
    ('uniform', 'gate_bad_rate', 'Uniform refiner: quality gate'),
    ('tail_weighted', 'gate_bad_rate', 'Tail-weighted refiner: quality gate'),
]:
    arr = np.array([[r['conditions'][variant][str(s)][key] * 100 for s in (0, 10, 20)] for r in rows])
    ax.errorbar([0, 10, 20], arr.mean(0), yerr=arr.std(0, ddof=1) if len(rows) > 1 else None,
                marker='o', capsize=3, label=label)
ax.set_xlabel('Feedback SNR (dB)')
ax.set_ylabel('Exceeding fixed validation threshold (%)')
ax.set_title('Bad-sample fraction: mean and SD across source seeds')
ax.grid(alpha=.25)
ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(ROOT / 'bad_sample_fraction.png', dpi=180)
fig2.savefig(ROOT / 'bad_sample_fraction.pdf')
print(json.dumps(summary, indent=2))
