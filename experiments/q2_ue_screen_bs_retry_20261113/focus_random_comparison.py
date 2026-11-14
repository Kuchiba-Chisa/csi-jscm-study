"""Restate the user's clarified question without changing protocol or results."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
rows = [json.loads(p.read_text(encoding='utf8')) for p in sorted(root.glob('results_s*.json'))]
assert len(rows) == 3
lines = ['# 问题澄清：UE先筛选后，BS选择重传相对随机提升多少？', '',
'用户澄清的主问题是：固定同一UE预检规则，在相同通过样本中，BS根据实际接收y选择重传，是否比随机选择更有效。之前汇报过度强调“是否超过UE单侧”，该附加比较不能替代用户指定的随机对照。', '',
'原实验已经包含所需流程与随机对照，下面只重新组织已有结果；没有重训、改变筛选/动作门限或重新挑选数据。protocol.json保留原始设置。', '',
'UE门限来自验证集，排除预计最差约20%；通过者中匹配约20%额外完整重复次数。随机和BS方法使用相同通过者、相同当前/未来噪声、相同重复合并动作及同样追加次数。随机也可以由BS发请求，所以这项比较中相同每次请求开销是共同成本，不应套用“UE主动追加不需请求”的另一项比较。', '',
'三种子，新测试H600:1000×3组当前噪声、每观测16组未来噪声。表内是通过者整体的指标，非被选中样本均值。', '',
'## 平均NMSE：BS预测重传收益最大优先', '',
'| SNR | 随机重传改善/dB | BS选择改善/dB | BS额外改善/dB |', '|---|---|---|---|']
numbers = {}
for s in ('0', '10', '20'):
    c = [r['fresh_test'][s] for r in rows]
    random = float(np.mean([v['policies']['random']['gain_db'] for v in c]))
    selected = float(np.mean([v['policies']['bs_conditional_gain']['gain_db'] for v in c]))
    numbers[s] = dict(random_gain_db=random, BS_gain_db=selected, extra_gain_db=selected-random)
    lines.append(f'| {s} dB | {random:.5f} | {selected:.5f} | {selected-random:+.5f} |')
lines += ['', '均值上0dB约多改善0.0143dB，10dB约0.0025dB，20dB接近0。9个种子×SNR的H聚类95%区间均包含0，因此尚未建立平均NMSE的稳定额外收益。', '',
'## 减少坏反馈：BS预测当前质量最差优先', '',
'这是另一条固定的BS选择规则，不能把它的坏样本率与上一条收益头的NMSE改善混成同一种策略。坏门限固定为验证集中UE通过者基础NMSE的80%分位数，不是业务失效门限。', '',
'| SNR | 不重传坏比例 | 随机重传后 | BS按差选后 | 相对随机少多少个百分点 |', '|---|---|---|---|---|']
for s in ('0', '10', '20'):
    c = [r['fresh_test'][s] for r in rows]
    base = float(np.mean([v['base_passed_bad_fraction'] for v in c]))
    random = float(np.mean([v['policies']['random']['bad_fraction'] for v in c]))
    selected = float(np.mean([v['policies']['bs_conditional_quality_worst']['bad_fraction'] for v in c]))
    numbers[s].update(base_bad_fraction=base, random_bad_fraction=random, BS_worst_bad_fraction=selected,
                      bad_reduction_percentage_points=100*(random-selected),
                      relative_bad_reduction=(random-selected)/random)
    lines.append(f'| {s} dB | {100*base:.2f}% | {100*random:.2f}% | {100*selected:.2f}% | {100*(random-selected):.2f} |')
lines += ['', '**0dB：相对随机少约1.34个百分点，坏比例相对下降约11.1%；三个种子的配对区间均支持下降。**10dB只有均值趋势，三个区间均跨0；20dB没有跨种子稳定优势。', '',
'所以按用户指定的问题，已有证据支持“UE预检通过后，BS按差选择重传在低SNR下能比随机更有效地减少越限反馈”；但不支持“平均NMSE也稳定更好”，也不推广成中高SNR或真实业务失效率结论。', '',
'UE单侧的附加比较、事后UE质量最差对照、完整区间、SGCS与控制开销敏感性仍保留在原报告中；它们不改写这项相对随机的结果。', '',
'边界：被UE排除样本的回退未模拟；一次重复、原SNR输入解码；合成CSI、AWGN；未模拟真实控制误码、往返时延、老化或吞吐。', '']
(root / 'UE筛选后BS相对随机_主问题结果.md').write_text('\n'.join(lines), encoding='utf8')
(root / 'random_comparison_summary.json').write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(numbers, ensure_ascii=False, indent=2))
