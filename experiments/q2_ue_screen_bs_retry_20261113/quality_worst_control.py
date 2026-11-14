"""Posthoc missing strong control: UE-worst-first for the BAD-RATE objective.

Added after observing secondary bad-risk results. No fitting/threshold tuning;
uses the identical eligible cohort and frozen UE proxy quality scores. This
must not be described as a preregistered primary comparison.
"""
from pathlib import Path
import json
import numpy as np
import torch
import run_screen_retry as run

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
records = []
for seed in range(3):
    result = json.loads((ROOT / f'results_s{seed}.json').read_text(encoding='utf8'))
    z = np.load(ROOT / f'decisions_s{seed}.npz')
    rows = dict(snr=torch.from_numpy(z['snr']), h_index=torch.from_numpy(z['h_index']))
    quality = torch.from_numpy(z['ue_quality'])
    passed = torch.from_numpy(z['passed'])
    flags = {}
    for s in run.SNRS:
        mask = (rows['snr'] == s) & passed
        cell = result['fresh_test'][str(int(s))]
        k = cell['matched_retry_count']
        w = run.rank_weights(rows, quality, mask, k, True)[mask]
        # Separately verify the H-level ranking without the policy helper.
        h = rows['h_index'][mask]
        ids = h.unique(sorted=True)
        q = quality[mask]
        qh = torch.stack([q[h == i][0] for i in ids])
        selected = ids[torch.argsort(qh, descending=True, stable=True)[:k // 3]]
        expected = torch.isin(h, selected).float()
        torch.testing.assert_close(w, expected, rtol=0, atol=0)
        assert abs(float(w.sum()) - k) < 1e-6
        current = torch.from_numpy(z['current_loss'])[mask]
        future = torch.from_numpy(z['future_loss'])[mask]
        cut = result['bad_thresholds'][str(s)]
        b = (current > cut).float()
        rb = (future > cut).float().mean(1)
        gain = current - future.mean(1)
        wb = torch.from_numpy(z[f'flag_{int(s)}_bs_conditional_quality_worst'])
        costs = {}
        for flag_cost, request_cost in ((0, 1), (1, 1), (1, 4)):
            available = (12.8 - flag_cost) * int(mask.sum())
            wu = run.rank_weights(rows, quality, mask, available / 64, True)[mask]
            bscore = torch.from_numpy(z['score_bs_conditional_quality_worst'])
            wbs = run.rank_weights(rows, bscore, mask, available / (64 + request_cost), False)[mask]
            costs[f'passflag{flag_cost}_request{request_cost}'] = dict(
                UE_bad_fraction=float((b + wu * (rb - b)).mean()),
                BS_bad_fraction=float((b + wbs * (rb - b)).mean()),
                BS_minus_UE_bad=run.ci(((wbs - wu) * (rb - b)).numpy(), h.numpy(), 297000 + seed * 100 + int(s)),
                expected_UE_retries=float(wu.sum()), expected_BS_retries=float(wbs.sum()))
        records.append(dict(seed=seed, snr=s,
                            UE_quality_worst_gain_db=float((w * gain).mean()),
                            UE_quality_worst_bad_fraction=float((b + w * (rb - b)).mean()),
                            BS_quality_worst_bad_fraction=cell['policies']['bs_conditional_quality_worst']['bad_fraction'],
                            random_bad_fraction=cell['policies']['random']['bad_fraction'],
                            abstract_control_cost_sensitivity=costs,
                            BS_quality_worst_minus_UE_quality_worst_bad=run.ci(((wb - w) * (rb - b)).numpy(), h.numpy(), 299000 + seed * 100 + int(s)),
                            BS_quality_worst_minus_UE_quality_worst_gain=run.ci(((wb - w) * gain).numpy(), h.numpy(), 298000 + seed * 100 + int(s))))
        flags[f'UE_worst_{int(s)}'] = w.numpy()
    np.savez_compressed(ROOT / f'UE_worst_control_s{seed}.npz', **flags)
run.dump(ROOT / 'UE_worst_control.json', dict(posthoc=True, description=__doc__, records=records,
                                             audit='Unique-H ranking, noise-invariant decision and exact matched counts independently checked'))
lines = ['', '## 补充强对照：降低坏样本率时，UE也应按差样本选择', '',
'这是查看次指标后发现缺少的对照，明确标为事后补充；不训练模型、不更改门限、不搜索权重，使用同一冻结UE代理、同一通过者和同样追加次数。不能将其称为事前主实验。', '',
'BS按当前质量最差挑，在0dB的坏样本率低于随机，这一结果必须保留。但降低坏样本率时，UE也应比较“预计最差优先”，不能只拿优化平均NMSE的UE收益优先当作强基线。', '',
'| SNR | 随机最终坏比例 | UE预计最差优先 | BS当前最差优先 |', '|---|---|---|---|']
for s in run.SNRS:
    rr = [r for r in records if r['snr'] == s]
    vals = [np.mean([r[key] for r in rr]) for key in ('random_bad_fraction', 'UE_quality_worst_bad_fraction', 'BS_quality_worst_bad_fraction')]
    lines.append(f'| {int(s)} | ' + ' | '.join(f'{100*v:.2f}%' for v in vals) + ' |')
lines += ['', '0dB：BS当前最差优先相对随机的坏比例下降，在三个种子的H聚类95%区间均为负；但相对UE预计最差优先，三个种子的坏比例均更高，其中两个配对区间完全为正。后者仍未构成BS优于UE的证据。10/20dB差距更小，完整配对区间见UE_worst_control.json。', '',
'额外通过标识0/1和BS请求1/4符号当量的同预算结果也保存在UE_worst_control.json；与主实验一样，它是抽象资源敏感性，不代表真实控制链路。', '',
'这说明需要按动作目标选对基线：本轮平均NMSE上UE收益优先较强；坏样本率上UE预计最差优先较强。不能把某个BS策略在次指标上优于目标不匹配的UE策略写成BS独有观测带来的优势。', '']
addition = '\n'.join(lines)
for name in ('UE预检后BS重传_结果.md', '实验记录追加.md'):
    p = ROOT / name
    content = p.read_text(encoding='utf8')
    content = content.partition('\n## 补充强对照：降低坏样本率时')[0].rstrip() + '\n'
    p.write_text(content + '\n' + addition, encoding='utf8')
print(json.dumps(records, ensure_ascii=False, indent=2))
