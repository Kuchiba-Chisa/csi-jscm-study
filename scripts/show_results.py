"""Print frozen BN/GN summary JSON. No model weights or training required."""
import argparse
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def table():
    obj=json.loads((REPO/'results/gn_bn_unified_20261114/results.json').read_text(encoding='utf-8'))
    lines=['| Ns | SNR/dB | BN-S0 | BN-S2 | GN | GN − BN-S2/dB |',
           '|---|---|---|---|---|---|']
    for ns in (16,32,64):
        for snr in (0,10,20):
            cell=obj['summary'][f'ns{ns}_snr{snr}']
            vals=[cell[m]['mean_nmse_db']['mean'] for m in ('BN_S0','BN_S2','GN')]
            a,b,g=vals
            lines.append(f'| {ns} | {snr} | {a:.4f} | {b:.4f} | {g:.4f} | {g-b:+.4f} |')
    lines.extend(['','三训练种子均值，mean(逐样本 NMSE_dB)，越低越好。',
                  '末列正值表示目标重标定 BN 均值更好，不代表每个种子同向或显著。',
                  '原始逐种子值、协议和区间见 results/gn_bn_unified_20261114/。'])
    return '\n'.join(lines)+'\n'


def recent():
    obj=json.loads((REPO/'results/gn_sharing_gap_20261116/results.json').read_text(encoding='utf-8'))
    cal=json.loads((REPO/'results/gn_sharing_gap_20261116/output_calibration/results.json').read_text(encoding='utf-8'))
    lines=['固定 Ns64 的 GN 共享/专用（单训练种子）',
           '| SNR/dB | GN shared | GN specialist | gap | gap after both affine |',
           '|---|---|---|---|---|']
    for snr in (0,10,20):
        s=str(snr)
        a=obj['results']['shared'][s]['mean_nmse_db']
        b=obj['results'][f'specialist_{snr}'][s]['mean_nmse_db']
        gap=cal['gaps'][s]['affine']['nmse_shared_minus_specialist']['mean']
        lines.append(f'| {snr} | {a:.4f} | {b:.4f} | {a-b:+.4f} | {gap:+.4f} |')
    lines += ['', '均为 mean(逐样本 NMSE_dB)；正差距表示共享较差。',
              'GN 无 BN 运行统计量，0dB 仍差约0.31dB；仿射基本无效。',
              '这是固定长度新实验，与旧多长度 BN/GN 表分开报告。',
              '其他新增实验与运行顺序见 docs/RECENT_EXPERIMENTS.md。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=('all','bn-gn','recent'),default='all')
    args=parser.parse_args()
    if args.section in ('all','bn-gn'):print(table(),end='')
    if args.section in ('all','recent'):print('\n'+recent(),end='')
