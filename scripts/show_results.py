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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(table(),end='')
