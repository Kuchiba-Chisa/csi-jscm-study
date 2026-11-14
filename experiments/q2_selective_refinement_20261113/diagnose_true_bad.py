"""Posthoc diagnostic; true-quality and realized-gain rankings use test truth.

These are diagnostic comparisons, never deployable policies or gate training.
All gains are means of per-observation NMSE_dB improvements over the whole set.
"""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
rows = []
for seed in range(3):
    z = np.load(root / f'outcomes_s{seed}.npz')
    for snr in (0, 10, 20):
        mask = z['snr'] == snr
        base = z['base_loss'][mask]
        gain = base - z['refined_loss'][mask]
        gate = z[f'gate_{snr}_0.2'].astype(bool)
        k = int(gate.sum())
        worst = np.argsort(base)[-k:]
        best_gain = np.argsort(gain)[-k:]
        rows.append(dict(seed=seed, snr=snr, count=len(base), calls=k,
                         predicted_bad_gain_db=float((gain * gate).mean()),
                         random_same_calls_gain_db=float(gain.mean() * k / len(base)),
                         true_bad_gain_db=float(gain[worst].sum() / len(base)),
                         hindsight_best_gain_db=float(gain[best_gain].sum() / len(base))))
means = {str(s): {key: float(np.mean([r[key] for r in rows if r['snr'] == s]))
                  for key in ('predicted_bad_gain_db', 'random_same_calls_gain_db',
                              'true_bad_gain_db', 'hindsight_best_gain_db')}
         for s in (0, 10, 20)}
result = dict(note=__doc__, by_seed=rows, three_seed_means=means)
(root / 'diagnose_true_bad.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(means, indent=2))
