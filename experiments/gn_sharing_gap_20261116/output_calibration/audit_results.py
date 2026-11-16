from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/'results.json').read_text(encoding='utf-8'))
a=np.load(root/'observations.npz')
db_error=0.;sgcs_error=0.
for name,cells in r['results'].items():
    for snr,row in cells.items():
        for mode in ('raw','scale_only','affine'):
            db=a[f'{name}__snr{snr}__{mode}__db']
            sc=a[f'{name}__snr{snr}__{mode}__sgcs']
            assert db.shape==sc.shape==(3,1000)
            db_error=max(db_error,abs(float(db.astype(np.float64).mean())-row[mode]['mean_nmse_db']))
            sgcs_error=max(sgcs_error,abs(float(sc.astype(np.float64).mean())-row[mode]['sgcs']))
# Torch and NumPy reduction order differs in FP32; compare with double sums.
assert db_error<1e-6 and sgcs_error<3e-7,(db_error,sgcs_error)
for p,h in r['protocol']['source_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
assert hashlib.sha256((root/'affine_gn.py').read_bytes()).hexdigest()==r['protocol']['script_sha256']
out={'status':'pass','all_saved_metrics_recomputed':True,'source_and_checkpoints_unchanged':True,
     'double_reduction_max_difference_db':db_error,'double_reduction_max_difference_sgcs':sgcs_error,
     'raw_replay_max_error_db':r['audit']['raw_max_per_observation_replay_error_db'],
     'scale_only_SGCS_max_difference':r['audit']['scale_only_SGCS_max_absolute_difference']}
(root/'independent_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
