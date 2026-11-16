"""Read trained outputs and replay held-out observations without fitting."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import run_finetune as ft

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(8)
out = Path(__file__).resolve().parent
protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
result = json.loads((out / 'results.json').read_text(encoding='utf-8'))
archive = np.load(out / 'test_arrays.npz')
for p, expected in protocol['protected_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
assert ft.old.sha(out / 'run_finetune.py') == protocol['source_script_sha256']
original = ft.load('shared')
x = ft.old.data()[2][600:1000]
replay_error = 0.
parameter_changes = {}
for name in ('shared_original', 'continue_final', 'distill_final', 'continue_best', 'distill_best'):
    net = original if name == 'shared_original' else ft.load_output(name)[0]
    if name.endswith('_final'):
        changed = {prefix: sum(not torch.equal(v, dict(original.named_parameters())[k])
                    for k, v in net.named_parameters() if k.startswith(prefix)) for prefix in ('enc.', 'dec.')}
        assert all(changed.values())
        parameter_changes[name] = changed
    for snr in (0, 10, 20):
        row, arrays = ft.old.observe(net, x, snr, result['power'][name], protocol['args']['seed'])
        for metric, values in arrays.items():
            saved = archive[f'{name}/{snr}/{metric}']
            replay_error = max(replay_error, float(np.max(np.abs(saved - values))))
            np.testing.assert_allclose(saved, values, atol=1e-7, rtol=0)
        assert abs(row['mean_nmse_db'] - result['results'][name][str(snr)]['mean_nmse_db']) < 1e-7
    print('REPLAY', name, 'PASS', flush=True)

for arm in ('continue', 'distill'):
    trace = json.loads((out / f'trace_{arm}.json').read_text(encoding='utf-8'))
    assert [r['epoch'] for r in trace] == list(range(11))
    best = min(trace, key=lambda r: r['validation']['equal_snr_mean_nmse_db'])
    assert best['epoch'] == result['validation_selections'][arm]['epoch']
    if arm == 'continue':
        assert all(r['kd_mse'] == 0 for r in trace[1:])
    else:
        assert all(r['kd_mse'] > 0 for r in trace[1:])
audit = dict(status='PASS', unchanged_source_hashes=True, replay_max_error=replay_error,
             replay='all400H x3 noises x3 anchorSNR x5 checkpoints',
             trained_parameter_tensor_changes=parameter_changes,
             validation_epoch_selection_recomputed=True)
(out / 'independent_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False), flush=True)
