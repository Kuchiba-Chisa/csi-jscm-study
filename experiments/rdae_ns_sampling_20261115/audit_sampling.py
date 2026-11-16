"""Replay completed sampling ablation; no training or test selection."""
import sys
sys.dont_write_bytecode = True
import copy
import json
from pathlib import Path
import numpy as np
import torch
import run_sampling as R

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(8)
out = Path(__file__).resolve().parent
protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
result = json.loads((out / 'results.json').read_text(encoding='utf-8'))
arrays = np.load(out / 'observations.npz')
for p, expected in protocol['source_sha256'].items():
    assert R.sha(p) == expected
assert R.sha(out / 'run_sampling.py') == protocol['script_sha256']
xte = R.load01('test', root=str(R.PROJECT / 'data'))
replay_error = 0.
gains = {}
for arm in R.ARMS:
    obj = torch.load(out / f'{arm}_final.pt', weights_only=True, map_location='cpu')
    net = R.LinkRDAE().eval()
    net.load_state_dict(obj['state_dict'])
    for ns in R.LENGTHS:
        for snr in R.SNRS:
            key = f'ns{ns}_snr{snr}'
            power = result['powers'][arm][str(ns)][str(snr)]
            target = copy.deepcopy(net)
            state = torch.load(out / arm / 'bn_states' / f's0_ns{ns}_snr{snr}.pt', weights_only=True)
            all_state = target.state_dict()
            all_state.update(state)
            target.load_state_dict(all_state)
            assert all(torch.equal(v, dict(net.named_parameters())[k]) for k, v in target.named_parameters())
            assert R.U.tensor_hash(net, 'enc.') == R.U.tensor_hash(target, 'enc.')
            for mode, model in (('S0', net), ('S2', target)):
                row, ar = R.U.evaluate(model, xte, snr, ns, power, R.SEED, R.BS, 3)
                for metric, values in ar.items():
                    stored = arrays[f'{arm}/{key}/{mode}/{metric}']
                    replay_error = max(replay_error, float(np.max(np.abs(stored - values))))
                    np.testing.assert_allclose(values, stored, atol=1e-7, rtol=0)
                saved = result['results'][arm][key][mode]
                assert abs(row['mean_nmse_db'] - saved['mean_nmse_db']) < 1e-7
                assert row['y_sha256'] == saved['y_sha256']
            a = arrays[f'{arm}/{key}/S0/db']
            b = arrays[f'{arm}/{key}/S2/db']
            gains[f'{arm}/{key}'] = float((a - b).mean())
            assert abs(gains[f'{arm}/{key}'] - result['results'][arm][key]['S0_minus_S2_db']) < 1e-6
            print('REPLAY', arm, key, 'PASS', flush=True)
traces = {arm: json.loads((out / f'trace_{arm}.json').read_text(encoding='utf-8')) for arm in R.ARMS}
assert all([r['epoch'] for r in trace] == list(range(1, 41)) for trace in traces.values())
assert all(a['length_counts'] == b['length_counts'] for a, b in zip(traces['batch_ns'], traces['sample_ns']))
check = dict(status='PASS', source_hashes_unchanged=True, final_epoch40_both_arms=True,
             matched_length_counts_every_epoch=True, replay_max_error=replay_error,
             replay='2 training arms x9 conditions x2 BN states x1000H x3 noises',
             recalibrated_parameters_and_encoder_unchanged=True)
(out / 'independent_audit.json').write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(check, ensure_ascii=False), flush=True)
