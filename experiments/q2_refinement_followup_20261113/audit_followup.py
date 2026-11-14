"""Independently rebuild final observations and replay both frozen refiners."""
from pathlib import Path
import json
import numpy as np
import torch
import run_followup as run

root = Path(__file__).resolve().parent
protocol = json.loads((root / 'protocol.json').read_text(encoding='utf8'))
assert run.sha(root / 'run_followup.py') == protocol['script_sha256']
torch.set_num_threads(4)
tr, va, te = run.data()
torch.set_num_threads(4)
checks = []
for path in sorted(root.glob('results_s*.json')):
    result = json.loads(path.read_text(encoding='utf8'))
    seed = result['seed']
    z = np.load(root / f'outcomes_s{seed}.npz')
    net, source = run.build(seed, tr)
    assert source == result['source']
    test = run.make_cache(net, te[600:1000], seed, 'test')
    head, hb = run.score_head(seed)
    scores = run.predict(head, hb, run.head_inputs(test))
    np.testing.assert_allclose(scores.numpy(), z['quality'], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(test['h_index'].numpy(), z['h_index'])
    np.testing.assert_allclose(test['base_loss'].numpy(), z['base_loss'], atol=1e-6, rtol=0)
    validation = json.loads((root / f'calibration_s{seed}.json').read_text(encoding='utf8'))
    assert validation['selected_before_test'] == result['selected_before_test']
    for variant in run.VARIANTS:
        bundle = torch.load(root / f'{variant}_s{seed}.pt', weights_only=True)
        model = run.Refiner(bundle['width'])
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        output = run.evaluate(model, test, bundle['norm'])
        error = float(np.max(np.abs(output['loss'].numpy() - z[f'{variant}_refined_loss'])))
        assert error < 1e-5
        assert sum(p.numel() for p in model.parameters()) == result['cost']['parameters']
        cuts = {int(s): float(c) for s, c in result['gate_cuts'].items()}
        bad = {int(s): float(c) for s, c in result['bad_cuts'].items()}
        cells, arrays = run.policies(test, scores, output, cuts, bad, seed, result['cost'])
        for s in run.SNRS:
            np.testing.assert_array_equal(arrays[f'gate_{s}'], z[f'{variant}_gate_{s}'])
            for key in ('gate_gain_db', 'random_same_MAC_gain_db', 'gate_minus_random_same_MAC_db',
                        'gate_bad_rate', 'random_same_MAC_bad_rate'):
                assert abs(cells[str(s)][key] - result['conditions'][variant][str(s)][key]) < 1e-7
        checks.append(dict(seed=seed, variant=variant, max_loss_replay_error=error,
                           gate_and_budget_replayed=True))
        print('PASS', seed, variant, flush=True)
run.dump(root / 'audit.json', dict(status='PASS', checks=checks,
                                  scope='Fresh-noise forward, gates, same-MAC expectation, parameters and source identities replayed. Does not establish external validity or audit all historical test use.'))
