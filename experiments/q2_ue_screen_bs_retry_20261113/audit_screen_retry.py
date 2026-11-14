"""Independent forward/decision/budget replay; checks selected future draws."""
from pathlib import Path
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import run_screen_retry as run

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))
assert run.sha(ROOT / 'run_screen_retry.py') == protocol['script_sha256']
tr, va, te = run.data()
torch.set_num_threads(4)
checks = []
for path in sorted(ROOT.glob('results_s*.json')):
    result = json.loads(path.read_text(encoding='utf8'))
    seed = result['seed']
    assert all(run.sha(Path(p)) == v for p, v in result['input_hashes'].items())
    net, source = run.build(seed, tr)
    assert source == result['source']
    net.eval()
    r = torch.load(ROOT / f'fresh_cache_s{seed}.pt', weights_only=True)
    f = torch.load(ROOT / f'fresh_future_s{seed}.pt', weights_only=True)
    z = np.load(ROOT / f'decisions_s{seed}.npz')
    captured = {}
    hook = net.dec.blocks[-1].af2.register_forward_hook(lambda m, a, o: captured.update(feature=F.adaptive_avg_pool2d(o, (2, 2)).flatten(1)))
    errors = []
    with torch.no_grad():
        for ix in torch.arange(len(r['snr'])).split(128):
            rec = net.dec(r['y'][ix], 10 ** (-r['snr'][ix, None] / 20))
            truth = te[600:1000][r['h_index'][ix]]
            errors.append(float((run.nmse_db_per(rec, truth) - r['target'][ix]).abs().max()))
            torch.testing.assert_close(captured['feature'], r['feature'][ix], rtol=1e-5, atol=1e-6)
    hook.remove()
    assert max(errors) < 1e-5
    calibration = {float(k): v for k, v in json.loads((run.PROXY / f'single_pass_s{seed}.json').read_text(encoding='utf8'))['calibration'].items()}
    pb = torch.load(run.PROXY / f'proxy_s{seed}.pt', weights_only=True)
    quality = run.proxy_quality(net, r, te[600:1000], pb, calibration)
    np.testing.assert_allclose(quality.numpy(), z['ue_quality'], rtol=0, atol=1e-6)
    cuts = {float(k): float(v) for k, v in result['eligibility_thresholds'].items()}
    passed = run.eligible(r, quality, cuts)
    np.testing.assert_array_equal(passed.numpy(), z['passed'])
    for s in run.SNRS:
        for h in r['h_index'][r['snr'] == s].unique():
            use = (r['snr'] == s) & (r['h_index'] == h)
            assert bool(torch.all(passed[use] == passed[use][0]))
            assert bool(torch.all(quality[use] == quality[use][0]))
    scores = {'ue_quality_best': -quality}
    for name in ('bs_unconditional_gain', 'bs_conditional_gain', 'bs_conditional_quality', 'ue_gain'):
        bundle = torch.load(ROOT / f'{name}_s{seed}.pt', weights_only=True)
        model = run.fit_arrays.__globals__['Monitor'](bundle['input_dim'], bundle['budget'])
        model.load_state_dict(bundle['state_dict']); model.eval()
        inputs = run.ue_x(r, quality) if name == 'ue_gain' else run.bs_x(r)
        assert inputs.shape[1] == (3 if name == 'ue_gain' else 194)
        key = name + '_worst' if name == 'bs_conditional_quality' else name
        scores[key] = run.predict(model, bundle, inputs)
        np.testing.assert_allclose(scores[key].numpy(), z['score_' + key], atol=1e-6, rtol=0)
    budget_errors = []
    for s in run.SNRS:
        use = (r['snr'] == s) & passed
        hh = r['h_index'][use].numpy()
        n = int(use.sum())
        current = r['target'][use].numpy()
        gain = current - f['loss'][use].numpy().mean(1)
        cell = result['fresh_test'][str(int(s))]
        count = cell['matched_retry_count']
        for policy in run.POLICIES + ('random',):
            w = z[f'flag_{int(s)}_{policy}']
            assert abs(float(w.sum()) - count) < 1e-4
            assert abs(float(np.mean(w * gain)) - cell['policies'][policy]['gain_db']) < 1e-6
            if policy == 'random':
                np.testing.assert_allclose(w, count / n, atol=1e-7, rtol=0)
            elif policy.startswith('ue_'):
                for h in np.unique(hh):
                    assert np.all(w[hh == h] == w[hh == h][0])
            else:
                order = np.argsort(-scores[policy][use].numpy(), kind='stable')[:count]
                expected = np.zeros(n)
                expected[order] = 1
                np.testing.assert_array_equal(w, expected)
        for scenario in cell['control_cost_sensitivity'].values():
            for policy in run.POLICIES:
                err = abs(scenario[policy]['total_extra_equivalent_symbols_per_observation'] - 12.8)
                budget_errors.append(err)
                assert err < 2e-5
        # Match the original generation batch for two independent future draws.
        indices = torch.where(r['snr'] == s)[0][:128]
        sym, y = r['symbols'][indices], r['y'][indices]
        sigma = math.sqrt(net.p_ref_of(64, s)) * 10 ** (-s / 20) / math.sqrt(2)
        with torch.no_grad():
            for draw in (0, f['loss'].shape[1] - 1):
                g = torch.Generator().manual_seed(1011000000 + seed * 30000000 + int(s) * 1000000 + int(indices[0]) * 100 + draw)
                second = sym + torch.randn(sym.shape, generator=g) * sigma
                rec = net.dec((y + second) / 2, torch.full((len(indices), 1), 10 ** (-s / 20)))
                truth = te[600:1000][r['h_index'][indices]]
                torch.testing.assert_close(run.nmse_db_per(rec, truth), f['loss'][indices, draw], atol=1e-5, rtol=0)
    checks.append(dict(seed=seed, current_loss_max_replay_error=max(errors),
                       sampled_future_draws_replayed=[0, int(f['loss'].shape[1] - 1)],
                       UE_scores_and_flags_noise_invariant=True, BS_scores_and_topk_replayed=True,
                       max_equivalent_budget_error=max(budget_errors), original_hashes_unchanged=True))
    print('PASS', seed, checks[-1], flush=True)
run.dump(ROOT / 'audit.json', dict(status='PASS', checks=checks,
    limitation='Checks implementation, replay and inputs; does not validate a real control protocol, independent population generalization or a causal error decomposition.'))
