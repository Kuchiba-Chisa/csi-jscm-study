"""Batch-one CPU latency for the actual monitor/conditional-refiner path."""
from pathlib import Path
import json
import time
import numpy as np
import torch
import run_followup as run

root = Path(__file__).resolve().parent
torch.set_num_threads(4)
tr, va, te = run.data()
torch.set_num_threads(4)
records = []
for path in sorted(root.glob('results_s*.json')):
    row = json.loads(path.read_text(encoding='utf8'))
    seed = row['seed']
    variant = row['selected_before_test']
    if variant == 'none':
        continue
    net, source = run.build(seed, tr)
    # Stratified small subset for timing, not the whole accuracy evaluation.
    cache = run.make_cache(net, te[600:608], seed, 'test')
    head, hb = run.score_head(seed)
    bundle = torch.load(root / f'{variant}_s{seed}.pt', weights_only=True)
    model = run.Refiner(bundle['width'])
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    X = run.head_inputs(cache)
    all_indices = list(torch.arange(len(cache['snr'])).split(1))
    cuts = {int(s): float(c) for s, c in row['gate_cuts'].items()}
    torch.set_num_threads(1)

    @torch.no_grad()
    def perform(mode):
        calls = 0
        for ix in all_indices:
            if mode == 'always':
                run.output(model, cache, bundle['norm'], ix)
                calls += 1
                continue
            score = head((X[ix] - hb['mean']) / hb['sd']) * hb['target_sd'] + hb['target_mean']
            if mode == 'conditional' and float(score) >= cuts[int(cache['snr'][ix])]:
                run.output(model, cache, bundle['norm'], ix)
                calls += 1
        return calls

    for mode in ('monitor', 'always', 'conditional'):
        perform(mode)
    raw = {mode: [] for mode in ('monitor', 'always', 'conditional')}
    calls = {}
    rng = np.random.default_rng(631000 + seed)
    for _ in range(7):
        for mode in rng.permutation(list(raw)):
            tick = time.perf_counter()
            calls[mode] = perform(mode)
            raw[mode].append((time.perf_counter() - tick) * 1000 / len(all_indices))
    records.append(dict(seed=seed, variant_selected_on_validation=variant, observations=len(all_indices),
                        milliseconds_per_observation=raw, timing_calls=calls,
                        median_ms={k: float(np.median(v)) for k, v in raw.items()}))
    torch.set_num_threads(4)
run.dump(root / 'timing.json', dict(records=records, device='same Ryzen7735H CPU, one thread, FP32, batch1',
                                   caveat='Additional receiver work only; excludes base decoding. Tiny stratified timing subset call rates differ from full accuracy test. Not a real BS latency/power measurement.'))
print(json.dumps(records, indent=2))
