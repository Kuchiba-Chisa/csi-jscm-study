"""Prospective matched one-seed batch-vs-sample Ns training experiment."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PROJECT = Path(__file__).resolve().parents[2] / 'simulation'
sys.path.insert(0, str(PROJECT))
from model_rdae import LinkRDAE
from model_v4 import normalize_power, to_symbols, qam_map_ste, tx_power, from_symbols, pair_mask
from train_rdae import load01

spec = importlib.util.spec_from_file_location('unified_eval', ROOT / 'experiments/gn_bn_unified_20261114/run_comparison.py')
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)

SEED = 0
EPOCHS = 40
BS = 128
LENGTHS = (16, 32, 64)
SNRS = (0, 10, 20)
ARMS = ('batch_ns', 'sample_ns')

def dump(name, obj):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def generator(seed):
    return torch.Generator().manual_seed(seed)

def forward(net, x, snr, ns, eps):
    feat = (10 ** (-snr / 20)).view(-1, 1)
    sym = qam_map_ste(to_symbols(normalize_power(net.enc(x, feat), ns)))
    power = tx_power(sym, ns).detach().clamp_min(1e-6)
    sigma = (power / 2).sqrt() * 10 ** (-snr / 20)
    y = from_symbols(sym + eps * sigma[:, None, None]) * pair_mask(ns)
    return net.dec(y, feat)

def epoch_schedule(epoch, n):
    steps = n // BS
    order = torch.randperm(n, generator=generator(310000 + epoch))[:steps * BS]
    # Draw the same batch-level length frequencies as the old training family.
    batch_lengths = torch.tensor(LENGTHS)[torch.randint(3, (steps,), generator=generator(311000 + epoch))]
    length_pool = batch_lengths.repeat_interleave(BS)
    # Exact per-epoch count matching; random allocation to individual samples.
    sample_lengths = length_pool[torch.randperm(len(length_pool), generator=generator(312000 + epoch))]
    snr = torch.rand((steps, BS), generator=generator(313000 + epoch)) * 20
    eps = torch.randn((steps, BS, 64, 2), generator=generator(314000 + epoch))
    return order, length_pool.view(steps, BS), sample_lengths.view(steps, BS), snr, eps

def smoke():
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    net = LinkRDAE().eval()
    x = load01('train', root=str(PROJECT / 'data'))[:4]
    ns = torch.tensor([16, 32, 64, 16])
    snr = torch.tensor([0., 5., 10., 20.])
    torch.manual_seed(900)
    expected = net(x, snr, ns)
    torch.manual_seed(900)
    actual = forward(net, x, snr, ns, torch.randn(4, 64, 2))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.enc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.dec.parameters())
    _, a, b, _, _ = epoch_schedule(0, 6000)
    assert all(int((a == ns).sum()) == int((b == ns).sum()) for ns in LENGTHS)
    assert torch.all(a == a[:, :1])
    assert all(len(torch.unique(row)) == 3 for row in b)
    print('SMOKE PASS: mixed-length manual forward matches original; both ends gradients; exact length counts; within-batch mixing', flush=True)

def train(xtr):
    torch.manual_seed(SEED)
    base = LinkRDAE(cond=False)
    models = {arm: copy.deepcopy(base) for arm in ARMS}
    initial_hash = U.tensor_hash(base)
    assert all(U.tensor_hash(net) == initial_hash for net in models.values())
    dump('initialization.json', {'state_sha256': initial_hash, 'identical_both_arms': True})
    optimizers = {arm: torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5) for arm, net in models.items()}
    schedulers = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) for arm, opt in optimizers.items()}
    traces = {arm: [] for arm in ARMS}
    counts_all = {str(n): 0 for n in LENGTHS}
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        order, batch_ns, sample_ns, snrs, eps = epoch_schedule(epoch, len(xtr))
        assert all(torch.count_nonzero(batch_ns == n) == torch.count_nonzero(sample_ns == n) for n in LENGTHS)
        counts = {str(n): int((batch_ns == n).sum()) for n in LENGTHS}
        for n, count in counts.items():
            counts_all[n] += count
        for net in models.values():
            net.train()
        sums = dict.fromkeys(ARMS, 0.)
        for step in range(len(batch_ns)):
            xb = xtr[order[step * BS:(step + 1) * BS]]
            for arm, ns in (('batch_ns', batch_ns[step]), ('sample_ns', sample_ns[step])):
                pred = forward(models[arm], xb, snrs[step], ns, eps[step])
                loss = (pred - xb).square().mean()
                assert torch.isfinite(loss)
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                sums[arm] += float(loss.detach())
        for arm in ARMS:
            schedulers[arm].step()
            traces[arm].append(dict(epoch=epoch + 1, mse=sums[arm] / len(batch_ns),
                length_counts=counts, elapsed_seconds=time.perf_counter() - t0))
            dump(f'trace_{arm}.json', traces[arm])
        dump('progress.json', dict(stage='training', epoch=epoch + 1, epochs=EPOCHS,
            elapsed_seconds=time.perf_counter() - t0, mse={arm: sums[arm] / len(batch_ns) for arm in ARMS}))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            for arm in ARMS:
                torch.save({'state_dict': models[arm].state_dict(), 'epoch': epoch + 1,
                    'optimizer': optimizers[arm].state_dict(), 'scheduler': schedulers[arm].state_dict()},
                    OUT / f'{arm}_progress.pt')
        print('TRAIN', epoch + 1, '/', EPOCHS, 'batch_mse', round(sums['batch_ns'] / len(batch_ns), 6),
            'sample_mse', round(sums['sample_ns'] / len(batch_ns), 6),
            'seconds', round(time.perf_counter() - t0, 1), flush=True)
    for arm, net in models.items():
        net.eval()
        torch.save({'state_dict': net.state_dict(), 'epoch': EPOCHS, 'seed': SEED, 'arm': arm}, OUT / f'{arm}_final.pt')
    dump('matched_counts.json', {'length_counts_each_arm': counts_all, 'steps_each_arm': EPOCHS * (len(xtr) // BS),
        'batch_order_snr_standard_noise_identical': True, 'initialization_sha256': initial_hash})
    return models

def paired(a, b, seed):
    # Positive = method a has a smaller NMSE than method b.
    d = (b - a).mean(axis=0)
    rng = np.random.default_rng(seed)
    estimates = d[rng.integers(0, len(d), size=(1500, len(d)))].mean(axis=1)
    return {'improvement_db': float(d.mean()), 'H_cluster_CI95': np.quantile(estimates, [.025, .975]).tolist()}

def evaluate(models, xtr, xte):
    rows, arrays, powers = {}, {}, {}
    for arm, net in models.items():
        U.OUT = OUT / arm
        (U.OUT / 'bn_states').mkdir(parents=True, exist_ok=True)
        powers[arm] = U.calibrate_power(net, xtr, BS)
        rows[arm] = {}
        for ns in LENGTHS:
            for snr in SNRS:
                key = f'ns{ns}_snr{snr}'
                power = powers[arm][str(ns)][str(snr)]
                raw, ar = U.evaluate(net, xte, snr, ns, power, SEED, BS, 3)
                calibrated, changed = U.recalibrate_decoder(net, xtr, snr, ns, power, SEED, BS)
                target, at = U.evaluate(calibrated, xte, snr, ns, power, SEED, BS, 3)
                assert raw['y_sha256'] == target['y_sha256']
                assert raw['standard_noise_sha256'] == target['standard_noise_sha256']
                rows[arm][key] = {'S0': raw, 'S2': target,
                    'S0_minus_S2_db': raw['mean_nmse_db'] - target['mean_nmse_db'],
                    'changed_bn_buffer_count': len(changed)}
                arrays[(arm, key, 'S0')] = ar
                arrays[(arm, key, 'S2')] = at
                dump('partial_results.json', rows)
                dump('progress.json', {'stage': 'evaluating', 'arm': arm, 'condition': key})
                print('EVAL', arm, key, 'S0', round(raw['mean_nmse_db'], 5), 'S2', round(target['mean_nmse_db'], 5),
                    'gain', round(raw['mean_nmse_db'] - target['mean_nmse_db'], 5), flush=True)
    contrasts = {}
    for ns in LENGTHS:
        for snr in SNRS:
            key = f'ns{ns}_snr{snr}'
            contrasts[key] = {
                arm + '_recalibration': paired(arrays[(arm, key, 'S2')]['db'], arrays[(arm, key, 'S0')]['db'], ns + snr)
                for arm in ARMS}
            for mode in ('S0', 'S2'):
                contrasts[key]['sample_vs_batch_' + mode] = paired(arrays[('sample_ns', key, mode)]['db'],
                    arrays[('batch_ns', key, mode)]['db'], ns + snr)
    flat = {f'{arm}/{key}/{mode}/{metric}': v for (arm, key, mode), arr in arrays.items() for metric, v in arr.items()}
    np.savez_compressed(OUT / 'observations.npz', **flat)
    dump('results.json', {'results': rows, 'contrasts': contrasts, 'powers': powers,
        'scope': 'one retrained source seed; 40 final epochs; H1000 x3 noises; no test selection'})
    return rows

def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ns in zip(axes, LENGTHS):
        for arm, label, color in [('batch_ns', 'Batch-wise Ns', 'tab:blue'), ('sample_ns', 'Sample-wise Ns', 'tab:orange')]:
            for mode, style in [('S0', '--'), ('S2', '-')]:
                ys = [rows[arm][f'ns{ns}_snr{s}'][mode]['mean_nmse_db'] for s in SNRS]
                ax.plot(SNRS, ys, style, marker='o', color=color, label=f'{label}, {mode}')
        ax.set(title=f'Ns = {ns}', xlabel='Feedback SNR (dB)', xticks=SNRS)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean per-sample NMSE (dB)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Matched length counts, fixed 40 epochs, one training seed')
    fig.tight_layout()
    fig.savefig(OUT / 'sampling_comparison.png', dpi=180)
    fig.savefig(OUT / 'sampling_comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    matrices = [np.array([[rows[arm][f'ns{ns}_snr{s}']['S0_minus_S2_db'] for s in SNRS] for ns in LENGTHS]) for arm in ARMS]
    vmin, vmax = min(0., min(m.min() for m in matrices)), max(m.max() for m in matrices)
    for ax, matrix, title in zip(axes, matrices, ('Batch-wise Ns', 'Sample-wise Ns')):
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=vmin, vmax=vmax)
        ax.set(xticks=range(3), xticklabels=SNRS, yticks=range(3), yticklabels=LENGTHS,
               xlabel='Feedback SNR (dB)', ylabel='Ns', title=title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center',
                    color='white' if matrix[i,j] > (vmin + vmax) * .55 else 'black')
    fig.colorbar(im, ax=axes, label='BN recalibration improvement (dB)', shrink=.8)
    fig.suptitle('S0 - S2, positive = lower NMSE after recalibration; one seed')
    fig.savefig(OUT / 'recalibration_gain.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'recalibration_gain.pdf', bbox_inches='tight')
    plt.close(fig)

def report(rows):
    lines = ['# 反馈长度逐批次与逐样本采样：BN 重标定对照', '',
        '2026-11-15；固定16-QAM、RD-AE改编模型；一个训练种子、两组从相同初始化重训40轮。', '',
        '## 预先固定的比较', '',
        '- 两组SNR均逐样本U(0,20)，只改变反馈长度在batch内的分配方式；Ns取16/32/64。',
        '- 逐批次组每batch一个随机Ns；逐样本组每epoch打乱同一长度池，使每个样本随机分配长度、批内混合。',
        '- 每epoch两组各长度总次数严格一致；逐样本组是次数匹配的随机分配，不宣称严格独立同分布抽样。',
        '- 相同初始参数、CSI顺序、逐样本SNR、标准高斯噪声、优化器和轮数。Adam2e-3，wd1e-5，余弦40轮，batch128。',
        '- 采用既有实际功率链路，端到端训练；无独立Ns条件输入，保留两端已有SNR AF。',
        '- 测试前用train6000H标定各模型各条件实际发送功率；目标BN也仅用train6000H重统计。',
        '- test全部1000H×3配对噪声，固定0/10/20dB；历史测试集，非全新外部确认。',
        '- 主比较是新训练的两组之间，不将新模型与历史权重差异直接归因于采样。', '',
        '## 主结果', '',
        '| Ns | SNR | 逐批次S0 | 逐批次S2 | 重标定改善 | 逐样本S0 | 逐样本S2 | 重标定改善 |',
        '|---|---|---|---|---|---|---|---|']
    for ns in LENGTHS:
        for s in SNRS:
            key = f'ns{ns}_snr{s}'
            vals = []
            for arm in ARMS:
                r = rows[arm][key]
                vals += [r['S0']['mean_nmse_db'], r['S2']['mean_nmse_db'], r['S0_minus_S2_db']]
            lines.append(f'| {ns} | {s} | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    lines += ['', 'S0=训练保存的BN；S2=目标条件重标定。NMSE越低越好；改善=S0−S2，正数为好。', '',
        '## 如何解读', '',
        '先看逐样本组S0是否改善，再看其S0−S2是否仍存在，不能只看重标定收益变小就说采样更好。',
        '若逐样本训练使原始模型更好且重标定不再改善，支持优先改训练协议；若仍改善，说明批内混合未完全消除当前设置下的条件统计适配空间。',
        '这里还没有逐条件专用模型，不是完整的多长度共享代价测量。', '',
        '## 证据与边界', '',
        '- 每组重标定前后固定所有权重、编码器和功率，接收y哈希逐格相同；只更新解码器BN缓冲。',
        '- 两组训练与评估时间独立记录；主结果固定最后40轮，不根据test选轮或改采样。',
        '- 一个初始化/训练种子不能证明普遍规律；按H区间仅反映固定模型下测试样本不确定性。',
        '- 原权重、数据、源代码和旧结果未覆盖。协议及哈希见protocol.json，逐样本数组见observations.npz。']
    (OUT / '采样对照结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if '--smoke' in sys.argv:
        smoke()
        return
    torch.set_num_threads(8)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Refusing to overwrite prior protocol/results')
    protected = [PROJECT / f for f in ('train_rdae.py', 'model_rdae.py', 'model_v4.py')]
    protected += list((PROJECT / 'ckpt_rdae').glob('*.pt'))
    protected += [PROJECT / 'data' / f'{name}.npz' for name in ('train', 'val', 'test')]
    hashes = {str(p): sha(p) for p in protected}
    dump('protocol.json', {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BS, 'threads': 8,
        'arms': ARMS, 'snr': 'per-sample U(0,20), identical across arms',
        'ns': 'batch uniform draw vs randomly permuted per-epoch identical length pool; exact counts matched',
        'initialization': 'identical newly initialized weights; both arms retrained, no historical checkpoint comparison',
        'optimizer': 'Adam2e-3 weight_decay1e-5 cosine40',
        'selection': 'fixed final40, no validation/test-driven checkpoint or parameter selection',
        'evaluation': 'train6000 power/BN calibration; test1000H x3common noises; 3length x3SNR xraw/targetBN',
        'source_sha256': hashes, 'script_sha256': sha(__file__),
        'scope': 'single training seed; H intervals not retraining uncertainty; no specialist models'})
    xtr = load01('train', root=str(PROJECT / 'data'))
    xte = load01('test', root=str(PROJECT / 'data'))
    t0 = time.perf_counter()
    models = train(xtr)
    trained_seconds = time.perf_counter() - t0
    rows = evaluate(models, xtr, xte)
    assert all(sha(p) == h for p, h in hashes.items())
    dump('audit.json', {'status': 'PASS', 'original_files_unchanged': True,
        'paired_y_raw_target_identical': True, 'only_decoder_bn_buffers_changed': True,
        'length_counts_identical_per_epoch': True, 'training_seconds_both_arms': trained_seconds,
        'total_seconds_before_report': time.perf_counter() - t0})
    report(rows)
    figures(rows)
    dump('progress.json', {'stage': 'complete'})
    print('COMPLETE', OUT, flush=True)

if __name__ == '__main__':
    main()
