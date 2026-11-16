"""Matched continuation and KD fine-tuning from the same trained RD-AE.

All source models/data are read only. Protocol is written before any training.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments' / 'rdae_q1_supplement'
spec = importlib.util.spec_from_file_location('q1_source', SOURCE / 'run_supplement.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

def dump(name, value):
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(p)

def load(name):
    obj = torch.load(SOURCE / 'checkpoints' / f'{name}.pt', weights_only=True, map_location='cpu')
    net = old.LinkRDAE(cond=False)
    net.load_state_dict(obj['state_dict'])
    return net.eval()

def save(name, net, meta):
    torch.save({'state_dict': net.state_dict(), 'metadata': meta}, OUT / f'{name}.pt')

def val(net, x, seed):
    scores = old.validation(net, x, seed)
    return {'by_snr': scores, 'equal_snr_mean_nmse_db': float(np.mean(list(scores.values())))}

def train(arm, args, source, teachers, xtr, xval, schedule):
    net = copy.deepcopy(source)
    assert old.state_hash(net) == old.state_hash(source)
    # Reset Adam in both arms; source checkpoint contains no optimizer state.
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    ordergen = old.scalar_seed(args.seed, 193000)
    noisegen = old.scalar_seed(args.seed, 194000)
    baseline_val = val(net, xval, args.seed + 1000)
    best_metric = baseline_val['equal_snr_mean_nmse_db']
    best_epoch = 0
    save(f'{arm}_best', net, {'arm': arm, 'epoch': 0, 'validation': baseline_val})
    trace = [{'epoch': 0, 'validation': baseline_val}]
    started = time.perf_counter()
    for ep in range(args.epochs):
        net.train()
        permutation = torch.randperm(len(xtr), generator=ordergen)
        sums = np.zeros(3)
        for step, (anchor, snr) in enumerate(schedule[ep]):
            xb = xtr[permutation[step * args.bs:(step + 1) * args.bs]]
            eps = torch.randn((len(xb), 64, 2), generator=noisegen)
            pred, _, _ = old.forward(net, xb, snr, eps)
            rec = (pred - xb).square().mean()
            kd = torch.zeros(())
            if arm == 'distill' and anchor is not None:
                with torch.no_grad():
                    target, _, _ = old.forward(teachers[anchor], xb, snr, eps)
                kd = (pred - target).square().mean()
            loss = rec + (args.lam * kd if arm == 'distill' else 0.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += [float(rec.detach()), float(kd.detach()), float(loss.detach())]
        scheduler.step()
        check = val(net, xval, args.seed + 1000)
        row = dict(epoch=ep + 1, reconstruction_mse=sums[0] / len(schedule[ep]),
                   kd_mse=sums[1] / len(schedule[ep]), objective=sums[2] / len(schedule[ep]),
                   validation=check, elapsed_seconds=time.perf_counter() - started)
        trace.append(row)
        if check['equal_snr_mean_nmse_db'] < best_metric:
            best_metric = check['equal_snr_mean_nmse_db']
            best_epoch = ep + 1
            save(f'{arm}_best', net, {'arm': arm, 'epoch': best_epoch, 'validation': check})
        dump(f'trace_{arm}.json', trace)
        dump('progress.json', {'stage': 'training', 'arm': arm, **row})
        print('TRAIN', arm, 'epoch', ep + 1, 'val', check['by_snr'],
              'seconds', round(row['elapsed_seconds'], 1), flush=True)
    save(f'{arm}_final', net, {'arm': arm, 'epoch': args.epochs, 'best_val_epoch': best_epoch})
    dump(f'selection_{arm}.json', {'best_epoch': best_epoch, 'best_val_metric': best_metric})
    return net

def load_output(name):
    obj = torch.load(OUT / f'{name}.pt', weights_only=True, map_location='cpu')
    net = old.LinkRDAE(cond=False)
    net.load_state_dict(obj['state_dict'])
    return net.eval(), obj['metadata']

def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'shared_original': 'Original shared', 'continue_final': 'Ordinary fine-tuning',
              'distill_final': 'KD fine-tuning', 'specialist': 'Specialist'}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, label in labels.items():
        xs = sorted(int(k) for k in results[name])
        ax.plot(xs, [results[name][str(s)]['mean_nmse_db'] for s in xs], marker='o', label=label)
    ax.set(xlabel='Feedback SNR (dB)', ylabel='Mean per-sample NMSE (dB)',
           title='RD-AE, fixed 64 complex symbols, one training seed')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'finetune_comparison.png', dpi=180)
    fig.savefig(OUT / 'finetune_comparison.pdf')
    plt.close(fig)

def report(results, contrasts, selections):
    lines = ['# RD-AE：从共享检查点开始蒸馏微调', '',
             '固定 Ns=64、16-QAM、AWGN，单训练种子；与此前从零蒸馏属于不同实验。', '',
             '## 固定协议', '',
             '- 普通微调和蒸馏微调都加载同一个已训练40轮的共享模型；全网权重参与更新，BN训练统计量正常更新。',
             '- 两组都额外训练10轮；Adam重置，初始学习率2e-4，余弦降至2e-5，batch128；数据顺序、SNR和标准噪声相同。',
             '- 一半batch是0/10/20 dB锚点，另一半逐样本U(0,20)；λ=1，蒸馏只在锚点加入，教师冻结eval。',
             '- 教师/学生共享标准噪声抽样，但各自按实际发射功率定噪声；保持原蒸馏目标，不按测试结果改λ。',
             '- 主结果固定最后一轮；附加报告按验证集三个SNR等权NMSE选出的最佳检查点，允许epoch0。',
             '- 每个模型用训练集校准测试发射功率；不追加BN重标定，以隔离微调作用。',
             '- 测试H600:1000，三个配对噪声，0/5/10/15/20 dB；历史用过的测试H，不是独立外部确认集。', '',
             '## 主结果：固定额外10轮', '',
             '| SNR | 原共享 | 普通微调 | 蒸馏微调 | 专用 | 蒸馏相对原共享改善 | 蒸馏相对普通微调改善 |',
             '|---|---|---|---|---|---|---|']
    for s in old.GRID:
        k = str(s)
        row = [results[n][k]['mean_nmse_db'] for n in ('shared_original', 'continue_final', 'distill_final')]
        teacher = results['specialist'].get(k, {}).get('mean_nmse_db')
        lines.append('| ' + k + ' | ' + ' | '.join(f'{v:.4f}' for v in row)
                     + ' | ' + ('—' if teacher is None else f'{teacher:.4f}')
                     + f' | {row[0]-row[2]:+.4f} | {row[1]-row[2]:+.4f} |')
    lines += ['', 'NMSE越低越好；改善为前后差，正数表示微调更好。专用只比较其训练锚点。', '',
              '## 蒸馏相对普通微调：按H配对区间', '',
              '| SNR | 改善/dB | 95%区间 |', '|---|---|---|']
    for s in old.GRID:
        r = contrasts[str(s)]['distill_final_vs_continue_final']
        lo, hi = r['H_paired_CI95']
        lines.append(f"| {s} | {r['improvement_db']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '上述区间按H聚类保留三组噪声相关性，只反映固定模型下测试样本不确定性，不能代替独立训练种子。', '',
              '## 验证选轮补充', '',
              f"普通微调最佳epoch={selections['continue']['epoch']}；蒸馏最佳epoch={selections['distill']['epoch']}。", '',
              '| SNR | 普通微调验证最佳 | 蒸馏验证最佳 |', '|---|---|---|']
    for s in old.GRID:
        lines.append(f"| {s} | {results['continue_best'][str(s)]['mean_nmse_db']:.4f} | {results['distill_best'][str(s)]['mean_nmse_db']:.4f} |")
    lines += ['', '## 解释边界', '',
              '- 只有一个教师/骨干训练种子、一个事前固定λ和学习率配置；不能据此否定全部蒸馏微调。',
              '- 教师和学生的发送映射不同，输出MSE蒸馏并不等于复制同一个接收映射。',
              '- 两组训练轮数相同，但蒸馏额外运行教师，因此训练计算量更高。推理不需要教师。',
              '- 本轮没有单独冻结BN，因此微调收益包含权重和BN统计量共同变化，不能全部归因于其中之一。',
              '- 原共享、教师、数据及原结果文件不覆盖；无测试调参，无多种λ搜索。', '',
              '产物：protocol.json、两组训练轨迹、最终/验证最佳检查点、results.json、test_arrays.npz、audit.json、对比图。']
    (OUT / '微调实验结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lam', type=float, default=1.)
    ap.add_argument('--bs', type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(8)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Output directory already contains a protocol; do not overwrite completed runs.')
    protected = [SOURCE / 'checkpoints' / f'{n}.pt' for n in ('shared', 'teacher_0', 'teacher_10', 'teacher_20')]
    protected += [old.PROJECT / f for f in ('model_rdae.py', 'model_v4.py', 'train_rdae.py')]
    protected += [old.PROJECT / 'data' / f'{s}.npz' for s in ('train', 'val', 'test')]
    hashes = {str(p): old.sha(p) for p in protected}
    dump('protocol.json', dict(args=vars(args),
         initialization='both arms load identical trained 40-epoch shared checkpoint; fresh Adam states',
         objective='MSE reconstruction; distill adds lambda1 output MSE on anchor batches only',
         primary_selection='fixed final10 epochs',
         secondary_selection='lowest equal-SNR validation mean NMSE; epoch0 allowed',
         calibration='train-only power calibration; original BN state after fine-tuning, no extra recalibration',
         data='train6000; validation first400; historical test600:1000,3 common noises; five SNR points',
         protected_sha256=hashes, source_script_sha256=old.sha(Path(__file__)),
         report='single source seed; H confidence intervals do not assess retraining uncertainty'))
    xtr, xval, xtest = old.data()
    source = load('shared')
    source_hash = old.state_hash(source)
    teachers = {s: load(f'teacher_{s}') for s in old.ANCHORS}
    teacher_hashes = {s: old.state_hash(t) for s, t in teachers.items()}
    for t in teachers.values():
        for p in t.parameters():
            p.requires_grad_(False)
    schedule = old.conditions(args.seed + 1000, args.epochs, len(xtr) // args.bs, args.bs)
    for arm in ('continue', 'distill'):
        train(arm, args, source, teachers, xtr, xval, schedule)
    models = {'shared_original': source}
    selections = {}
    for arm in ('continue', 'distill'):
        for kind in ('final', 'best'):
            name = f'{arm}_{kind}'
            models[name], meta = load_output(name)
            if kind == 'best':
                selections[arm] = meta
    results, arrays, powers = {}, {}, {}
    for name, net in models.items():
        power = old.power_calibration(net, xtr)
        powers[name] = power
        results[name] = {}
        for s in old.GRID:
            row, ar = old.observe(net, xtest[600:1000], s, power, args.seed)
            results[name][str(s)] = row
            arrays[f'{name}/{s}'] = ar
            print('EVAL', name, s, round(row['mean_nmse_db'], 5), flush=True)
    results['specialist'] = {}
    for s, net in teachers.items():
        power = old.power_calibration(net, xtr)
        row, ar = old.observe(net, xtest[600:1000], s, power, args.seed)
        results['specialist'][str(s)] = row
        arrays[f'specialist/{s}'] = ar
    contrasts = {}
    for s in old.GRID:
        contrasts[str(s)] = {}
        for a, b in [('continue_final', 'shared_original'), ('distill_final', 'shared_original'),
                     ('distill_final', 'continue_final'), ('distill_best', 'continue_best')]:
            contrasts[str(s)][f'{a}_vs_{b}'] = old.ci(arrays[f'{a}/{s}']['db'], arrays[f'{b}/{s}']['db'], args.seed + s)
    dump('results.json', dict(results=results, contrasts=contrasts, power=powers, validation_selections=selections))
    np.savez_compressed(OUT / 'test_arrays.npz', **{f'{k}/{metric}': value for k, row in arrays.items() for metric, value in row.items()})
    previous = json.loads((SOURCE / 'results.json').read_text(encoding='utf-8'))
    max_baseline_difference = max(abs(results['shared_original'][str(s)]['mean_nmse_db']
        - previous['results']['shared']['raw'][str(s)]['mean_nmse_db']) for s in old.GRID)
    assert max_baseline_difference < 1e-6, max_baseline_difference
    assert old.state_hash(source) == source_hash
    assert all(old.state_hash(t) == teacher_hashes[s] for s, t in teachers.items())
    assert all(old.sha(Path(p)) == h for p, h in hashes.items())
    # Independent metric recomputation from saved per-observation outputs.
    max_metric_error = 0.
    for name, rows in results.items():
        for s, row in rows.items():
            ar = arrays[f'{name}/{s}']
            max_metric_error = max(max_metric_error, abs(float(ar['db'].mean()) - row['mean_nmse_db']))
            np.testing.assert_allclose(ar['db'], 10 * np.log10(ar['ratio'] + 1e-12), atol=2e-6)
    dump('audit.json', dict(status='PASS', source_files_unchanged=True, frozen_teacher_states_unchanged=True,
         original_shared_state_unchanged=True, original_baseline_reproduction_max_db_error=max_baseline_difference,
         saved_metric_max_error=max_metric_error))
    report(results, contrasts, selections)
    plot(results)
    dump('progress.json', {'stage': 'complete'})
    print('COMPLETE', str(OUT), flush=True)

if __name__ == '__main__':
    main()
