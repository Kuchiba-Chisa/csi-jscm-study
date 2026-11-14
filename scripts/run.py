"""Stage independent source copies; keep all new outputs outside tracked results."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]


def stage(run_dir):
    root = (REPO / run_dir).resolve()
    allowed = (REPO / 'runs').resolve()
    if root == allowed or not root.is_relative_to(allowed):
        raise SystemExit('--run-dir must name a child directory within this repository/runs/')
    sources = []
    for folder in ('simulation', 'experiments'):
        for source in (REPO / folder).rglob('*.py'):
            relative = source.relative_to(REPO)
            target = root / relative
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise SystemExit(f'Staged source differs: {relative}. Choose a new --run-dir.')
            sources.append((source, target))
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():shutil.copy2(source, target)
    return root


def generate(root, small):
    import numpy as np
    sys.path.insert(0, str(root / 'simulation'))
    from channel_data import gen_dataset
    counts=(256,64,64) if small else (6000,1000,1000)
    folder=root/'simulation/data'
    specs=list(zip(('train','val','test'),counts,(2024,7,11)))
    # Validate the whole existing set before creating anything.
    for split,n,seed in specs:
        path=folder/f'{split}.npz'
        if path.exists():
            with np.load(path) as saved:
                if saved['x'].shape!=(n,2,32,32):
                    raise SystemExit(f'{split}: existing data has a different size. Choose a new run directory.')
    folder.mkdir(parents=True,exist_ok=True)
    for split,n,seed in specs:
        path=folder/f'{split}.npz'
        if not path.exists():np.savez_compressed(path,x=gen_dataset(n,seed))
        print(f'{split}: {n} H -> {path.relative_to(REPO)}')
    (folder/'generation.json').write_text(json.dumps(dict(counts=counts,seeds=[2024,7,11],small=small),indent=2),encoding='utf-8')


def execute(root, script, arguments, threads):
    os.environ.setdefault('OMP_NUM_THREADS',str(threads))
    os.environ.setdefault('MKL_NUM_THREADS',str(threads))
    os.environ.setdefault('MPLBACKEND','Agg')
    import torch
    torch.set_num_threads(threads)
    os.chdir(root/'simulation')
    sys.path[:0]=[str(script.parent),str(root/'simulation')]
    sys.argv=[str(script)]+arguments
    runpy.run_path(str(script),run_name='__main__')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='runs/local')
    parser.add_argument('--threads',type=int,default=2)
    sub=parser.add_subparsers(dest='command',required=True)
    gen=sub.add_parser('generate',help='generate the original synthetic splits')
    gen.add_argument('--small',action='store_true',help='256/64/64 H; smoke only')
    train=sub.add_parser('train',help='isolated core training entry point',add_help=False)
    train.add_argument('model',choices=('bn','gn','stage-a','distill'))
    exp=sub.add_parser('experiment',help='run a research script; prerequisites vary',add_help=False)
    exp.add_argument('name',choices=sorted(p.name for p in (REPO/'experiments').iterdir() if p.is_dir()))
    exp.add_argument('script',help='a .py filename in that experiment directory')
    args,extra=parser.parse_known_args()
    if args.threads<1:parser.error('--threads must be positive')
    if extra and extra[0]=='--':extra=extra[1:]
    if args.command=='generate' and extra:parser.error(f'Unrecognized arguments: {extra}')
    root=stage(args.run_dir)
    if args.command=='generate':return generate(root,args.small)
    if args.command=='train':
        name={'bn':'train_rdae.py','gn':'train_rdae_gn.py','stage-a':'run_experiments.py','distill':'distill.py'}[args.model]
        script=root/'simulation'/name
    else:
        if Path(args.script).name!=args.script or not args.script.endswith('.py'):
            parser.error('script must be a plain .py filename')
        script=root/'experiments'/args.name/args.script
        if not script.is_file():parser.error('unknown script')
    if '--help' not in extra and '-h' not in extra:
        if not (root/'simulation/data/train.npz').is_file():
            raise SystemExit('Missing data. Run generate first or import your local assets.')
    print(f'Running {script.relative_to(root)} in {root.relative_to(REPO)}',flush=True)
    execute(root,script,extra,args.threads)


if __name__=='__main__':main()
