"""Copy selected local data/checkpoints into an isolated run; never modify source."""
import argparse
import hashlib
from pathlib import Path
import shutil
from run import stage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True,help='local original simulation folder')
    p.add_argument('--run-dir',default='runs/replay')
    p.add_argument('--checkpoint-folders',nargs='*',default=['ckpt_rdae','ckpt_rdae_gn'],
                   help='plain checkpoint directory names to copy')
    args=p.parse_args()
    source=args.project.resolve();pairs=[]
    if not source.is_dir():p.error('project does not exist')
    for split in ('train','val','test'):
        src=source/'data'/f'{split}.npz'
        if not src.is_file():p.error(f'missing data/{split}.npz')
        pairs.append((src,Path('data')/src.name))
    for folder in args.checkpoint_folders:
        if Path(folder).name!=folder or folder in ('.','..'):p.error('checkpoint folders must be plain directory names')
        paths=sorted((source/folder).glob('*.pt'))
        if not paths:p.error(f'no checkpoint files in {folder}')
        pairs.extend((s,Path(folder)/s.name) for s in paths)
    run=stage(args.run_dir);target=run/'simulation'
    if source==target:p.error('source and destination must differ')
    # Preflight all conflicts before copying any data or checkpoints.
    for src,relative in pairs:
        dst=target/relative
        if dst.exists() and hashlib.file_digest(src.open('rb'),'sha256').hexdigest()!=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest():
            p.error(f'different existing asset: {relative}; choose a new run directory')
    for src,relative in pairs:
        dst=target/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
    print(f'Copied/verified {len(pairs)} assets into {args.run_dir}; source untouched.')


if __name__=='__main__':main()
