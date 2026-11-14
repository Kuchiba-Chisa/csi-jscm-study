"""Replay supplementary source-receiver continuation weights on fixed observations."""
from pathlib import Path
import json
import numpy as np
import torch
import run_joint as r

root=Path(__file__).resolve().parent
tr,va,te=r.rf.data()
torch.set_num_threads(4)
checks=[]
for seed in (0,1,2):
    net,source=r.rf.build(seed,tr)
    obs=torch.load(root/f'test_observations_s{seed}.pt',weights_only=True)['observations']
    cache=r.decoded(net,obs,te[600:1000],seed,False)
    model,b=r.old_refiner(seed)
    updated=torch.load(root/f'continuation_control/updated_refiner_s{seed}.pt',weights_only=True)
    model.load_state_dict(updated['state_dict']);model.eval()
    out=r.rf.evaluate(model,cache,updated['norm'])
    saved=np.load(root/f'continuation_control/outcomes_s{seed}.npz')
    error=float(np.abs(saved['refined_loss']-out['loss'].numpy()).max())
    assert error<1e-6
    checks.append(dict(seed=seed,max_forward_replay_error=error))
    print('CONTROL AUDIT PASS',seed,error,flush=True)
r.rf.dump(root/'continuation_control/audit.json',dict(status='PASS',checks=checks))
