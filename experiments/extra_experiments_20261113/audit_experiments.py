"""Checks against consequential failure modes, using real saved data/states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from common import OUT, PROJECT, build, data, dump, seed_all
from q2_quality_networks import Monitor, cache, predict


def hashes(x):
    return {hashlib.sha256(sample.numpy().tobytes()).hexdigest() for sample in x}


def main():
    xtr,xval,xte=data()
    sets=[hashes(x) for x in (xtr,xval,xte)]
    overlap={f'{a}_{b}':len(sets[a]&sets[b]) for a,b in ((0,1),(0,2),(1,2))}
    assert all(n==0 for n in overlap.values()),'H split leakage'
    checks={'train_val_test_H_overlap':overlap,'q1_frozen_state':[],'q2_invariance':[]}
    rows=json.loads((OUT/'q1_results.json').read_text(encoding='utf8'))['cells']
    for row in rows:
        seed,ns=row['seed'],row['ns']
        source=torch.load(PROJECT/'ckpt_rdae'/f'rdae_sample_s{seed}.pt',map_location='cpu',weights_only=True)
        s2=torch.load(OUT/'q1_states'/f's{seed}_ns{ns}_S2.pt',map_location='cpu',weights_only=True)
        for mode in ('bn_affine','af','decoder'):
            adapted=torch.load(OUT/'q1_states'/f's{seed}_ns{ns}_{mode}.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(source[k],v) for k,v in adapted.items() if k.startswith('enc.'))
            assert all(torch.equal(s2[k],v) for k,v in adapted.items()
                       if k.endswith(('running_mean','running_var','num_batches_tracked')))
            changed=[k for k in s2 if not torch.equal(s2[k],adapted[k])]
            if mode=='bn_affine':
                net,_=build(seed,xtr)
                allowed=set()
                for name,module in net.dec.named_modules():
                    if isinstance(module,torch.nn.BatchNorm2d):
                        allowed.update({f'dec.{name}.weight',f'dec.{name}.bias'})
                assert set(changed)<=allowed
            if mode=='af':
                assert all('.af' in key for key in changed)
            checks['q1_frozen_state'].append({'seed':seed,'ns':ns,'mode':mode,'changed_tensors':len(changed),'passed':True})
    # Quality extraction and prediction use only received y, conditions, frozen receiver.
    net,_=build(0,xtr)
    original={k:v.clone() for k,v in net.state_dict().items()}
    rows_a=cache(net,xte[:16],(20,),lengths=(16,),noise_seed=40000)
    rows_b=cache(net,xte[:16],(20,),lengths=(16,),noise_seed=40000)
    assert all(torch.equal(rows_a[k],rows_b[k]) for k in rows_a),'Cache non-determinism'
    assert all(torch.equal(original[k],v) for k,v in net.state_dict().items()),'Backbone mutated'
    bundle=torch.load(OUT/'q2_head_s0_feature.pt',map_location='cpu',weights_only=True)
    head=Monitor(bundle['input_dim']);head.load_state_dict(bundle['state_dict']);head.eval()
    prediction=predict(head,bundle,rows_a)
    modified=dict(rows_a)
    modified['target']=torch.full_like(rows_a['target'],1e9)
    assert (prediction==predict(head,bundle,modified)).all(),'Predictor reads labels'
    changed_ns={k:v.clone() for k,v in original.items()}
    net.load_state_dict(torch.load(OUT/'q1_states'/'s0_ns16_S2.pt',map_location='cpu',weights_only=True))
    rows_s2=cache(net,xte[:16],(20,),lengths=(16,),noise_seed=40000)
    assert torch.equal(rows_a['y'],rows_s2['y']),'Receiver change altered transmitted observations'
    checks['q2_invariance']=['paired noise reproducible','backbone state unchanged during feature collection',
                             'prediction invariant to target-label mutation','y unchanged by decoder BN calibration']
    checks['status']='PASS'
    dump(OUT/'audit.json',checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
