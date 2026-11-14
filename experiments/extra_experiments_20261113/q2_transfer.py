"""Complete quality-monitor transfer checks after ALL Q1 checkpoints exist."""
from __future__ import annotations

import json

import torch

from common import OUT, build, data, dump
from q2_quality_networks import Monitor, cache, metrics, predict


def main():
    torch.set_num_threads(2)
    path = OUT / 'q2_results.json'
    result = json.loads(path.read_text(encoding='utf8'))
    args = result['protocol']
    xtr, xval, xte = data()
    for row in result['models']:
        seed = row['seed']
        net, _ = build(seed, xtr)
        original = {k:v.clone() for k,v in net.state_dict().items()}
        learned = {}
        for mode in ('y','feature','y_feature'):
            bundle = torch.load(OUT/f'q2_head_s{seed}_{mode}.pt', map_location='cpu', weights_only=True)
            head = Monitor(bundle['input_dim'])
            head.load_state_dict(bundle['state_dict']); head.eval()
            learned[mode] = head,bundle
        for ns in (16,32):
            original_test = cache(net,xte[:args['ntest']],(20,),lengths=(ns,),
                                  repeats=args['test_repeats'],noise_seed=40000)
            state_path = OUT/'q1_states'/f's{seed}_ns{ns}_S2.pt'
            net.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True))
            new_test = cache(net,xte[:args['ntest']],(20,),lengths=(ns,),
                             repeats=args['test_repeats'],noise_seed=40000)
            assert torch.equal(new_test['y'],original_test['y']), 'Received signal changed'
            new_val = cache(net,xval[:args['nval']],(20,),lengths=(ns,),noise_seed=30000)
            key = f'20dB_ns{ns}'
            row['transfer'][key] = {}
            row.setdefault('transfer_validation_bias_correction',{})[key] = {}
            for mode,(head,bundle) in learned.items():
                pred = predict(head,bundle,new_test)
                before = predict(head,bundle,original_test)
                if mode == 'y':
                    assert (pred == before).all(), 'y-only head changed although y was identical'
                row['transfer'][key][mode] = metrics(new_test,pred)[key]
                # Optional single-scalar correction uses ONLY adaptation validation H.
                vp = predict(head,bundle,new_val)
                offset = float((new_val['target'].numpy()-vp).mean())
                corrected = metrics(new_test,pred+offset)[key]
                corrected['validation_offset_db'] = offset
                row['transfer_validation_bias_correction'][key][mode] = corrected
                print(f'TRANSFER seed={seed} ns={ns} {mode} '+
                      f'AUROC={row["transfer"][key][mode]["auroc"]:.3f} '+
                      f'MAE={row["transfer"][key][mode]["mae_db"]:.3f} '+
                      f'bias_corrected_MAE={corrected["mae_db"]:.3f}',flush=True)
            net.load_state_dict(original)
        dump(path,result)


if __name__=='__main__':
    main()
