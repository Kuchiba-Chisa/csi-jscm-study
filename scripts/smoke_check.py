"""Weight-free checks of the actual finite-QAM link and joint training path."""
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simulation'))
sys.path.insert(0,str(ROOT/'experiments/extra_experiments_20261113'))
from model_rdae import LinkRDAE
from model_v4 import LEVELS, from_symbols, normalize_power, pair_mask, qam_map_ste, to_symbols
from common import recalibrate


def main():
    torch.set_num_threads(2);torch.manual_seed(7)
    x=torch.randn(8,2,32,32)/3
    net=LinkRDAE(cond=False).eval()
    snr=10.;ns=16
    prediction,parts=net(x,snr,ns,return_parts=True)
    assert prediction.shape==x.shape and torch.isfinite(prediction).all()
    mask=pair_mask(torch.full((len(x),),ns))
    # Check the symbols actually sent, not unused intermediate slots.
    af=torch.full((len(x),1),10**(-snr/20))
    z=normalize_power(net.enc(x,af),ns)
    symbols=qam_map_ste(to_symbols(z))
    distances=(symbols[:,:ns,:,None]-LEVELS).abs().amin(-1)
    assert float(distances.max().detach())<1e-6
    # parts['sigma'] is complex-noise RMS; each real dimension uses sigma/sqrt(2).
    expected_sigma=parts['p_tx'].sqrt()*10**(-snr/20)
    torch.testing.assert_close(parts['sigma'].flatten(),expected_sigma.flatten())
    # Inspect the real decoder input through a pre-hook.
    captured=[]
    hook=net.dec.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
    torch.manual_seed(541)
    try:net(x,snr,ns)
    finally:hook.remove()
    assert torch.count_nonzero(captured[-1]*(1-mask))==0
    torch.manual_seed(541)
    eps=torch.randn_like(symbols)
    manual_y=from_symbols(symbols+eps*(expected_sigma/2**0.5)[:,None,None])*mask
    torch.testing.assert_close(captured[-1],manual_y)
    # Gradients must reach both UE and BS despite hard finite constellation.
    net.zero_grad(set_to_none=True)
    loss=(net(x,snr,ns)-x).square().mean();loss.backward()
    grad={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
          for name,module in [('encoder',net.enc),('decoder',net.dec)]}
    assert min(grad.values())>0 and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    before={k:v.clone() for k,v in net.state_dict().items()}
    params={k:v.detach().clone() for k,v in net.named_parameters()}
    recalibrate(net,x,snr=snr,ns=ns,seed=17,bs=4)
    after=net.state_dict()
    changed=[k for k in before if not torch.equal(before[k],after[k])]
    assert changed and all(k.startswith('dec.') and k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changed)
    assert all(torch.equal(v,dict(net.named_parameters())[k]) for k,v in params.items())
    assert not any(m.training for m in net.modules() if isinstance(m,nn.BatchNorm2d))
    print(json.dumps(dict(status='PASS',finite_qam=True,padding_zero=True,power_noise_consistent=True,
                          joint_gradients=grad,recalibration_changes_weights=False,
                          note='Random untrained model and synthetic tensors; no performance claim.'),indent=2))


if __name__=='__main__':main()
