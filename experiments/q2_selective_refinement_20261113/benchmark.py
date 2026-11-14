"""Same-PC single-sample timing of monitor, refiner and real conditional branch."""
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from run_refinement import OUT,PREV,Refiner,build,data,score_head,model_input,dump,sha

torch.set_num_threads(1)
truths=data();torch.set_num_threads(1)
results=[]
with torch.inference_mode():
    for seed in (0,1,2):
        net,_=build(seed,truths[0]);torch.set_num_threads(1)
        b=torch.load(OUT/f'refiner_s{seed}.pt',weights_only=True)
        model=Refiner(b['width']);model.load_state_dict(b['state_dict']);model.eval()
        head,hb=score_head(seed)
        rows=torch.load(PREV/f'cache_s{seed}_test.pt',weights_only=True)
        r=json.loads((OUT/f'results_s{seed}.json').read_text(encoding='utf8'))
        indices=torch.cat([torch.where(rows['snr']==s)[0][:32] for s in (0,10,20)])
        f={}
        tap=net.dec.blocks[-1].af2.register_forward_hook(lambda m,a,o:f.update(value=o))
        y=rows['y'][indices];s=rows['snr'][indices]
        base=net.dec(y,10**(-s[:,None]/20))
        maps=torch.cat([base,f['value']],1);tap.remove()
        samples=[(maps[i:i+1],s[i:i+1],rows['condition'][indices[i:i+1]]) for i in range(len(indices))]
        def score(item):
            z,ss,c=item
            feature=torch.cat([F.adaptive_avg_pool2d(z[:,2:],(2,2)).flatten(1),c],1)
            return head((feature-hb['mean'])/hb['sd'])*hb['target_sd']+hb['target_mean']
        def refine(item):
            z,ss,c=item
            return z[:,:2]+model(model_input(z,ss,b['mean'],b['sd']))
        def gated(item):
            z,ss,c=item
            threshold=r['conditions'][str(int(ss.item()))]['gates']['0.2']['threshold']
            return refine(item) if bool(score(item)>=threshold) else z[:,:2]
        functions={'monitor_only':score,'refiner_always':refine,'quality_gate20_pipeline':gated}
        for fn in functions.values():
            for item in samples[:20]:fn(item)
        timing={name:[] for name in functions}
        rng=np.random.default_rng(55000+seed)
        for block in range(7):
            for name in rng.permutation(list(functions)):
                tick=time.perf_counter_ns()
                for _ in range(2):
                    for item in samples:functions[name](item)
                timing[name].append((time.perf_counter_ns()-tick)/(2*len(samples))/1e6)
        count=0
        for item in samples:
            threshold=r['conditions'][str(int(item[1].item()))]['gates']['0.2']['threshold']
            count+=int(bool(score(item)>=threshold))
        results.append({'seed':seed,'sample_observations':len(samples),'gated_call_fraction':count/len(samples),
                        'timing':{name:{'block_ms':v,'median_ms':float(np.median(v))} for name,v in timing.items()}})
        print('TIMING',seed,{k:round(float(np.median(v)),4) for k,v in timing.items()},'calls',count,flush=True)
dump(OUT/'timing.json',{'cpu':'AMD Ryzen 7 7735H','torch':torch.__version__,'batch':1,'threads':1,
                      'results':results,'script_sha256':sha(Path(__file__)),
                      'note':'same PC timings, excludes base decoder; gate includes pool,normalization,head,branch and selected refiner. Prefix96 observations, not full-test time or mobile power.'})
