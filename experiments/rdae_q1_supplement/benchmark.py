"""Short in-memory timing; no checkpoint writes."""
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'simulation')
import time,json,torch
from model_rdae import LinkRDAE
from train_rdae import load01
x=load01('train',root='simulation/data')[:128]
rows=[]
for threads in (2,4,8):
    torch.set_num_threads(threads)
    torch.manual_seed(50)
    net=LinkRDAE(cond=False).train()
    opt=torch.optim.Adam(net.parameters(),lr=.002)
    snr=torch.linspace(0,20,len(x))
    ns=torch.full((len(x),),64,dtype=torch.long)
    for i in range(2):
        loss=(net(x,snr,ns)-x).square().mean();opt.zero_grad();loss.backward();opt.step()
    started=time.perf_counter()
    for i in range(6):
        loss=(net(x,snr,ns)-x).square().mean();opt.zero_grad();loss.backward();opt.step()
    elapsed=(time.perf_counter()-started)/6
    row={'threads':threads,'seconds_per_step':elapsed,'one_model_40epochs_46steps_minutes':elapsed*40*46/60}
    rows.append(row);print(json.dumps(row),flush=True)
print(json.dumps({'timings':rows,'no_original_writes':True}),flush=True)
