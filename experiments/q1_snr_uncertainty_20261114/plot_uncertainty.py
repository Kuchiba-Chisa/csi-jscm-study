"""Source-seed variation, not repeated-noise standard errors."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=[json.loads((root/f'results_s{s}.json').read_text(encoding='utf8')) for s in (0,1,2)]
errors=(-3,-1,0,1,3)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,2,figsize=(10.5,9),sharex=True,constrained_layout=True)
for i,ns in enumerate((16,32,64)):
    for j,mode in enumerate(('bn_only','all_receiver')):
        ax=axs[i,j]
        for method,label,color in [('vanilla','Full adaptation','#bc4749'),('conservative','Validation-selected shrinkage','#2a9d8f')]:
            values=[]
            for row in rows:
                values.append([np.mean([c['methods'][method]['gain_vs_input_matched_S0_db'] for c in row['cells']
                    if c['ns']==ns and c['mode']==mode and c['error']==e and c['true_snr'] in (5,10,15)]) for e in errors])
            values=np.array(values);mean=values.mean(0);sd=values.std(0,ddof=1)
            ax.plot(errors,mean,'o-',label=label,color=color)
            ax.fill_between(errors,mean-sd,mean+sd,color=color,alpha=.12)
        ax.axhline(0,color='gray',ls=':',lw=1)
        ax.set_title(f'Ns={ns}; '+('BN selection error only' if j==0 else 'BN + receiver AF error'))
        ax.grid(alpha=.18);ax.set_ylabel('NMSE gain vs input-matched S0 (dB)')
        ax.set_xticks(errors)
        if i==0:ax.legend(fontsize=8)
        if i==2:ax.set_xlabel('Signed BS SNR bias (dB)')
fig.suptitle('Finite 16-QAM CSI feedback: SNR uncertainty and conservative BN adaptation\nTrue SNR 5/10/15 dB equally weighted; mean +/- SD of 3 source seeds',fontsize=12)
fig.savefig(root/'snr_uncertainty.png',dpi=180)
fig.savefig(root/'snr_uncertainty.pdf')
