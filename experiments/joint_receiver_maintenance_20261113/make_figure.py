from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text(encoding='utf8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
x=np.arange(3)
for name,label,col in [('S0_old_monitor_old_refiner','Original receiver','#415a77'),
                       ('A_old_modules_old_gate','Adapted, old threshold','#bc4749'),
                       ('A_rethreshold_old_modules','Adapted, reset threshold','#2a9d8f')]:
    axs[0].plot(x,[100*s['policies'][str(v)][name]['refiner_call_rate'] for v in (0,10,20)],'o-',label=label,color=col)
axs[0].axhline(20,color='gray',ls=':',lw=1)
axs[0].set(ylabel='Actual refiner calls (%)',title='Threshold drift after BN adaptation')
for state,label,col in [('S0','Original','#415a77'),('A_old','Adapted, old monitor','#bc4749'),('A_affine','Score affine calibration','#d49a00'),('A_updated','Offline-updated monitor','#2a9d8f')]:
    axs[1].plot(x,[s['quality'][str(v)][state]['mae_db'] for v in (0,10,20)],'o-',label=label,color=col)
axs[1].set(ylabel='Quality prediction MAE (dB)',title='Calibration needs separate validation')
for name,label,col in [('S0_old','Original + old refiner','#415a77'),('A_old','Adapted + old refiner','#bc4749'),('A_moment','Adapted + input moments','#d49a00'),('A_updated','Adapted + updated refiner','#2a9d8f')]:
    axs[2].plot(x,[s['refiners'][str(v)][name]['always_refine_gain_db'] for v in (0,10,20)],'o-',label=label,color=col)
axs[2].axhline(0,color='gray',ls=':',lw=1)
axs[2].set(ylabel='Always-refine mean NMSE gain (dB)',title='Refiner transfer is not automatic')
for ax in axs:
    ax.set_xticks(x,['0','10','20']);ax.set_xlabel('Feedback SNR (dB)')
    ax.grid(alpha=.18);ax.legend(fontsize=7,loc='best')
fig.suptitle('Receiver maintenance: Ns=64, finite 16-QAM, AWGN; mean of 3 source seeds',fontsize=12)
fig.savefig(root/'joint_maintenance.png',dpi=180)
fig.savefig(root/'joint_maintenance.pdf')
