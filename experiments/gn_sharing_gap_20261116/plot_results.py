"""Plot locked results after the four models finish; no model training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent
result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
font=FontProperties(family=['Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans'])
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
snrs=[0,10,20]
shared=[result['results']['shared'][str(s)] for s in snrs]
special=[result['results'][f'specialist_{s}'][str(s)] for s in snrs]
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,key,title,ylab in zip(axes,['mean_nmse_db','sgcs'],['重建误差','CSI 方向相似性'],['平均逐样本 NMSE / dB（越低越好）','平方 SGCS（越高越好）']):
    ax.plot(snrs,[r[key] for r in shared],'o-',color='#235A91',label='GN 共享模型')
    ax.scatter(snrs,[r[key] for r in special],marker='s',s=65,color='#BE6A35',label='各锚点 GN 专用模型',zorder=3)
    ax.set_xticks(snrs);ax.set_xlabel('反馈 SNR / dB');ax.set_ylabel(ylab);ax.set_title(title)
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
fig.suptitle('RD-AE + GN：固定 64 个复符号，逐样本 SNR 共享与专用')
fig.text(.5,.01,'相同初始化与每模型训练预算，40 轮、单种子；test1000H × 3 组配对噪声',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.045,1,.92))
fig.savefig(ROOT/'GN共享与专用.png',dpi=170)
fig.savefig(ROOT/'GN共享与专用.pdf')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,key,title in zip(axes,['nmse_shared_minus_specialist','sgcs_specialist_minus_shared'],['NMSE 共享减专用 / dB','SGCS 专用减共享']):
    stats=[result['gaps'][str(s)][key] for s in snrs]
    means=[r['mean'] for r in stats]
    errors=[[r['mean']-r['H_cluster_CI95'][0] for r in stats],[r['H_cluster_CI95'][1]-r['mean'] for r in stats]]
    ax.errorbar(snrs,means,yerr=errors,fmt='o',capsize=5,color='#235A91')
    ax.axhline(0,color='#777777',lw=1)
    ax.set_xticks(snrs);ax.set_xlabel('反馈 SNR / dB');ax.set_title(title);ax.grid(alpha=.2)
fig.suptitle('正值表示共享较差，负值表示共享更好')
fig.text(.5,.01,'区间仅为测试 H 聚类 bootstrap 95% 区间，不包含重训种子不确定性',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.045,1,.92))
fig.savefig(ROOT/'GN共享代价与区间.png',dpi=170)
fig.savefig(ROOT/'GN共享代价与区间.pdf')
plt.close(fig)
print('Saved two PNG/PDF figures.')
