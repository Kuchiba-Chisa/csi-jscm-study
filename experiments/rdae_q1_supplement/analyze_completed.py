"""Audit and plot already completed outputs; does not train codecs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent

def audit():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    archive=np.load(OUT/'test_arrays.npz')
    errors=[]
    for original,expected in protocol['source_sha256'].items():
        if hashlib.sha256(Path(original).read_bytes()).hexdigest()!=expected:
            errors.append('original changed: '+original)
    for name,modes in result['results'].items():
        for mode,cells in modes.items():
            for snr,row in cells.items():
                key=f'{name}__{mode}__{snr}__'
                db,ratio,sgcs=[archive[key+suffix] for suffix in ('db','ratio','sgcs')]
                if db.shape!=(3,400):errors.append('unexpected shape: '+key)
                if not np.isfinite(db).all():errors.append('nonfinite loss: '+key)
                if not np.allclose(db,10*np.log10(ratio+1e-12)):errors.append('metric mismatch: '+key)
                for metric,value in [('mean_nmse_db',db.mean()),('db_mean_linear_nmse',10*np.log10(ratio.mean())),('sgcs',sgcs.mean())]:
                    if abs(row[metric]-float(value))>1e-6:errors.append('summary mismatch: '+key+metric)
                if row['mean_nmse_db']>row['db_mean_linear_nmse']+1e-5:errors.append('Jensen mismatch: '+key)
                if sgcs.min()<-1e-5 or sgcs.max()>1+1e-5:errors.append('SGCS range: '+key)
                if mode!='raw' and row['y_sha256']!=modes['raw'][snr]['y_sha256']:
                    errors.append('BN changed y: '+key)
    result_audit=dict(status='PASS' if not errors else 'FAIL',errors=errors,
                      checked_cells=sum(len(c) for m in result['results'].values() for c in m.values()),
                      caveat='one training seed; replay of saved arrays checks arithmetic, not population generalization')
    (OUT/'independent_audit.json').write_text(json.dumps(result_audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:raise RuntimeError(errors)
    return result

def main():
    result=audit()
    rows=result['results'];contrasts=result['contrasts']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
    grid=[0,5,10,15,20]
    colors={'shared':'#326ba1','weighted':'#d99a37','distilled':'#8a5ba7'}
    labels={'shared':'共享基准','weighted':'重加权','distilled':'蒸馏'}
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    for name in colors:
        raw=[rows[name]['raw'][str(s)]['mean_nmse_db'] for s in grid]
        target=[rows[name]['target'][str(s)]['mean_nmse_db'] for s in grid]
        axes[0].plot(grid,raw,'o-',color=colors[name],label=labels[name])
        axes[1].plot(grid,target,'o-',color=colors[name],label=labels[name]+'：目标BN')
    for ax,mode,title in [(axes[0],'raw','原始BN：训练目标的比较'),(axes[1],'target','双方目标BN：剩余共享代价')]:
        teacher=[rows[f'teacher_{s}'][mode][str(s)]['mean_nmse_db'] for s in (0,10,20)]
        ax.scatter([0,10,20],teacher,color='#1d344b',marker='*',s=140,label='专用教师（只比较自身SNR）',zorder=5)
        ax.set(title=title,xlabel='反馈 SNR / dB',ylabel='平均逐样本 NMSE / dB（越低越好）',xticks=grid)
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('固定 Ns=64、16-QAM、含AF的 RD-AE 改编模型｜单种子探索')
    fig.savefig(OUT/'专用与共享_主曲线.png',dpi=180)
    fig.savefig(OUT/'专用与共享_主曲线.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    x=np.arange(3)
    for j,name in enumerate(('weighted','distilled')):
        values=[];lower=[];upper=[]
        for s in (0,10,20):
            row=contrasts[str(s)][name+'_vs_shared_raw']
            value=row['improvement_db'];lo,hi=row['H_paired_CI95']
            values.append(value);lower.append(max(0,value-lo));upper.append(max(0,hi-value))
        axes[0].bar(x+(j-.5)*.32,values,width=.30,color=colors[name],label=labels[name])
        axes[0].errorbar(x+(j-.5)*.32,values,yerr=[lower,upper],fmt='none',color='#263b52',capsize=4)
    axes[0].set(xticks=x,xticklabels=['0 dB','10 dB','20 dB'],ylabel='相对共享基准改善 / dB（正为好）',title='改训练目标：配对H区间，不是多种子区间')
    for j,name in enumerate(colors):
        values=[rows[name]['raw'][str(s)]['mean_nmse_db']-rows[name]['target'][str(s)]['mean_nmse_db'] for s in grid]
        axes[1].plot(grid,values,'o-',color=colors[name],label=labels[name])
    axes[1].set(xticks=grid,xlabel='反馈 SNR / dB',ylabel='目标BN相对原始改善 / dB',title='训练目标之外：BN状态还会改变多少')
    for ax in axes:ax.axhline(0,color='#61758e',linewidth=.8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    fig.savefig(OUT/'改善与权衡.png',dpi=180);fig.savefig(OUT/'改善与权衡.pdf');plt.close(fig)
    lines=['# 如何解读这次单种子补实验','',
    '先看同骨干同预算的专用—共享差距，再看训练目标和 BN。这里没有将旧多长度模型与新固定长度模型混在一起。','',
    '## 1. 这个共享模型有没有代价','',
    '| SNR | 原始共享−原始专用/dB | 目标BN共享−目标BN专用/dB |','|---|---|---|']
    for s in (0,10,20):
        a=contrasts[str(s)]['shared_raw_minus_specialist_raw']['gap_db']
        b=contrasts[str(s)]['shared_target_minus_specialist_target']['gap_db']
        lines.append(f'| {s} | {a:+.4f} | {b:+.4f} |')
    lines+=['','正值表示共享较差；负值表示在这个单种子预算下共享反而更好。共享可能因跨条件训练的正则化或优化差异而赢，不应预设专用必然占优。','',
    '## 2. 重加权和蒸馏是否全面有效','',
    '| SNR | 重加权相对共享/dB | 蒸馏相对共享/dB |','|---|---|---|']
    for s in (0,10,20):
        vals=[contrasts[str(s)][n+'_vs_shared_raw']['improvement_db'] for n in ('weighted','distilled')]
        lines.append(f'| {s} | {vals[0]:+.4f} | {vals[1]:+.4f} |')
    lines+=['','正值为改善；逐点看有无权衡，不用三点平均掩盖单点退化。配对H区间在 results.json，不表示训练种子稳定性。','',
    '## 3. 能否写进答辩','',
    '可以作为附录中的同骨干探索性验证。若需要把它升为主结论，至少增加独立训练种子；不得把一次训练的差异当成普遍共享代价或确定的方法收益。','',
    '本轮不自动改 PPT、旧权重或历史结论；最终采用哪条表述应由完整结果决定。','',
    '![主曲线](专用与共享_主曲线.png)','![改善与权衡](改善与权衡.png)']
    (OUT/'结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PASS: independent arithmetic/state audit and figures complete')

if __name__=='__main__':main()
