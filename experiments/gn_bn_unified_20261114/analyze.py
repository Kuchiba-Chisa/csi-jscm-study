"""Independent saved-array audit and figures for the unified evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
NS=(16,32,64);SNRS=(0,10,20);MODES=('BN_S0','BN_S2','GN')

def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    arrays=np.load(OUT/'observations.npz');errors=[]
    for original,digest in protocol['source_sha256'].items():
        if hashlib.sha256(Path(original).read_bytes()).hexdigest()!=digest:errors.append('source modified: '+original)
    for seed in range(3):
        for ns in NS:
            for s in SNRS:
                cell=result['results'][str(seed)][f'ns{ns}_snr{s}']
                for mode in MODES:
                    prefix=f's{seed}_ns{ns}_snr{s}_{mode}_'
                    db,r,sc=[arrays[prefix+suffix] for suffix in ('db','ratio','sgcs')]
                    if db.shape!=(3,1000):errors.append('shape '+prefix)
                    if not np.isfinite(db).all():errors.append('finite '+prefix)
                    if not np.allclose(db,10*np.log10(r+1e-12)):errors.append('log metric '+prefix)
                    for key,value in [('mean_nmse_db',db.mean()),('db_mean_linear_nmse',10*np.log10(r.mean())),('sgcs',sc.mean())]:
                        if abs(float(value)-cell[mode][key])>2e-6:errors.append('summary '+prefix+key)
                    if sc.min()<-1e-5 or sc.max()>1+1e-5:errors.append('SGCS range '+prefix)
                    if db.mean()>10*np.log10(r.mean())+1e-5:errors.append('Jensen '+prefix)
                if cell['BN_S0']['y_sha256']!=cell['BN_S2']['y_sha256']:errors.append('BN y changed')
                if len({row['standard_noise_sha256'] for row in cell.values()})!=1:errors.append('noise mismatch')
    audit=dict(status='PASS' if not errors else 'FAIL',errors=errors,checked_deployment_cells=27,checked_model_evaluations=81,
               note='Arithmetic/state/hash audit; no new model selection. SingleH noise repeats not treated as independent training seeds.')
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:raise RuntimeError(errors)
    summary=result['summary'];contrasts=result['contrasts']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    colors={'BN_S0':'#6d88aa','BN_S2':'#2e7d69','GN':'#cb8e32'}
    names={'BN_S0':'BN 原始','BN_S2':'BN 目标条件重标定','GN':'GN（两端均替换）'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for ns,ax in zip(NS,axes):
        for mode in MODES:
            m=[summary[f'ns{ns}_snr{s}'][mode]['mean_nmse_db']['mean'] for s in SNRS]
            sd=[summary[f'ns{ns}_snr{s}'][mode]['mean_nmse_db']['sample_sd'] for s in SNRS]
            ax.errorbar(SNRS,m,yerr=sd,marker='o',capsize=3,color=colors[mode],label=names[mode])
        ax.set(title=f'Ns={ns} 复符号',xlabel='反馈 SNR / dB',xticks=SNRS)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('平均逐样本 NMSE / dB（越低越好）')
    axes[-1].legend(fontsize=9)
    fig.suptitle('同一协议：三种子均值 ± 样本SD；不是性能等价检验')
    fig.savefig(OUT/'统一协议_NMSE.png',dpi=180);fig.savefig(OUT/'统一协议_NMSE.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ns,ax in zip(NS,axes):
        for mode in MODES:
            m=[summary[f'ns{ns}_snr{s}'][mode]['sgcs']['mean'] for s in SNRS]
            sd=[summary[f'ns{ns}_snr{s}'][mode]['sgcs']['sample_sd'] for s in SNRS]
            ax.errorbar(SNRS,m,yerr=sd,marker='o',capsize=3,color=colors[mode],label=names[mode])
        ax.set(title=f'Ns={ns}',xlabel='反馈 SNR / dB',xticks=SNRS);ax.grid(alpha=.2)
    axes[0].set_ylabel('空频域平方 SGCS（越高越好）');axes[-1].legend(fontsize=9)
    fig.suptitle('平方SGCS直接由空频域CSI计算，不重复做FFT')
    fig.savefig(OUT/'统一协议_SGCS.png',dpi=180);fig.savefig(OUT/'统一协议_SGCS.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
    labels=[f'{ns}/{s}' for ns in NS for s in SNRS]
    for ax,name,title in [(axes[0],'GN_minus_BN_S2','GN − 重标定BN：正值为重标定BN好'),(axes[1],'BN_S0_minus_BN_S2','原始BN − 重标定BN：正值为修正有效')]:
        for i,(ns,s) in enumerate((n,s) for n in NS for s in SNRS):
            row=contrasts[f'ns{ns}_snr{s}'][name]
            mean=row['mean'];lo,hi=row['paired_seed_t_CI95']
            ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',capsize=4,color='#2e7d69' if mean>0 else '#cb8e32')
            ax.scatter(np.full(3,i),row['values'],marker='.',s=18,color='#6b7c91',alpha=.7)
        ax.axhline(0,color='#52657b',linewidth=.8);ax.set_xticks(range(9),labels,rotation=45)
        ax.set(title=title,xlabel='Ns / SNR(dB)',ylabel='平均NMSE差 / dB');ax.grid(axis='y',alpha=.2)
    fig.suptitle('3个模型差值与配对t区间(df=2)；探索性，未作多重比较校正')
    fig.savefig(OUT/'统一协议_差值与不确定性.png',dpi=180);fig.savefig(OUT/'统一协议_差值与不确定性.pdf');plt.close(fig)
    lines=['# 统一评估后的结论','',
      '本轮没有重训，比较现有BN与GN三种子；用同一完整test1000H×3组噪声。没有沿用已经作废的旧BN JSON，也没有混用NMSE公式。','',
      '## 直接回答：重标定BN是否比GN好','',
      '| Ns | SNR | GN−BN-S2 / dB | 三种子是否都支持同一方向 | 配对种子95%区间 |','|---|---|---|---|---|']
    for ns in NS:
        for s in SNRS:
            r=contrasts[f'ns{ns}_snr{s}']['GN_minus_BN_S2'];lo,hi=r['paired_seed_t_CI95']
            direction='三者均为BN-S2好' if all(v>0 for v in r['values']) else ('三者均为GN好' if all(v<0 for v in r['values']) else '方向不一致')
            lines.append(f"| {ns} | {s} | {r['mean']:+.4f} | {direction} | [{lo:+.4f}, {hi:+.4f}] |")
    win=sum(contrasts[f'ns{n}_snr{s}']['GN_minus_BN_S2']['mean']>0 for n in NS for s in SNRS)
    lines+=['',f'9个条件中，BN-S2在{win}个条件的均值优于GN，GN在{9-win}个条件更好。格子数量不是统计显著性或真实部署流量加权收益。','',
      '## 这能支持什么','',
      '- 按具体Ns/SNR报告结构替换与条件校准的选择，不预设重标定BN一定赢。',
      '- 直接比较BN-S2与GN，不从GN≈BN-S0间接推导。',
      '- 若区间跨0，表示证据不足以排除方向变化，不表示已经证明两者等价。',
      '- 若区间不跨0，仍仅是3种子、9条件的探索性证据；没有多重比较校正。',
      '- GN两端替换改变编码器和解码器；不是只删去一个接收端失配源，也不代表所有LN/GDN方案。',
      '- 是否降低共享代价仍需相应专用模型参照，此处衡量的是现有共享模型性能。','',
      '完整表格和指标见统一协议比较结果.md。原始文件完整性及81个模型评估的独立复核均为PASS。','',
      '![统一NMSE](统一协议_NMSE.png)','![差值和区间](统一协议_差值与不确定性.png)']
    (OUT/'结论导读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False));print('Plots and conclusion guide complete.')

if __name__=='__main__':main()
