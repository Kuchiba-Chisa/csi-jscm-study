"""Report all prespecified policies, fresh confirmation and limitations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = [json.loads(p.read_text(encoding='utf8')) for p in sorted(ROOT.glob('results_s*.json'))]
assert len(rows) == 3
assert json.loads((ROOT / 'audit.json').read_text(encoding='utf8'))['status'] == 'PASS'
POLICIES = ('random', 'ue_quality_best', 'ue_gain', 'bs_unconditional_gain',
            'bs_conditional_gain', 'bs_conditional_quality_worst')
NAMES = dict(random='随机追加', ue_quality_best='UE代理预计最好优先', ue_gain='UE条件收益头',
             bs_unconditional_gain='BS未筛选训练收益头', bs_conditional_gain='BS通过者专用收益头',
             bs_conditional_quality_worst='BS通过者当前质量最差优先')


def vals(s, policy, key='gain_db', cohort='fresh_test'):
    return [r[cohort][str(s)]['policies'][policy][key] for r in rows]


def pm(x, d=4, scale=1):
    x = np.asarray(x) * scale
    return f'{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}'


summary = {}
for s in (0, 10, 20):
    cells = [r['fresh_test'][str(s)] for r in rows]
    summary[str(s)] = dict(
        gain_db={p: dict(mean=float(np.mean(vals(s, p))), sd=float(np.std(vals(s, p), ddof=1))) for p in POLICIES},
        bs_minus_UE_by_seed=[c['paired']['bs_conditional_gain_minus_ue_gain']['nmse_gain'] for c in cells],
        bs_minus_random_by_seed=[c['paired']['bs_conditional_gain_minus_random']['nmse_gain'] for c in cells],
        conditional_minus_unconditional_by_seed=[c['paired']['bs_conditional_gain_minus_bs_unconditional_gain']['nmse_gain'] for c in cells])
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
lines = ['# Q2：UE预检通过后，BS能否利用实际y决定重传？', '',
'日期：2026-11-13。三个源模型种子，原CSI编解码器、UE代理、功率标定保持冻结；本轮只训练小型决策头。', '',
'**结论：按预设平均NMSE主指标，UE预检后加入BS条件收益头没有优于UE单侧策略。0dB三个种子均支持BS收益低于UE；BS收益头相对随机在平均NMSE上的9个种子×SNR配对区间均包含0。按坏样本率次指标，0dB BS质量最差优先低于随机；补齐同目标的UE预计最差优先后，UE更低，见文末事后补充对照。不能据此推出所有接收端监测/动作都无效。**', '',
'## 1. 固定的两阶段问题', '',
'UE持有H与干净发送码字，用已有单次代理预测质量。各SNR分别用前250验证H的UE分数第80分位数设门限；预计最差约20%不进入本轮BS动作比较。筛选不依赖当前真实反馈噪声。', '',
'这里排除的样本没有实现增码率或模式切换，因此不是完整双侧系统评估，不能报告全体样本或整个协议的净收益。UE代理分数是预测指标，筛选通过也不保证真实信源压缩误差小，不能当作严格的信源/信道解耦。', '',
'通过者中，BS输入为当前带噪y128维、现有解码器特征64维和SNR/Ns条件2维。BS不读取H、发送符号、未来噪声或UE连续质量分数；只假设通过反馈模式标识知道该样本已通过预检。', '',
'## 2. 训练、冻结与新噪声确认', '',
'- 源链路：有限16-QAM、AWGN、Ns64、SNR0/10/20 dB，RD-AE改编骨干，非官方RD-JSCC端到端复现。',
'- 决策训练：既有trainH3000:5000，每H/条件3次当前噪声、8次独立未来噪声。BS条件收益头和条件质量头只训练UE通过者；未筛选BS收益头用全部同一训练数据。三个BS头同194维输入、25081参数、同优化规则与初始化。',
'- UE收益头只读本地代理分数和SNR/Ns；按H平均训练标签，使决策不依赖本次真实反馈噪声。保留零额外收益网络的“UE预计最好优先”强对照。',
'- valH0:250用于UE筛选门限与小头早停；valH250:500用于动作门限。原代理质量校准和旧模型曾用过全部500个验证H，本轮不是全新独立验证集。',
'- 每种子的规则和权重在locked_s*.json中锁定后，先对旧testH0:600回顾；该结果不调参，也不选择是否停止。',
'- 新噪声确认：testH600:1000，每H/条件3次全新当前噪声，每当前观测16次独立未来重传噪声。此段H被另一精修实验用过，不宣称是整个项目完全未见的信源测试集。',
'- 固定动作：UE再发完全相同的64个复符号，BS平均两次接收向量，再送原解码器。保持原SNR条件输入，与旧动作一致；没有针对有效SNR提高约3dB另优化接收端。因此只约束这一动作。', '',
'## 3. 主比较：相同通过者、相同追加次数', '',
'主比较是离线资源池内精确匹配约20%追加次数。UE按独立H选择并在该H三次当前噪声下作同一决定；BS按当前观测选择，实际追加次数严格相同。随机策略报告解析期望。', '',
'该比较衡量资源分配能力，不意味着实际部署要收齐400份CSI，也不是已实现的UE自主全局排序或HARQ调度。', '',
'所有收益都是**通过者整体平均逐样本NMSE_dB的下降**，不是仅重传样本平均、线性NMSE、全体原始样本收益或吞吐收益。均值±SD来自3个源模型种子，重复噪声不当作独立训练种子。', '',
'| 策略 | 0dB收益/dB | 10dB收益/dB | 20dB收益/dB |',
'|---|---|---|---|']
for p in POLICIES:
    lines.append(f'| {NAMES[p]} | ' + ' | '.join(pm(vals(s, p)) for s in (0, 10, 20)) + ' |')
lines += ['', '### 实际通过比例', '', '| SNR | 通过比例 | 每种子通过H数 |', '|---|---|---|']
for s in (0, 10, 20):
    lines.append(f"| {s} | {pm([r['fresh_test'][str(s)]['pass_fraction'] for r in rows], 2, 100)}% | " +
                 '/'.join(str(r['fresh_test'][str(s)]['passed_unique_H']) for r in rows) + ' |')
lines += ['', '### 配对比较，按H聚类的95%区间', '',
'H聚类保留同H三次当前噪声的相关性；区间条件于固定训练头和当前决策集合，不包含重新训练或重新排序控制器的不确定性。没有做多重比较校正，不将边缘区间解释为普适保证。', '',
'| SNR | 对照 | 种子0差值[区间] | 种子1差值[区间] | 种子2差值[区间] |', '|---|---|---|---|---|']
for s in (0, 10, 20):
    for base in ('random', 'ue_gain', 'bs_unconditional_gain'):
        cells = [r['fresh_test'][str(s)]['paired'][f'bs_conditional_gain_minus_{base}']['nmse_gain'] for r in rows]
        lines.append(f'| {s} | BS条件−{NAMES[base]} | ' + ' | '.join(f"{c['mean']:+.5f} [{c['CI95'][0]:+.5f},{c['CI95'][1]:+.5f}]" for c in cells) + ' |')
lines += ['', 'BS条件训练相比未筛选训练没有稳定改善。0/10dB，UE收益头与简单的“预计最好优先”再次选出相同集合；不能将UE收益归因于新增收益网络。', '',
'## 4. SGCS和固定误差门限风险', '',
'坏样本门限取val前250H中UE通过者的基础NMSE第80分位数，各SNR固定；是本轮相对误差门限，不是业务中断门限。重传后坏率按16次未来噪声逐次越限概率平均，不能先平均损失再判断越限。', '',
'| SNR | 策略 | 最终平方SGCS | 最终坏样本比例 |', '|---|---|---|---|']
for s in (0, 10, 20):
    for p in ('random', 'ue_gain', 'bs_conditional_gain', 'bs_conditional_quality_worst'):
        lines.append(f'| {s} | {NAMES[p]} | {pm(vals(s, p, "final_sgcs"), 5)} | {pm(vals(s, p, "bad_fraction"), 2, 100)}% |')
lines += ['', 'SGCS与坏样本风险均完整保留，不能把NMSE的排序自动解释成真实波束赋形或吞吐排序。', '',
'## 5. 验证固定动作门限与资源漂移', '',
'动作门限来自后250验证H，测试时冻结。各策略实际追加比例不同，原始收益不能直接当作等预算优劣。文件另存按实际调用次数匹配的随机期望和UE排序资源曲线；后者允许边界H随机化，是离线期望预算对照，未实现因果部署控制器。', '',
'| SNR | 策略 | 测试追加比例 | NMSE收益/dB |', '|---|---|---|---|']
for s in (0, 10, 20):
    for p in ('ue_gain', 'bs_conditional_gain'):
        cc = [r['fresh_test'][str(s)]['fixed_threshold'][p] for r in rows]
        lines.append(f"| {s} | {NAMES[p]} | {pm([c['retry_rate'] for c in cc], 2, 100)}% | {pm([c['gain_db'] for c in cc])} |")
lines += ['', '## 6. 控制开销敏感性', '',
'以每个通过者12.8个额外复符号当量为预算（原64符号的20%）。UE主动追加不收BS请求费用，BS请求每次收1或4符号当量；通过标识每样本收0或1当量。所有方法按相同总预算重排，边界随机化报告期望。', '',
'这只是抽象预算敏感性：上/下行资源不一定可直接相加，未模拟控制误码、可靠编码、时延、CSI老化或被排除者的回退成本。因此不能称真实系统净收益。', '',
'| 通过标识/每次BS请求当量 | SNR | UE收益/dB | BS条件收益/dB |', '|---|---|---|---|']
for scenario in ('passflag0_request1', 'passflag1_request1', 'passflag1_request4'):
    for s in (0, 10, 20):
        c = [r['fresh_test'][str(s)]['control_cost_sensitivity'][scenario] for r in rows]
        lines.append(f"| {scenario} | {s} | {pm([v['ue_gain']['gain_db'] for v in c])} | {pm([v['bs_conditional_gain']['gain_db'] for v in c])} |")
lines += ['', '## 7. 是否真的识别出了“被噪声拖累”？', '',
'离线诊断固定相同发送符号与解码器，比较当前含噪重建与完全去噪后重建的逐样本NMSE_dB之差。该干预不提供给部署策略，也不是严格可加的信源/信道平均误差分解。', '',
'| SNR | 当前−无噪声解码误差/dB | 所有通过者都重复的平均收益/dB | 同H内BS分数与重传收益Spearman |', '|---|---|---|---|']
for s in (0, 10, 20):
    dd = [r['fresh_test'][str(s)]['diagnostics'] for r in rows]
    lines.append(f"| {s} | {pm([d['noise_excess_mean_db'] for d in dd])} | {pm([d['all_passed_repeat_gain_db'] for d in dd])} | {pm([d['within_H_BS_gain_score_vs_actual_repeat_gain_spearman'] for d in dd])} |")
lines += ['', '0dB存在明确的噪声损失和重复改善空间，不能把本轮失败简单解释成“没有噪声可救”。但当前BS头在同H内的排序关联较弱，未有效利用本次噪声来超越UE强基线。此观察只约束本轮小MLP、训练目标与固定动作，不是信息论不可能性。', '',
'## 8. 成本、回顾结果与审计', '',
'- 部署候选BS条件收益头25081参数、24904线性MAC/次；194维输入中的解码特征由已有解码器产生。其他BS头是实验对照，不要求部署时全部保留。',
'- UE已有单次代理112198参数、约4.68M卷积/线性MAC。由于本轮所有方法共享UE预检，这项工作对各策略共同存在；UE“预计最好优先”可省掉额外UE收益头。',
'- 原始回顾缓存结果在results_s*.json的cached_retrospective中全量保留；本轮主表只用新噪声确认，不混用两轮绝对数字或挑选有利种子。',
'- 原编解码器与代理权重、原缓存文件哈希未变。独立audit重放当前解码特征、损失、UE分数和筛选、BS分数和排序、预算，并抽样重放每SNR首批的第0/15次未来噪声。audit.json=PASS。',
'- 输入、门限和监督来源均有记录。通过审计不等于证明新颖性、真实协议收益或跨场景泛化。', '',
'完整产物：protocol.json、locked_s*.json、results_s*.json、decisions_s*.npz、四类小头权重、fresh_cache/future缓存、audit.json、图表和复现说明。']
(ROOT / 'UE预检后BS重传_结果.md').write_text('\n'.join(lines) + '\n', encoding='utf8')

brief = ['# Q2 补充：UE预检后BS决定重传（2026-11-13）', '',
'三个源模型种子已完成。目标是检验：先由UE排除预计差的样本，在相同通过者中，BS能否利用真实接收y做出更好的追加决策。原CSI模型和UE代理保持冻结。', '',
'**结论：本轮未建立该两阶段策略优于UE单侧的证据。按平均NMSE主指标，0dB BS条件收益头在三个种子均弱于UE策略；BS收益头相对随机的9个种子×SNR配对95%区间均包含0。坏样本率次指标上，0dB BS质量最差优先优于随机，但补齐UE同目标策略后仍未优于UE，见文末事后补充。不是对所有BS方法的否定。**', '',
'设置：UE筛选门限取前250验证H的代理质量80%分位数；另用后250H设动作门限。BS输入y128＋已有特征64＋条件2；同结构比较全部样本训练和只训练通过者的收益头，另训练当前质量头。UE保留单侧收益与简单反向质量排序。', '',
'确认测试：testH600:1000，3组新当前噪声，每观测16组独立未来噪声；三种子。H曾用于别的精修实验，不是整个项目全新信源样本。旧前600H结果只作回顾，不用于调参或选择种子。', '',
'相同通过者、相同约20%额外重复次数，表中是通过者整体平均逐样本NMSE_dB的下降（3种子均值）：', '',
'| SNR | 随机 | UE单侧收益 | BS未筛选训练 | BS通过者专用收益 | BS按当前最差挑 |', '|---|---|---|---|---|---|']
for s in (0, 10, 20):
    brief.append(f'| {s} | ' + ' | '.join(f'{np.mean(vals(s,p)):.4f}' for p in
                  ('random', 'ue_gain', 'bs_unconditional_gain', 'bs_conditional_gain', 'bs_conditional_quality_worst')) + ' |')
brief += ['',
'0dB的BS条件−UE配对差分别为：' + '；'.join(f"s{r['seed']} {r['fresh_test']['0']['paired']['bs_conditional_gain_minus_ue_gain']['nmse_gain']['mean']:+.4f}dB" for r in rows) + '，各H聚类区间均低于0。条件训练也没有稳定优于同结构的全部样本训练。', '',
'0/10dB UE收益头仍与简单“预计最好优先”选择一致，不能将收益归因于新收益网络。BS按当前最差挑的字面方案也未表现出优势。', '',
'不能将失败归因于“没有噪声可救”：0dB当前误差相对无噪声解码平均恶化约' +
 f"{np.mean([r['fresh_test']['0']['diagnostics']['noise_excess_mean_db'] for r in rows]):.2f}dB，全部通过者重复可改善约" +
 f"{np.mean([r['fresh_test']['0']['diagnostics']['all_passed_repeat_gain_db'] for r in rows]):.2f}dB；但当前BS分数同H内对重传收益变化的Spearman很低（各条件完整报告）。无噪声比较是离线干预诊断，不是可加误差分解。", '',
'已补：固定验证动作门限的实际使用率漂移、同使用量随机/UE对照、抽象通过标识和BS请求开销、SGCS、坏样本越限概率和实际发射能量。开销仅为符号当量敏感性，不是已实现的控制信道。', '',
'边界：UE被排除样本的模式切换未模拟；一次完整重复平均、原SNR条件输入保持不变；未做有效SNR接收优化、HARQ往返、CSI老化、真实调度或吞吐。该结果约束本轮表示、训练和动作，不证明BS观测理论上无价值。', '',
'产物目录：`03_仿真/Q2_UE预检_BS重传_20261113/`；详细报告`UE预检后BS重传_结果.md`。源文件/旧模型未覆盖，audit.json=PASS，当前解码损失复算最大差0，抽样未来噪声、门控与预算复算通过。']
(ROOT / '实验记录追加.md').write_text('\n'.join(brief) + '\n', encoding='utf8')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ['#8897a5', '#238b8e', '#386cb0', '#e68a2e']
for ax, s in zip(axes, (0, 10, 20)):
    ps = ('random', 'ue_gain', 'bs_conditional_gain', 'bs_conditional_quality_worst')
    x = np.arange(len(ps))
    a = np.array([vals(s, p) for p in ps])
    ax.bar(x, a.mean(1), yerr=a.std(1, ddof=1), capsize=3, color=colors)
    ax.set_xticks(x, ['Random', 'UE gain', 'BS gain', 'BS worst'], rotation=25)
    ax.set_title(f'{s} dB')
    ax.set_ylabel('Mean per-observation NMSE gain (dB)')
    ax.grid(axis='y', alpha=.2)
fig.suptitle('UE-pass cohort, matched 20% repeats; mean and SD across 3 source seeds', fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / 'screen_retry_confirmation.png', dpi=180)
fig.savefig(ROOT / 'screen_retry_confirmation.pdf')
print(json.dumps(summary, indent=2))
