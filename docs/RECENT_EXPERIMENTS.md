# 最近补充实验：代码、结果与运行顺序

2026-11-16 更新。Q1 先量化共享代价，再验证训练策略与固定模型后的校准。下面各组分别报告，改善量不跨实验相加。

## 先看哪些文件

| 问题 | 代码 | 已保存结果 | 结论范围 |
|---|---|---|---|
| 从训练好的共享模型蒸馏，比普通微调多多少收益？ | [run_finetune.py](../experiments/rdae_distill_finetune_20261115/run_finetune.py) | [结果](../results/rdae_distill_finetune_20261115/results.json)、[报告](../results/rdae_distill_finetune_20261115/微调实验结果.md) | 0 dB 蒸馏微调改善约0.067 dB，普通微调约0.056 dB；额外收益约0.011 dB，区间跨零 |
| 长度逐批次或逐样本分配，会改变精度与 BN 收益吗？ | [run_sampling.py](../experiments/rdae_ns_sampling_20261115/run_sampling.py) | [结果](../results/rdae_ns_sampling_20261115/results.json)、[报告](../results/rdae_ns_sampling_20261115/采样对照结果.md) | 20 dB、Ns16 时逐批次较好，Ns64 时逐样本较好；不是完整多长度共享/专用比较 |
| 输出缩放改善的是幅度还是方向？ | [affine_calib.py](../simulation/affine_calib.py)、[audit_affine.py](../experiments/output_calibration_20261115/audit_affine.py) | [复核JSON](../results/output_calibration_20261115/affine_audit.json)、[说明](../results/output_calibration_20261115/核对说明.md) | 前期 CNN 三种子0 dB仅缩放改善约0.472 dB NMSE；平方 SGCS 不变 |
| 不使用 BN，低 SNR 共享差距是否还存在？ | [run_gn_gap.py](../experiments/gn_sharing_gap_20261116/run_gn_gap.py) | [结果](../results/gn_sharing_gap_20261116/results.json)、[报告](../results/gn_sharing_gap_20261116/结果说明.md) | GN、逐样本 SNR、固定Ns64，0 dB共享仍差约0.311 dB，不能把差距只归因于BN |
| 仿射能否降低这组 GN 差距？ | [affine_gn.py](../experiments/gn_sharing_gap_20261116/output_calibration/affine_gn.py) | [结果](../results/gn_sharing_gap_20261116/output_calibration/results.json)、[报告](../results/gn_sharing_gap_20261116/output_calibration/结果说明.md) | 共享输出基本不改善；双方仿射后差距约0.310 dB，前期CNN收益不能外推 |

除 CNN 仿射复核为三个训练种子外，本页新增训练对照均为单种子。测试 H 聚类区间不代表重训不确定性。GN 对照只测0/10/20 dB三个锚点、一个反馈长度。

## 无权重阅读与检查

```bash
python scripts/show_results.py --section recent
python scripts/smoke_recent.py
```

前者读取已保存结果；后者检查混合长度链路、STE 梯度、长度次数匹配、依赖路径、仿射公式及 GN 显式信道实现。检查不需要正式数据或权重。

## 重新运行

所有命令从仓库根目录执行。新结果写入 `runs/`，已提交的历史 `results/` 不作训练输出。先生成完整数据，再运行所需实验；不要将小数据检查结果混入正式表格。

### 1. GN 共享与专用，再校准输出

```bash
python scripts/run.py --run-dir runs/gn-gap generate
python scripts/run.py --run-dir runs/gn-gap experiment gn_sharing_gap_20261116 run_gn_gap.py --epochs 40 --seed 0 --threads 8
python scripts/run.py --run-dir runs/gn-gap experiment gn_sharing_gap_20261116 audit_results.py
python scripts/run.py --run-dir runs/gn-gap experiment gn_sharing_gap_20261116 output_calibration/affine_gn.py
python scripts/run.py --run-dir runs/gn-gap experiment gn_sharing_gap_20261116 output_calibration/audit_results.py
```

先训练四个模型，生成本次运行的 `results.json`、`observations.npz`、协议与检查点，随后输出校准读取这些产物。正式审计脚本检查40轮、1000测试H等完整协议；小数据一轮检查应另建目录，不运行这些完整协议审计。

### 2. 长度采样与 BN

```bash
python scripts/run.py --run-dir runs/ns-sampling generate
python scripts/run.py --run-dir runs/ns-sampling experiment rdae_ns_sampling_20261115 run_sampling.py --smoke
python scripts/run.py --run-dir runs/ns-sampling experiment rdae_ns_sampling_20261115 run_sampling.py
python scripts/run.py --run-dir runs/ns-sampling experiment rdae_ns_sampling_20261115 audit_sampling.py
```

两组 SNR 均逐样本采样，只改变 Ns 在 batch 内的分配；每轮各长度样本数严格匹配。`run_sampling.py` 的完整实验固定40轮，检测到已有协议会停止，避免覆盖。

### 3. 共享起点蒸馏微调

```bash
python scripts/run.py --run-dir runs/finetune generate
python scripts/run.py --run-dir runs/finetune experiment rdae_q1_supplement run_supplement.py --epochs 40 --seed 0
python scripts/run.py --run-dir runs/finetune experiment rdae_distill_finetune_20261115 run_finetune.py --epochs 10 --seed 0
python scripts/run.py --run-dir runs/finetune experiment rdae_distill_finetune_20261115 audit_finetune.py
```

需要前序固定Ns64实验产生共享与专用教师检查点。两条微调分支从同一共享权重开始，均重置 Adam，比较同数据、同预算的普通继续训练与蒸馏；最终轮次和验证集选轮结果分别保存。

### 4. 前期 CNN 输出校准

按 [复现说明](REPRODUCING.md#6-前期共享模型与蒸馏)先训练三种子的 `learned_soft` 与 `learned_soft_f0`，保持同一个运行目录。也可使用原始资产导入 `ckpt_v2`。

```bash
python scripts/import_local_assets.py --project /path/to/original/simulation --run-dir runs/cnn-affine --checkpoint-folders ckpt_v2
python scripts/run.py --run-dir runs/cnn-affine train affine
python scripts/run.py --run-dir runs/cnn-affine experiment output_calibration_20261115 audit_affine.py
```

仿射用验证集拟合全局实数 a、b，再在测试集评价。纯缩放单独复核。测试集拟合列仅作诊断；由于拟合线性 MSE、报告均值 NMSE_dB，不能称为该评价指标的严格上界。

## 数据、文件和结论

- BN 重标定调整中间特征统计量，输出仿射调整最终输出；两者都冻结网络权重，但输入需求与作用不同。
- 仿射拟合需要带真值的离线验证数据；BN 重统计本身不需要 H 标签作监督。
- 大缓存、权重、原始数据不随仓库提交；审计脚本需要本次运行的完整产物，历史 JSON 不能替代它们。
- 历史结果的路径已归一化，数字保持不变。原运行代码哈希与重定位后的源码哈希不同，映射见 [本次整理清单](update_manifest_20261116.json)。
