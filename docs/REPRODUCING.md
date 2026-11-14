# 运行与复现

## 1. 三种不同目标

| 目标 | 是否需要权重/数据 | 命令或入口 |
|---|---|---|
| 阅读已有结果 | 不需要 | `python scripts/show_results.py`，或打开 `results/` |
| 验证链路和训练入口 | 不需要外部资产 | `smoke_check.py`，小数据生成和一轮训练 |
| 重做完整科研对照 | 需要完整数据、训练或匹配权重，有些需要前序缓存 | 下面的顺序及各脚本参数 |

本次整理验证了前两类入口。没有重新训练全部正式模型或从头重跑全部 Q2 流水线，也没有声称跨环境逐位复现历史曲线。

## 2. 安装

Python 3.12 为推荐运行版本。本机验证环境为 Python 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.5.4、Matplotlib 3.11.2。`requirements.txt`是兼容范围，不是完整锁定环境；不同版本、设备及随机实现可能产生数值差别。CI 配置已提供，本次未在 GitHub 上执行。

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
# 仅需要 .mat 数据转换时：
python -m pip install -r requirements-optional.txt
```

## 3. 无权重检查

```bash
python scripts/smoke_check.py
python scripts/show_results.py
```

前者验证实际有限 QAM、补零、复噪声功率、两端梯度，以及重统计只改接收端 BN 缓冲、不改网络权重。它使用随机未训练网络，没有性能结论。

## 4. 小数据训练

```bash
python scripts/run.py --run-dir runs/demo generate --small
python scripts/run.py --run-dir runs/demo train bn --smoke
python scripts/run.py --run-dir runs/demo train gn --smoke
```

使用 256/64/64 个 H。BN/GN 权重分别保存，不覆盖正式目录。运行入口会复制 `simulation/`和`experiments/`的 Python 源码，之后所有写入落到同一 `runs/demo/` 中；脚本自身仍可用原来的相邻目录依赖。

如果已有运行目录的源码与当前发布版不同，入口会停止，请改用新目录。小数据和完整数据也不要共用目录。旧脚本的 `--out` 往往只改变 JSON，不改变权重路径，因此建议始终通过此隔离入口运行。

## 5. 后期 BN / GN 完整对照

```bash
python scripts/run.py --run-dir runs/full generate
python scripts/run.py --run-dir runs/full train bn --seeds 0 1 2 --epochs 40
python scripts/run.py --run-dir runs/full train gn --seeds 0 1 2 --epochs 40
python scripts/run.py --run-dir runs/full experiment gn_bn_unified_20261114 run_comparison.py --threads 4
python scripts/run.py --run-dir runs/full experiment gn_bn_unified_20261114 analyze.py
```

完整数据为 6000/1000/1000 H，种子 2024/7/11；生成器输出键 `x`，`float16`，形状 `(N,2,32,32)`。生成器描述中的“角时延域”不精确：实际先合成天线—时延系数，再沿时延维 DFT 得空—频域矩阵；不宣称与 COST2100 同分布。

BN/GN 分别训练，统一评估使用同 H、同标准噪声，并按各模型量化后实际功率标定。BN-S0/S2 固定同一个 y。`analyze.py`会检查**本次运行**生成的协议、原数组和哈希；不能拿已整理的历史 `results/`去冒充这次运行。

有原始本地资产时，可省去重训，复制到单独运行目录：

```bash
python scripts/import_local_assets.py --project /path/to/original/simulation --run-dir runs/replay
python scripts/run.py --run-dir runs/replay experiment gn_bn_unified_20261114 run_comparison.py
```

资产导入只复制 `data/*.npz` 与指定权重目录，不修改源文件、不上传资产。默认导入 `ckpt_rdae`、`ckpt_rdae_gn`；其路径、大小和原始 SHA256 可查 `asset_manifest.json`。默认不复制历史锁文件、协议和结果，避免旧源码哈希与整理后源码冲突。

## 6. 前期共享模型与蒸馏

在另一个完整数据运行目录中：

```bash
python scripts/run.py --run-dir runs/stage-a generate
python scripts/run.py --run-dir runs/stage-a train stage-a --only learned_soft,learned_soft_f0,learned_soft_f10,learned_soft_f20 --seeds 0,1,2 --epochs 40
python scripts/run.py --run-dir runs/stage-a train distill --mode e3a --seeds 0,1,2 --epochs 40 --out results_e3a.json
python scripts/run.py --run-dir runs/stage-a train distill --mode e3b --anchor_weights 3,1,1 --seeds 0,1,2 --epochs 40 --out results_e3b.json
python scripts/run.py --run-dir runs/stage-a train distill --mode e4 --lam 1.0 --seeds 0,1,2 --epochs 40 --out results_e4.json
```

以上列的是运行方法，不保证生成与历史相同的数值；历史精确训练协议以对应结果报告为准。阶段 A 的 `fixed_hard` 也接受端到端训练且没有独立信道码，**不是完整分离式 SSCC 基线**。连续符号分支也只是参考，不是严格性能上界。

## 7. Q1 / Q2 依赖顺序

下面均通过 `experiment <目录名> <脚本.py>`调用。全程使用同一个运行目录。参数先通过脚本 `--help`查看；没有 argparse 的后处理脚本不要用 `--help`试探，因为其顶层可能直接执行。

| 内容 | 目录 / 脚本 | 前置条件 |
|---|---|---|
| BN、小规模参数适配 | `extra_experiments_20261113/q1_adaptation.py` | 基础 BN 权重和完整数据 |
| y / 特征质量头 | `extra_experiments_20261113/q2_quality_networks.py` | 基础 BN 权重；迁移对照还需 Q1 状态 |
| 稀疏锚点 | `deployment_followup_20261113/q1_anchor_stats.py` | 基础 BN 权重，生成 `anchor_states/` |
| SNR 偏差与保守混合 | `q1_snr_uncertainty_20261114/run_uncertainty.py` | 上一步锚点 |
| UE/BS 小头、观测缓存 | `two_sided_experiments_20261113/run_cooperative.py` | 基础权重，生成质量头和缓存 |
| 重复发送未来噪声 | `two_sided_experiments_20261113/run_actions.py` | 前一步缓存/头 |
| Guo 式代理 | `guo_proxy_experiments_20261113/run_proxy.py`、`run_single_pass.py` | 双侧观测及未来缓存；按此先后运行 |
| 检测成本 | `q2_detection_cost_20261113/evaluate.py` | 质量头与 UE 代理 |
| 早期精修 | `q2_selective_refinement_20261113/run_refinement.py` | BS 质量头和前序缓存 |
| 当前精修 | `q2_refinement_followup_20261113/run_followup.py` | 早期精修模块及 BS 头 |
| UE 预检后 BS 重复 | `q2_ue_screen_bs_retry_20261113/run_screen_retry.py` | 双侧缓存、未来缓存、UE 代理和单次预测 JSON |
| 维护与匹配继续训练 | `joint_receiver_maintenance_20261113/run_joint.py`、`continuation_control.py` | 当前精修器、头、锚点；先运行主验证 |
| 固定 Ns64 专用/蒸馏补充 | `rdae_q1_supplement/run_supplement.py` | 完整数据；独立单种子设置，不等同多长度实验 |

`results/`里的摘要不能替代缓存或 checkpoint。仓库没有捆绑这些大文件，不应声称单条命令已经重现所有历史结果。旧归档用的 `archive_entry.py`与复制回本机的脚本已去除；相邻英文实验目录就是依赖本身。

## 8. 指标和统计

- `mean(NMSE_dB)`与`10log10(mean(NMSE_linear))`分别保存/标注，不能按一个常数换算。
- 平方 SGCS 与早期非平方相关量区分；最新比较使用平方定义。
- 同一 H 的多次噪声不当成独立训练种子。部分区间按 H 聚类，最新 GN 差值区间依据三种子；详见各协议。
- 历史测试 H 反复使用，三种子也不足以说明广泛泛化。重新运行不消除这一实验设计限制。
