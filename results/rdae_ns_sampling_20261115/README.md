# 长度采样协议与 BN 重标定

本轮只检验一个问题：SNR 已逐样本采样时，把 Ns 从逐批次分配改为逐样本分配，能否改善基础重建，并使目标条件 BN 重标定收益减小？

## 四格对照

| 新训练协议 | 原始BN S0 | 目标条件重标定 S2 |
|---|---|---|
| 同batch相同Ns | 评估 | 评估 |
| 同batch混合Ns | 评估 | 评估 |

两组都从相同初始化重新训练40轮，不直接拿新模型与历史训练权重比较。每epoch抽取一组batch长度，将同样的长度池随机打乱分配给逐样本组，确保两组各长度出现总次数严格相同。逐样本组是次数匹配的随机分配，不宣称为严格独立同分布采样。

SNR逐样本U(0,20)，CSI顺序、标准噪声、优化器和轮数相同。固定16-QAM、RD-AE改编骨干，两端含SNR AF；不新增Ns条件输入。每epoch从6000个训练H随机顺序中取46个完整batch，丢弃最后不足128的部分，两组完全一致。

测试用1000H、每H三组噪声，九个条件为Ns16/32/64与SNR0/10/20。测试条件实际发送功率与目标BN统计量都只用训练集校准。S0和S2使用相同权重、发送端、功率及接收y；S2只改解码器BN缓冲。

## 读什么

- `采样对照结果.md`：固定协议与主表。
- `sampling_comparison.png`：四状态的NMSE曲线。
- `recalibration_gain.png`：两组重标定收益。
- `results.json`、`observations.npz`：指标与逐样本数据。
- `trace_batch_ns.json`、`trace_sample_ns.json`：逐轮训练损失、长度次数和时间。
- `batch_ns_final.pt`、`sample_ns_final.pt`：最终40轮模型。
- `batch_ns/bn_states/`、`sample_ns/bn_states/`：每个测试条件的解码器BN缓冲。
- `protocol.json`：训练前保存的配置、源文件哈希。
- `audit.json`、`independent_audit.json`：原文件未覆盖、同y对照与保存状态前向重放。

`*_progress.pt` 是每5轮保存的恢复检查点，包含优化器状态；最终以 `*_final.pt` 为正式模型。

## 在仓库中运行

代码位于 `experiments/rdae_ns_sampling_20261115/`，本目录仅保存历史结果和报告。

复现命令与前序依赖见 [补充实验运行顺序](../../docs/RECENT_EXPERIMENTS.md)。新运行的权重、缓存和日志写入 `runs/`，不包含在本结果目录。

## 边界

单初始化/训练种子、合成CSI、AWGN、既有测试信源集合。三次噪声与按H置信区间不能替代独立训练重复。没有各条件专用模型，因此这里测量的是采样协议与统计适配的效果，不是完整的多长度共享代价。
