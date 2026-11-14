# 文献、改编来源与许可说明

本项目不把下列论文方法或已有技术当作原创。文献引用不等于代码许可。

## RD-JSCC 自编码器结构

- 论文：*Residual Diffusion Models for Variable-Rate Joint Source Channel Coding of MIMO CSI*，https://arxiv.org/abs/2505.21681 。
- 作者仓库：https://github.com/UTAustin-ITML/rd-jscc 。
- 结构来源：作者仓库 `modules/vq_compressor.py` 的 AF、AFCsiNetPlus 编解码器。
- 本仓库对应：`simulation/model_rdae.py`。这是该 AE 结构的 PyTorch 移植/改编，涉及 BN、卷积、AF 和残差块。未包含上游完整 TensorFlow 工程、VQ、扩散模型及官方数据。
- 同权重编码器对齐是局部验证，不等于完整端到端复现。有限 16-QAM、补零、合成数据和后续监测属于本项目设置。

整理时检查的本地上游快照中未发现 LICENSE 文件，未据此确认可按 MIT/Apache 等许可再授权。保留来源并未解决再分发许可问题；如将此改编作为公开开源发行，应向原作者核实适用许可。本整理版不添加整体开源许可证。

## 其他主要文献

| 主题 | 文献 |
|---|---|
| 学习型 CSI 压缩 | Wen et al., *Deep Learning for Massive MIMO CSI Feedback*（CsiNet），https://arxiv.org/abs/1712.08919 |
| 有噪端到端 JSCC | Bourtsoulatze et al., *Deep Joint Source-Channel Coding for Wireless Image Transmission*，https://arxiv.org/abs/1809.01733 |
| SNR 条件共享 | Xu et al., ADJSCC，https://arxiv.org/abs/2012.00533 |
| CSI 联合信源信道编码 | CSI-DeepJSCC，https://doi.org/10.1109/JSAC.2022.3221963 |
| 有限星座约束 | DeepJSCC-Q，https://doi.org/10.1109/JSAIT.2022.3231042 |
| UE 侧 CSI 质量监测 | Guo et al., *Performance Monitoring-Enabled Reliable AI-Based CSI Feedback*，IEEE TWC 24(1):197–212，2025，https://doi.org/10.1109/TWC.2024.3490600 |

UE 代理是借鉴 Guo 等思路的本项目实现，不是官方代码复现。BN 统计量调整、蒸馏、损失重加权、GN 与残差 CNN 均为已有技术。

没有把下载的论文、学校模板、校徽、答辩材料或第三方数据作为本仓库代码附件再发布。
