# 仓库维护

仓库地址：https://github.com/Kuchiba-Chisa/csi-jscm-study 。

从仓库根目录查看变化、执行检查，再提交：

```bash
git status
git diff --check
python scripts/smoke_check.py
python scripts/smoke_recent.py
python scripts/show_results.py
git add <准备发布的代码和文档>
git commit -m "Describe the experiment or documentation update"
git push origin main
```

`runs/`、原始数据、权重及逐样本缓存由 `.gitignore` 排除。历史结果放在 `results/`，新运行通过 `scripts/run.py` 写入独立运行目录，避免覆盖已提交结果。

认证通过 Git Credential Manager 等本地机制完成；不把令牌写入源码或远程地址。代码来源及改编范围见 [第三方说明](../THIRD_PARTY_NOTICES.md)。
