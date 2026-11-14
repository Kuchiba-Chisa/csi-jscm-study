# 上传 GitHub

整理目录本身就是仓库根目录，不要上传原来的整个科研硬盘目录。

## Git 命令

在 GitHub 新建一个空仓库，例如 `csi-jscm-study`，不要预建 README。然后在本项目目录运行：

```bash
git init --initial-branch=main
git status
git add .
git commit -m "Organize CSI JSCM study code and experiment records"
git remote add origin https://github.com/YOUR_USERNAME/csi-jscm-study.git
git push -u origin main
```

将 URL 中的 `YOUR_USERNAME`换成自己的账号。如果 Git 提示尚未设置作者，按自己真实信息设置仓库级 `git config user.name`与`git config user.email`。认证通过 GitHub 或 Git Credential Manager 完成，不把访问令牌写进源码、README 或 remote URL。

本整理工作没有创建远程仓库、提交或推送。GitHub Actions 工作流只有上传后才会在 GitHub 上运行；本地检查通过不等同于远程 CI 已通过。

## 网页上传

也可以在仓库的 Add file / Upload files 中上传本目录的代码与文档。压缩包只方便传递，GitHub 不会把上传的 ZIP 自动展开成源码。项目文件较多，优先使用 Git 命令。

## 发布内容

`runs/`、二进制权重、数据与缓存已被 `.gitignore`排除。发布前用 `git status`看实际待提交列表；需要公开展示时，先核实 `THIRD_PARTY_NOTICES.md`所列 RD-AE 改编的许可，不要给不属于自己的代码统一加 MIT。
