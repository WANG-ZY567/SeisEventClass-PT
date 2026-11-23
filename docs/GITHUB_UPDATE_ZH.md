# GitHub 仓库更新步骤

本目录是清理后的发布副本，不包含原仓库的 `.git` 历史。更新远程仓库时，
请保留现有 Git 仓库的 `.git` 目录，不要用发布副本覆盖它。

## 推荐流程

1. 克隆或打开现有 GitHub 仓库。
2. 将本发布副本中的文件复制到该仓库工作区，保留工作区内原有 `.git`。
3. 在仓库根目录运行公开发布审计：

   ```bash
   python tools/audit_public_release.py
   ```

4. 检查变更，确认没有数据、权重或私人路径：

   ```bash
   git status --short
   git diff --check
   git diff
   ```

5. 提交并推送：

   ```bash
   git add -A
   git commit -m "Align public code with revised SeisEventClass-PT manuscript"
   git push origin main
   ```

如果默认分支不是 `main`，请将最后一条命令中的分支名替换为实际分支。

## 推送前人工确认

- `CITATION.cff` 的作者顺序和仓库地址正确；
- `docs/DATA_ACCESS.md` 符合数据提供方的使用要求；
- 不提交 `gpt2_model_path.txt`、`.npy`、`.hdf5`、检查点或实验日志；
- README 中的论文模型名保持为 `SeisMoLLM_evt6_multihead_w005`；
- 正式随机种子为 `20261107`；
- 训练类别计数为 `4629,4629,4287,1440,160,2357`。
