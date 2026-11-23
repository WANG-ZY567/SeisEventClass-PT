# 论文与代码对应说明

本文档说明修订稿中正式模型、消融实验和代码入口的对应关系。

## 1. 正式任务

- 输入：50 Hz 三分量 `Z/N/E` 波形，每个分量 8,192 点（163.84 s）。
- 类别：`eq`、`ep`、`ss`、`sp`、`se`、`ot`，代码编号依次为 `0..5`。
- 划分：按事件划分训练、验证和测试集；同一事件的台站记录不跨集合。
- 正式模型：`SeisMoLLM_evt6_multihead_w005`。
- 随机种子：`20261107`。

## 2. 模型流程

```text
[B, 3, 8192] 三分量波形
    -> 多尺度一维卷积分词器
    -> [B, 96, 1024] 局部特征
    -> 每 8 个相邻时间位置重排为一个连续标记
    -> [B, 128, 768]
    -> GPT-2 前 3 层（inputs_embeds）+ LoRA
    -> EVT6 主分类头及辅助分类头
    -> 六分类输出
```

连续波形标记直接通过 `inputs_embeds` 输入 GPT-2，不进行文本分词或
词表映射。

## 3. 论文名称与实现名称

| 论文名称 | 代码实现 | 说明 |
|---|---|---|
| SeisEventClass-PT | `SeisMoLLM_evt6_multihead_w005` | 完整模型 |
| SeisEventClass-RandInit | `EVT6_ABLATION=random_lora` | 随机初始化 GPT-2，保留 LoRA 与匹配架构 |
| SeisEventClass-SingleScale | `EVT6_ABLATION=single_scale` | 单尺度卷积分词器 |
| SeisEventClass-NoLoRA | `EVT6_ABLATION=pretrained_no_lora` | 预训练 GPT-2，不使用 LoRA |
| SeisEventClass-FrozenGPT2 | `EVT6_ABLATION=pretrained_frozen` | GPT-2 完全冻结 |

`w005` 表示两个辅助损失的权重均为 `0.05`。上述消融均通过
`tools/evt6_paired_ablation.py` 构建，并由
`tools/run_evt6_paired_ablation.sh` 在相同划分和训练设置下运行。

## 4. 关键代码位置

| 功能 | 代码位置 |
|---|---|
| EVT6 数据加载 | `datasets/diting2_evt6.py` |
| 正式模型及 LoRA | `models/SeisMoLLM.py` |
| 损失和任务注册 | `config.py` |
| 训练、验证和测试 | `main.py`、`training/` |
| 正式训练脚本 | `scripts/train_paper_evt6.sh`、`.ps1` |
| 正式评估脚本 | `scripts/eval_paper_evt6.sh`、`.ps1` |
| 逐类别及事件级分析 | `tools/analyze_evt6_predictions.py` |
| ComCat 严格六类输出分析 | `tools/evaluate_comcat_shared_binary.py` 中的 `strict_six_class_output` |

## 5. 复现边界

仓库不分发 DiTing 2.0、ComCat 原始波形、生成的 `.npy`、GPT-2 权重
或训练检查点。复现论文数值需要取得原始数据并按仓库提供的统一筛选、
预处理和事件级划分流程在本地生成数据。正式配置以
`docs/REPRODUCIBILITY.md` 和 `scripts/train_paper_evt6.*` 为准。
