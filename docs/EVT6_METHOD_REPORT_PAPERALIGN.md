# EVT6 六分类任务方法总结（paper-align 版本）

- **生成时间**：2026-01-02
- **代码目录**：``$REPO_ROOT`
- **数据目录**：`/path/to/diting2_evt6`
- **数据报告**：``$REPO_ROOT/reports/EVT6_DATASET_REPORT_PAPERALIGN.md`

---

## 1. 目标与对齐口径

- **目标**：用 SeisMoLLM（微调 GPT-2）完成 DiTing2.0 的 `evtype` 六分类，并与另一篇 CNN 论文做公平对比。
- **对齐策略**：
  - **EQ/EP/CO 总量对齐论文**：EQ=7078，EP=5613，CO=5311
  - **CO 定义**：论文中的 **CO(坍塌)** 在 DiTing2.0 的 `evtype` 编码里对应 **`ss`**，因此本项目把 `co` 作为 `ss` 的别名（统一为 `evt6 id=2`）。
  - **其它类**：`sp/se/ot` **不够就全用**，不做重复过采样。
  - **评测口径**：论文无独立测试集，因此采用 **test=val**（`test_from_val=true`），即 test 阶段评测同一份验证集。

---

## 1.1 本方法到底是什么（不是传统 CNN）

本项目的 `SeisMoLLM_evt6` 属于 **“卷积嵌入 + 预训练 GPT‑2 + LoRA 参数高效微调（PEFT）”** 的混合架构（foundation model fine-tuning），不是纯 CNN 分类器：

- **输入**：三分量波形 `z/n/e`（长度 8192）
- **前端**：多尺度 1D 卷积块（`Multi_Scale_Conv_Block`）把波形编码成 patch/token 表示
- **主干**：预训练 `GPT2Model` 作为序列建模器（使用 `inputs_embeds`，不走词表）
- **微调策略**：默认 `pretrain=True, freeze=True`，冻结大多数 GPT‑2 权重，只训练 LN/wpe 以及 **LoRA** 注入的少量参数
- **任务头**：evt6 六分类头输出类别概率（交叉熵损失）

你那篇 CNN 论文的方法是“纯 CNN 端到端分类”，这里的对比核心就是：**同一划分/同一数据口径下，CNN vs LLM+LoRA 微调的效果差异**。

---

## 2. 数据构建（保持现有取数逻辑）

数据由 `tools/prepare_diting2_evt6.py` 生成，来源分两部分：

- **自然（nat）**
  - **波形来源**：复用 `diting2_seismollm_full5` 预处理结果（`meta_full5.csv` 的 `_npy_path` 指向 `waves/*.npy`）
  - **标签来源**：`CENC_DiTingv2_natural_earthquake.json` 的 `evtype`
- **非自然（non）**
  - **波形来源**：`CENC_DiTingv2_non_natural_earthquake.hdf5` 裁窗生成 `waves_non/*.npy`
  - **过滤**：必须有 `Pg`（无 Pg 样本被跳过）
  - **裁窗**：以 `Pg - pre` 为左端（默认 `pre=2000`），输出 `in_samples=8192`
  - **标签来源**：`CENC_DiTingv2_non_natural_earthquake.json` 的 `evtype`

---

## 3. 类别映射（evt6）

- `eq -> 0`
- `ep -> 1`
- `co/ss -> 2`
- `sp -> 3`
- `se -> 4`
- `ot -> 5`

---

## 4. 样本选择、划分与记录（可复现）

- **配额模式**：`quota_mode=paper_eq_ep_co`
  - 选择总量：`final_counts` = {0:7078, 1:5613, 2:5311, 3:1566, 4:210, 5:3034}
- **划分方式**：`split_mode=strict`
  - 默认尝试 `val_per_class=300, test_per_class=300`
  - 对小类（例如 `se`）若不足以满足固定 val/test，会自动回退为 **按比例切分且全用不丢**（详见记录文件的 `actual_per_class`）
- **test=val**：`test_from_val=true`（`meta_evt6_test.csv` 与 `meta_evt6_val.csv` 完全一致）
- **记录文件**（强制保留）：`selection_record_evt6.json`

---

## 5. 一键复现（数据生成命令）

```bash
source /path/to/venv/bin/activate
cd `$REPO_ROOT

python tools/prepare_diting2_evt6.py \
  --out_dir "/path/to/diting2_evt6" \
  --quota_mode paper_eq_ep_co \
  --balance none \
  --split_mode strict \
  --val_per_class 300 --test_per_class 300 \
  --test_from_val \
  --seed 100 --quota_seed 100
```

---

## 6. 训练与评测命令（从零开始）

```bash
source /path/to/venv/bin/activate
cd `$REPO_ROOT

python main.py \
  --mode train_test \
  --model-name SeisMoLLM_evt6 \
  --device cuda:0 \
  --data /path/to/diting2_evt6 \
  --dataset-name diting2_evt6 \
  --shuffle true --workers 0 \
  --in-samples 8192 \
  --batch-size 32 \
  --augmentation false \
  --norm-mode max \
  --label-width 0 --label-shape 2 \
  --epochs 30 \
  --use-torch-compile false
```

---

## 7. Epoch 设为多少合适？

由于你采用 **test=val**（论文口径），为了避免“用评测集挑最优 epoch”产生信息泄露，推荐：

- **推荐 epoch：30（固定训练轮数）**
  - 直接报告 epoch=30 的结果（或报告全程曲线），更符合“固定训练预算”的对比方式。
  - 若训练仍在上升趋势，可再做一个 **epoch=50** 的补充实验，但依然建议按固定轮数报告。


