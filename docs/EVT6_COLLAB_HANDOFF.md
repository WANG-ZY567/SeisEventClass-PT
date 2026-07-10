## EVT6 项目协作交接清单（同门共享用）

目的：让进入同一台服务器的协作者，能快速定位 **数据处理方式、协议口径、模型实现与结果证据链**，并按论文分工（摘要/技术/方法/设置/结果分析）各自取用同一套可追溯材料。

> 项目根目录：``$REPO_ROOT`

---

### A. 论文写作入口（先看这三份）

- **论文 LaTeX 主稿（当前版本）**  
  - `reports/EVT6_PAPER_DRAFT.tex`  
  - 对应：全文（术语、表格、边界说明都在这里）

- **paper-aligned 口径对齐说明（协议/数据构建/方法边界）**  
  - `reports/EVT6_METHOD_REPORT_PAPERALIGN.md`  
  - 对应：Experimental Setup（protocol、test-from-val 边界）、Method（hierarchical / multi-aux / two-stage）

- **结果写作对齐/大纲（如果持续维护）**  
  - `reports/results/EVT6_PAPER_OUTLINE_V2.md`

---

### B. 数据处理与样本构建（Experimental Setup: data construction pipeline）

- **EVT6 数据准备（生成 meta + 非自然波形裁窗）**  
  - `tools/prepare_diting2_evt6.py`  
  - 关注点：EVT6 label mapping（含 `co/ss`）、自然/非自然两支、非自然分支对 Pg 的要求、Pg-pre 裁窗（`pre=2000`）、固定长度（`in_samples=8192`）。

- **协议目录派生/可追溯划分（可选）**  
  - `tools/create_evt6_protocol_dir.py`  
  - 用途：从已有 EVT6 数据目录派生出不同 protocol variant 目录（生成 `meta_evt6_{train,val,test}.csv` 等），避免重复生成波形文件。

- **EVT6 数据集读取逻辑（训练/评测实际读什么 meta）**  
  - `datasets/diting2_evt6.py`（以及 `datasets/_factory.py`）

---

### C. 模型与损失（Method）

#### C1) CNN baselines（表格中的 CNNsmall / ResNet1D / InceptionTime）

- `models/cnnsmall1d.py`  
- `models/resnet1d.py`  
- `models/inceptiontime.py`

#### C2) GPT-2 backbone + structured auxiliary supervision（EVT6 主线）

本工作空间使用 `main.py` 作为统一训练/测试入口（支持 `--mode train/test/train_test`）。同时，每次运行会在对应的 run 目录中保存参数日志与模型快照，作为可追溯的“该 run 实现事实”。

- **每个 EVT6 run 的模型快照（用于核对该 run 的结构/改动）**  
  - 模式：`logs/<timestamp>_SeisMoLLM_evt6_*/model_backup.py`  
  - 示例：  
    - hierarchical：`logs/2026-03-23_16-52-00_SeisMoLLM_evt6_hier_sp_diting2_evt6/model_backup.py`  
    - multi-aux (w0.1)：`logs/2026-03-19_00-45-59_SeisMoLLM_evt6_multihead_w01_diting2_evt6/model_backup.py`  
    - multi-aux (w0.05)：`logs/2026-03-19_14-39-57_SeisMoLLM_evt6_multihead_w005_diting2_evt6/model_backup.py`  
    - coarse-only：`logs/2026-03-19_15-13-44_SeisMoLLM_evt6_coarse_only_diting2_evt6/model_backup.py`

- **损失实现（若需要进一步核对）**  
  - `models/loss.py`

---

### D. 训练/评测证据链（Results 的数字从哪里来）

建议把每个 run 目录视作一个“可追溯证据包”，通常包含：

- `global.log`：运行参数（最重要的口径来源）
- `test.log`：测试阶段日志
- `checkpoints/`：权重
- `tensorboard/`：曲线
- `model_backup.py`：当次模型定义快照

示例：  
- `logs/2026-03-23_16-52-00_SeisMoLLM_evt6_hier_sp_diting2_evt6/`

---

### E. 结果复核与报表生成（写论文表格/核对数字）

- **从 `test_results_*.csv` 生成 Accuracy/Macro-F1/混淆矩阵等报表**  
  - `tools/report_evt6_results.py`  
  - 输入：某个 run 目录下的 `test_results_*.csv`（由验证/测试流程在保存开关打开时生成）

- **CNN baseline 训练（生成可复核的 test_results CSV）**  
  - `tools/train_evt6_cnn_baseline.py`

---

### F. 推理侧补充（TTA / ensemble）

- TTA sweep：`tools/sweep_tta_evt6.py`  
- 概率平均 ensemble：`tools/ensemble_evt6_results.py`

---

### G. 写作分工建议（按论文章节对齐）

- **摘要/引言/相关工作**：`reports/EVT6_PAPER_DRAFT.tex` + `reports/EVT6_METHOD_REPORT_PAPERALIGN.md`  
- **方法（模型与损失）**：`reports/EVT6_PAPER_DRAFT.tex` + `logs/*/model_backup.py`（对齐“实际实现”）  
- **实验设置（数据构建/协议/训练超参）**：`tools/prepare_diting2_evt6.py` + `datasets/diting2_evt6.py` + `logs/*/global.log` + Table~1/TrainCfg 表  
- **结果表与数字复核**：`tools/report_evt6_results.py` + `logs/*/test.log`/`global.log` + `reports/EVT6_PAPER_DRAFT.tex` 表格

---

### H. 口径提醒（写作不要踩雷）

- paper-aligned 的 `test_from_val=true` 不等同严格独立 hold-out；写作需保留边界说明。  
- `sp` 仅按 EVT6 类别 id 使用，不附加未经核实的物理解释。  
- Table 3/4 为 selected runs / diagnostic comparisons，不写成严格单变量或严格 paired ablation。

---
### I. 训练/评测入口脚本清单（写技术部分按这条链路看）

建议按“入口 -> 数据 -> 模型 -> loss/metrics -> 评测/落盘”的顺序阅读。

#### I1) 总入口（训练/测试的统一入口）
- `main.py`
  - 负责：参数解析（argparse）、logdir 组织、调用训练/测试 worker
  - 相关：`--mode train/test/train_test`，以及 EVT6 常用的 checkpoint/两阶段/日志控制参数

#### I2) 训练/验证/测试主流程（核心循环与评测位置）
- `training/train.py`
  - 负责：DataLoader 构建、model/optimizer/scheduler 初始化、训练循环、调用验证、保存 checkpoint
- `training/validate.py`
  - 负责：val/test 循环、metrics 计算、可选保存 `test_results_*.csv`、（可选）TTA 前向
- `training/test.py`
  - 负责：test_worker（构建 test loader + 调用 `validate(...)`）
- `training/postprocess.py`
  - 负责：把模型输出转成指标计算需要的格式、结果落盘（`ResultSaver`）
- `training/preprocess.py`
  - 负责：把 `datasets/*` 读出的 `event/meta` 组装成：
    - model inputs（如三分量波形）
    - loss targets（one-hot / 多头结构的 targets）
    - metrics targets（one-hot，用于 `utils/metrics.py`）
    - meta_data_json（用于保存 test_results）

#### I3) 数据集实现（EVT6 读什么、怎么对齐 meta）
- `datasets/diting2_evt6.py`
  - 负责：读取 `meta_evt6_{train,val,test}.csv` 或回退到 `meta_evt6.csv`，加载波形 `.npy`，产出 `event` 与 `meta`
- `datasets/_factory.py`
  - 负责：dataset registry 与 `build_dataset(...)`

#### I4) 模型实现与模型注册（Method 写作主要依据）
- `models/SeisMoLLM.py`
  - 负责：多尺度卷积 tokenizer + GPT-2 backbone + EVT6 头部（flat/hier/multihead/coarse-only 等）
  - 注意：具体某次 run 的“当时模型结构事实”以 `logs/*/model_backup.py` 为准
- `models/_factory.py`
  - 负责：model registry（`register_model/create_model`）、checkpoint 的 load/save
- `models/loss.py`
  - 负责：CE / multi-task 组合损失（ConbinationLoss）等实现

#### I5) 指标实现（Results/Discussion 引用的 Accuracy/Macro-F1 等从这里算）
- `utils/metrics.py`
  - 负责：accuracy/precision/recall/f1 等指标的 batch 累积与汇总