# EVT6 数据预处理代码说明

仓库包含数据筛选、截窗、质量控制、信号处理、事件级划分和模型输入
标准化代码。正式流程不是由单个脚本完成，而是拆分成可审计的阶段。

## 1. 原始记录筛选与截窗

脚本：`tools/build_diting2_evt6_review_protocol.py`

主要功能：

- 读取 DiTing 2.0 两份 JSON 元数据和 HDF5 波形；
- 按事件来源和 EVT6 标签收集候选记录；
- 检查三分量可读性、有限值和恒定分量；
- 要求有效 `Pg`、`Sg`，并满足 `Pg < Sg`；
- 以 `Pg` 为参考截取三分量窗口；
- 输出波形数组、清单、剔除原因及数据构建报告。

修订稿正式数据使用 50 Hz、8,192 点窗口，窗口从 `Pg` 前约 500 点
开始，并要求 `Sg` 位于窗口内。因此运行时必须显式传入：

```bash
--in_samples 8192 --pre_samples 500 --require_valid_sg
```

该脚本保留的 `--pre_samples 2000` 默认值属于早期实验设置，不能用于
复现修订稿数据。

## 2. 统一信号处理

脚本：`tools/create_evt6_bandpass_dataset.py`

对每条截取后的三分量记录统一执行：

1. 逐分量线性去趋势；
2. Tukey 锥形衰减（两端各 5%）；
3. 1-20 Hz 四阶 Butterworth 零相位带通滤波。

正式参数为：

```bash
python tools/create_evt6_bandpass_dataset.py \
  --source-dir /path/to/cropped_evt6 \
  --output-dir /path/to/evt6_bp1_20 \
  --sampling-rate-hz 50 \
  --low-hz 1 \
  --high-hz 20 \
  --order 4
```

该步骤不进行重采样、补零或均值填充。

## 3. 补充天然事件源中的其他 EVT6 标签

脚本：`tools/augment_evt6_natural_classes.py`

该脚本从天然事件源中检查 `ep`、`ss`、`sp`、`se` 和 `ot` 记录，处理
两份源文件之间的重复事件标识和标签冲突，并对新增记录执行与正式
数据一致的 8,192 点截窗、500 点 Pg 前置、去趋势、Tukey 衰减和
1-20 Hz 零相位带通滤波。

输入的 `--source-dir` 应当是已经完成相同信号处理的基础数据目录，
以保证基础记录和新增记录的处理一致。

## 4. 事件级划分

脚本：`tools/resplit_evt6_event_8020.py`

该脚本按数值型原始事件编号在每个类别内排序，并生成事件互不重叠的
训练、验证和测试清单。同一事件的所有保留台站记录只进入一个集合。
脚本同时输出事件交集检查和各类别数量报告。

## 5. 模型输入标准化

代码：`training/preprocess.py`

训练、验证和测试读取波形后，使用 `--norm-mode std` 对每个分量分别
去均值并按标准差归一化。训练增强仅在训练集启用，验证和测试阶段
关闭增强。

## 6. ComCat 外部数据准备

脚本：`tools/prepare_comcat_evt6_zeroshot.py`

该脚本负责 ComCat 外部队列的元数据筛选、三分量波形读取、P/S 到时
检查、窗口构建和质量控制。ComCat 数据仅用于外部测试，不参与训练、
超参数调整或检查点选择。

## 7. 预处理结果检查

开始训练前建议运行：

```bash
python tools/validate_evt6_splits.py /path/to/final_evt6 --check-files
python tools/audit_evt6_before_training.py --help
```

检查重点包括：三个集合之间不存在事件交叉、波形文件可访问、六类标签
完整、训练类别计数与损失配置一致，以及每条波形形状为 `[3, 8192]`。
