## EVT6 交叉应用新增实验：数据划分与指标定义（记录稿）

生成时间：2026-03-31  
仓库根目录：``$REPO_ROOT`

本文档目的：
- 先把 **三个新增实验**（station transfer / event-level aggregation / class-conditioned phase picking）所需的 **数据与协议**准备好并可追溯记录；
- 明确每个实验的 **评估指标** 与 **输出物**；
- 不改坏现有 paper-aligned 主线流程。

> 说明：下文的“xapp”指扩展元数据版本（在 EVT6 分类 meta 的基础上加入 event/station/picking 等字段）。

---

### 0. 已生成的数据目录（xapp）

#### 0.1 xapp 数据目录（含扩展字段 + strict split）
- **数据目录**：`/path/to/diting2_evt6_holdout_xapp`
- **文件**：
  - `meta_evt6.csv`
  - `meta_evt6_train.csv`
  - `meta_evt6_val.csv`
  - `meta_evt6_test.csv`
  - `selection_record_evt6.json`
  - `waves_non/`（non 样本的裁窗波形）

#### 0.2 xapp meta 的关键字段（用于 3 个实验）
在原始 EVT6 字段（`key, part, _src, _rid, _evtype, _evt6, _npy_path`）基础上，xapp 增加：
- **跨台站/跨事件标识**
  - `src_domain`：`nat` / `non`
  - `event_id_raw`：从 key 解析得到（去掉前缀后 `event_station` 的 event 部分）
  - `station_id_raw`：从 key 解析得到（station 部分）
  - `event_uid`：`{src_domain}_{event_id_raw}`（避免 `nat_*` 与 `non_*` 的 event id 冲突）
  - `station_uid`：`{src_domain}_{station_id_raw}`（如需要按域区分台站）
  - `trace_uid`：`{src_domain}_{event_id_raw}_{station_id_raw}_{part}`（唯一样本）
- **picking/震相相关（用于 class-conditioned phase picking）**
  - `p_pick, s_pick`：对齐到裁窗后的波形坐标系（与 npy 对齐）
  - 以及若可用：`Pg, Sg, Pg_res, Sg_res, Pg_azi, Sg_azi, Pg_dist, Sg_dist`
- **震级/其它（可选用于分析）**
  - `mag, magtype, se_time, sn_time, se_mag, sn_mag`
- **台站信息（若可用）**
  - `net, sta_id`（来自 full5 meta 的 best-effort 字段）

> 重要：`event_uid` 是 event-level 聚合与“更严格的 station+event 独立划分”的基础字段。

---

### 1) 实验一：台站迁移（Station Transfer）

#### 1.1 实验目的
在部分台站训练，在 **未见台站** 测试，评估跨台站泛化。

#### 1.2 两种可选协议（是否允许 event overlap）
- **协议 A（默认优先）**：仅台站独立（station-disjoint）
  - 约束：train/val/test 的 `station_id_raw` 不重叠
  - 允许：同一 `event_uid` 可能同时出现在 train/test（不同台站）
  - 优点：最贴近“台站迁移”问题本身；实现最稳
- **协议 B（更严格）**：台站独立 + 事件独立（station-disjoint + event-disjoint）
  - 约束：train/val/test 的 `station_id_raw` 不重叠，且 `event_uid` 也不重叠
  - 优点：避免“同一事件跨台站泄漏”
  - 代价：样本量会更小，且需要额外的 split 逻辑（当前脚本默认先做协议 A）

#### 1.3 当前已实现的数据划分脚本（协议 A）
- 脚本：`tools/create_evt6_station_transfer_dir.py`
- 输入：xapp 目录下的 `meta_evt6.csv`（要求含 `station_id_raw`）
- 输出：新的 protocol 目录：
  - `meta_evt6.csv`（链接或复制）
  - `meta_evt6_{train,val,test}.csv`（按 station 划分）
  - `selection_record_evt6_station_transfer.json`（统计与可追溯信息）

#### 1.4 划分方式（当前脚本）
- 先收集所有 `station_id_raw` 的集合
- 随机打乱（seed 固定），按 **台站数量比例**切分：
  - `test_station_frac`（默认 0.2）
  - `val_station_frac`（默认 0.1）
  - 其余为 train stations
- 再把样本按 station 归入 train/val/test
- split 内部再按同一个 seed 打乱（保证可复现）

#### 1.5 运行命令（生成 station-transfer 协议目录）

```bash
source /path/to/venv/bin/activate
cd `$REPO_ROOT

python tools/create_evt6_station_transfer_dir.py \
  --src_dir "/path/to/diting2_evt6_holdout_xapp" \
  --dst_dir "/path/to/diting2_evt6_station_transfer" \
  --seed 100 \
  --test_station_frac 0.2 \
  --val_station_frac 0.1 \
  --link_mode symlink
```

生成后，划分统计写在：
- `.../selection_record_evt6_station_transfer.json`

#### 1.6 station transfer 的评估指标（分类）
- **Accuracy**
- **Macro-F1**
- **per-class F1**
- **confusion matrix**

（结果生成工具复用 `tools/report_evt6_results.py`，输入为 run 目录中的 `test_results_*.csv`）

---

### 2) 实验二：事件级聚合（Event-level Aggregation）

#### 2.1 实验目的
把 waveform/station-level 的预测聚合成 **event-level** 决策，评估是否更符合监测流程与更稳定。

#### 2.2 数据前提（当前 xapp 已满足）
- `event_uid`：可作为 event 分组 id（避免 nat/non 冲突）
- 在 `--save-test-probs true` 时，`test_results_*.csv` 会包含 `prob_evt6_0..5`（用于概率聚合与置信度加权）

#### 2.2.1 事件级聚合“数据集/协议目录”
- **不需要额外生成新的 dataset 目录**。直接基于你训练跑出的 `test_results_*.csv` 做聚合即可。
- 依赖字段：
  - `event_uid`（来自 xapp meta；会跟随 `ResultSaver` 写入 test_results）
  - `tgt_evt6` / `pred_evt6`
  - `prob_evt6_0..5`（若做概率平均/置信度加权，需开启 `--save-test-probs true`）

#### 2.3 计划实现的聚合策略（将以工具脚本形式落地）
基于 `test_results_*.csv`：
- **Hard voting**：对 `pred_evt6` 多数投票
- **Probability averaging**：对 `prob_evt6_k` 按 event_uid 做平均后取 argmax
- **Confidence-weighted averaging**：以每条样本的 `max(prob)` 为权重做加权平均后取 argmax

#### 2.4 event-level 的评估指标
同分类指标，但统计粒度换为 event：
- event-level **Accuracy**
- event-level **Macro-F1**
- event-level **per-class F1**
- event-level **confusion matrix**

以及附加分析（用于论文更“应用化”叙述）：
- event 内样本数分布（1 / 2–3 / 4–5 / 6+）
- 分桶后的 event-level accuracy（或 Macro-F1）

> 注：event-level ground truth 需来自 `tgt_evt6` 的一致性（每个 event 内的真值应一致；若出现不一致，需要在脚本里显式报告并处理策略）。

---

### 3) 实验三：Class-conditioned Phase Picking

#### 3.1 实验目的
探索 EVT6 类别信息是否能作为条件，辅助下游 picking（不预设一定提升）。

#### 3.2 数据前提（当前 xapp 已初步具备）
- `p_pick, s_pick` 已写入 meta（对齐裁窗后的波形坐标系）
- 可同时使用 `evt6`（分类标签）做 oracle conditioning 或训练条件

#### 3.2.1 picking subset 协议目录（建议先生成）
由于部分样本可能缺少 `s_pick` 或越界，建议先派生一个 “picking-ready 子集”目录，避免训练/评测中混入无效样本。

- 脚本：`tools/create_evt6_picking_subset_dir.py`
- 输入：`.../meta_evt6_{train,val,test}.csv`
- 输出：仅保留满足条件的子集 split + 记录文件 `selection_record_evt6_picking_subset.json`

生成命令（默认只要求 p_pick；若要做 P+S 联合 picking，请加 `--require_s`）：

```bash
source /path/to/venv/bin/activate
cd `$REPO_ROOT

python tools/create_evt6_picking_subset_dir.py \
  --src_dir "/path/to/diting2_evt6_holdout_xapp" \
  --dst_dir "/path/to/diting2_evt6_picking_subset" \
  --in_samples 8192 \
  --link_mode symlink
```

#### 3.3 三个版本（计划）
1. **Baseline picker**：不使用类别条件
2. **Oracle-conditioned picker**：用真实 `evt6` 条件
3. **Predicted-conditioned picker**：用分类器预测（hard label 或 prob）

#### 3.4 picking 指标（沿用仓库已有 Metrics）
（以 `utils/metrics.py` 的 `ppk/spk` 分支为准）
- **P-phase F1**
- **S-phase F1**
- （若回归形式可用）P/S **MAE** 等

> 仍需要补齐：picking 任务的数据接口/训练配置如何直接复用当前 EVT6 波形与这些 `p_pick/s_pick` 字段（后续会在工具/数据管线层落地，不在本记录稿里假设已完成）。

---

### 4) 统一的结果落盘与报告生成（分类部分已具备）

#### 4.1 run 目录的证据链
每次训练会落在 `logs/<timestamp>_SeisMoLLM_evt6_..._diting2_evt6/`，包含：
- `global.log`（参数）
- `checkpoints/`
- `test.log`
- `test_results_*.csv`（若 `--save-test-results true`）

#### 4.2 从 test_results 生成分类报告
```bash
source /path/to/venv/bin/activate
cd `$REPO_ROOT

python tools/report_evt6_results.py --run_dir "<RUN_DIR>"
```

输出包括：
- `EVT6_TEST_REPORT.md`
- `confusion_matrix_evt6.csv`

