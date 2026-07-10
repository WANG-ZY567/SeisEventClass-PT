# EVT6 六分类数据集报告
- 生成时间：2026-01-02 01:12:51
- 数据目录：`/path/to/diting2_evt6`
- 主 meta：`/path/to/diting2_evt6/meta_evt6.csv`

---
## 0. 生成记录（可复现性）
- 记录文件：`/path/to/diting2_evt6/selection_record_evt6.json`
- 评测口径：**test 与 val 相同（test_from_val=true）**

---
## 1. 任务定义（evt6）
`evtype` -> `evt6 id`：
- `eq` -> `0`
- `ep` -> `1`
- `co` -> `2`
- `sp` -> `3`
- `se` -> `4`
- `ot` -> `5`

---
## 2. 数据来源与样本选择（来自代码）
该数据集由 `tools/prepare_diting2_evt6.py` 生成：
- **自然事件（nat）**：复用 `diting2_seismollm_full5` 已生成的波形 `.npy`（避免重新裁窗全量 natural JSON）。
  - 波形来自：`{natural_full5_dir}/waves/*.npy`（通过 `meta_full5.csv` 的 `_npy_path` 连接）。
  - `evtype` 来自：`CENC_DiTingv2_natural_earthquake.json`，按 key 关联。
- **非自然事件（non）**：从 `CENC_DiTingv2_non_natural_earthquake.hdf5` 裁窗生成 `waves_non/*.npy`。
  - 过滤条件：必须存在 `Pg`（没有 Pg 的样本会被跳过）。
  - 裁窗：以 `Pg - pre` 为左端（默认 `pre=2000`），输出长度 `in_samples=8192`。

> 字段 `_src` 标记样本来源：`nat` 或 `non`。

---
## 3. 类别均衡（当前数据）
- 总样本数：**22812**
- 来源分布（_src）：

```
_src
non    13987
nat     8825
```
### 3.1 6 类分布（_evt6）
```
_evt6
0    7078
1    5613
2    5311
3    1566
4     210
5    3034
```
### 3.2 6 类分布（名称）
```
_evt6_name
eq    7078
ep    5613
co    5311
sp    1566
se     210
ot    3034
```

---
## 4. 路径与文件存在性（当前数据）
- `_npy_path` 绝对路径：true=22812 / false=0
- 文件存在：missing=0

---
## 5. 数据划分方式
检测到**严格计数划分文件**（推荐用于与论文严格对齐）：
- `train`：`/path/to/diting2_evt6/meta_evt6_train.csv`
- `val`：`/path/to/diting2_evt6/meta_evt6_val.csv`
- `test`：`/path/to/diting2_evt6/meta_evt6_test.csv`

各 split 的真实分布：

### 5.1 train
- 行数：19770
```
_evt6
0    6478
1    5013
2    4711
3     966
4     168
5    2434
```

### 5.2 val
- 行数：1521
```
_evt6
0    300
1    300
2    300
3    300
4     21
5    300
```

### 5.3 test
- 行数：1521
```
_evt6
0    300
1    300
2    300
3    300
4     21
5    300
```

---
## 6. 如何生成“严格计数划分”（建议方案）
为了参考你贴的 CNN 论文那种“固定数量划分”的口径，建议使用：

```bash
python tools/prepare_diting2_evt6.py \
  --out_dir "/path/to/diting2_evt6" \
  --quota_mode paper_eq_ep_co \
  --balance none \
  --split_mode strict \
  --val_per_class 300 --test_per_class 300 \
  --test_from_val \
  --seed 100 --quota_seed 100
```

说明：
- 上面会输出 `meta_evt6_train.csv / meta_evt6_val.csv / meta_evt6_test.csv`，并确保每类数量严格一致。
- 训练时 `datasets/diting2_evt6.py` 会自动优先读取对应 split 的 meta 文件。
