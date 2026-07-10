# DiTing2.0 数据集适配 SeisMoLLM 使用指南

> 完整步骤：从数据预处理到模型训练

---

## 📋 前置准备

### 1. 环境配置

```bash
# 安装依赖
cd /path/to/SeisEventClass-PT
pip install -r requirements.txt

# 下载 GPT-2 预训练权重
# 方法1：使用 huggingface-cli
huggingface-cli download gpt2 --local-dir ./pretrained/gpt2

# 方法2：Python 脚本下载
python -c "from transformers import GPT2Model; GPT2Model.from_pretrained('gpt2').save_pretrained('./pretrained/gpt2')"
```

### 2. 配置 GPT-2 路径

推荐运行下载脚本自动生成 `gpt2_model_path.txt`：

```bash
python download_gpt2_simple.py
```

也可以通过环境变量显式指定：

```bash
export SEISMOLLM_GPT2_PATH=/path/to/gpt2
```

### 3. 注册 DiTing2 数据集

编辑 `datasets/__init__.py`，在文件末尾添加：

```python
from .diting2 import DiTing2, DiTing2_light
```

**⚠️ 硬性规则**：
- `--dataset-name` 参数必须和 dataset registry 中注册的名字**完全一致**
- 本指南使用 `diting2`（对应 `DiTing2._name = "diting2"`）
- 如果名字不匹配，会加载到旧的 DiTing 类或直接报错
- 验证方法：运行后检查日志中的 "Dataset: diting2_train" 字样

---

## 🔧 数据预处理

### Step 1: 确认方位角定义（可选但推荐）

运行以下脚本验证 `Pg_azi` 是 azimuth 还是 back-azimuth：

```python
import json
import numpy as np
from obspy.geodetics import gps2dist_azimuth

json_path = r'/path/to/CENC_DiTingv2_natural_earthquake.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

diffs = []
keys = list(data.keys())[:1000]  # 采样 1000 个

for k in keys:
    meta = data[k]
    required_fields = ['event_latitude', 'event_longitude', 'station_latitude', 'station_longitude', 'Pg_azi']
    if all(field in meta and meta[field] not in [None, ''] for field in required_fields):
        ev_lat, ev_lon = float(meta['event_latitude']), float(meta['event_longitude'])
        st_lat, st_lon = float(meta['station_latitude']), float(meta['station_longitude'])
        pg_azi = float(meta['Pg_azi'])
        
        # 计算 back-azimuth（从台站指向震源的方向，常见定义）
        _, _, baz_calc = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
        diff = abs((pg_azi - baz_calc + 180) % 360 - 180)
        diffs.append(diff)

print(f"平均差值: {np.mean(diffs):.1f}°")
print(f"中位数差值: {np.median(diffs):.1f}°")
# 如果接近 0°：Pg_azi 是 back-azimuth，--azi_offset 0
# 如果接近 180°：Pg_azi 是 azimuth，--azi_offset 180
```

**注意**：`gps2dist_azimuth(ev, st)` 返回的 back-azimuth 是**从台站指向震源**的方向（常见地震学定义）。

### Step 2: 运行预处理脚本

**⚠️ 裁窗策略声明**（重要）：
> 本指南采用【**离线裁窗到 8192**】路线：预处理时以 Pg 为中心裁到 8192，训练时保持在线裁窗不再移动 Pg（确保不会二次裁窗导致索引漂移）。

**方案 A：全标签子集（严格复现，5 任务）**

```bash
cd /path/to/SeisEventClass-PT

python tools/prepare_diting2.py --h5 /path/to/CENC_DiTingv2_natural_earthquake.hdf5 --json /path/to/CENC_DiTingv2_natural_earthquake.json --out_dir /path/to/diting2_preprocessed --meta_csv meta_full5.csv --in_samples 8192 --pre 2000 --azi_offset 0 --require_full5
```

**关键说明**：
- **pmp 口径**：使用 `Pg_polarity ∈ {U,C,R,D}`，映射 `U/C→0, R/D→1`（不是只用 U/D）
- **dpk 筛选**：要求 Pg **和** Sg 都存在（不是只要 Pg）
- **裁窗方式**：以 Pg 为中心，`[Pg - 2000, Pg + 6192]`，边界 pad 0

**方案 B：dpk 专用（Phase Picking，要求 Pg+Sg）**

```bash
python tools/prepare_diting2.py --h5 /path/to/CENC_DiTingv2_natural_earthquake.hdf5 --json /path/to/CENC_DiTingv2_natural_earthquake.json --out_dir /path/to/diting2_dpk --meta_csv meta_dpk.csv --in_samples 8192 --pre 2000
```

**⚠️ 重要**：dpk 训练集默认要求同时存在 `Pg` 和 `Sg`；若保留缺 `Sg` 样本，必须对 S 通道 loss 做 mask（需修改代码）。

**预期输出**：
- 方案 A：约 **150-200k** 条波形（5 任务标签齐全，按 U/C/R/D 口径）
- 方案 B：约 **800k** 条波形（Pg 和 Sg 都存在）

---

## 🚀 模型训练

### Phase 1: 先跑 Phase Picking（dpk）

**目的**：验证数据管线、裁窗、标签生成、训练流程

```bash
cd /path/to/SeisEventClass-PT

python main.py --seed 0 --mode train --model-name SeisMoLLM_dpk --log-base ./logs --log-step 300 --data /path/to/diting2_dpk --dataset-name diting2 --data-split true --train-size 0.8 --val-size 0.1 --shuffle true --workers 4 --in-samples 8192 --batch-size 32 --augmentation true --epochs 200 --patience 30 --base-lr 0.0005 --max-lr 0.001 --warmup-steps 2500 --down-steps 3000 --label-width 0.5 --label-shape gaussian --norm-mode std
```

**⚠️ 关键参数说明**（显式指定，避免依赖默认值）：
- `--in-samples 8192`：输入波形长度（**必须与预处理一致**）
- `--label-width 0.5`：软标签宽度 0.5 秒 = 25 samples @ 50Hz
- `--label-shape gaussian`：高斯形状软标签
- `--norm-mode std`：标准化方式（按标准差归一化）
- `--dataset-name diting2`：**必须与 datasets/__init__.py 中注册的名字完全一致**

**关键检查点**（先小规模验证）：
1. **第 1 epoch**：
   - 检查 loss 是否正常（不是 nan/inf）
   - 检查日志确认加载的是 `diting2_train` 而非 `diting_train`
   - 检查 P/S 通道的 loss 是否都在下降
2. **前 10 epochs**：
   - loss 是否稳定下降
   - 验证集指标是否合理
   - GPU 显存占用（单卡 4060 (8GB) 应该够用 batch_size=32）
3. **估算训练时间**：根据前 10 epochs 的速度估算完整 200 epochs 需要多久
4. **确认无误后**：运行完整 200 epochs

### Phase 2: 其他任务

**成功跑通 dpk 后**，按以下顺序训练：

#### 2.1 Magnitude (emg)

```bash
python main.py --seed 0 --mode train --model-name SeisMoLLM_emg --log-base ./logs --log-step 300 --data /path/to/diting2_preprocessed --dataset-name diting2 --data-split true --train-size 0.8 --val-size 0.1 --shuffle true --workers 4 --in-samples 8192 --batch-size 32 --augmentation true --epochs 200 --patience 30 --base-lr 0.0005 --max-lr 0.001 --warmup-steps 2500 --down-steps 3000 --label-width 0.5 --label-shape gaussian --norm-mode std
```

#### 2.2 Back-Azimuth (baz)

```bash
python main.py --seed 0 --mode train --model-name SeisMoLLM_baz --log-base ./logs --log-step 300 --data /path/to/diting2_preprocessed --dataset-name diting2 --data-split true --train-size 0.8 --val-size 0.1 --shuffle true --workers 4 --in-samples 8192 --batch-size 32 --augmentation true --epochs 200 --patience 30 --base-lr 0.0005 --max-lr 0.001 --warmup-steps 2500 --down-steps 3000 --label-width 0.5 --label-shape gaussian --norm-mode std
```

#### 2.3 Distance (dis)

```bash
python main.py --seed 0 --mode train --model-name SeisMoLLM_dis --log-base ./logs --log-step 300 --data /path/to/diting2_preprocessed --dataset-name diting2 --data-split true --train-size 0.8 --val-size 0.1 --shuffle true --workers 4 --in-samples 8192 --batch-size 32 --augmentation true --epochs 200 --patience 30 --base-lr 0.0005 --max-lr 0.001 --warmup-steps 2500 --down-steps 3000 --label-width 0.5 --label-shape gaussian --norm-mode std
```

#### 2.4 Polarity (pmp)

**⚠️ 注意**：pmp 使用 `Pg_polarity ∈ {U,C,R,D}`，映射 `U/C→0, R/D→1`（类别基本平衡 1.07:1，无需 class_weight）

**⚠️ 重要（必须设置）**：pmp 是 onehot 分类标签，数据增强里的“生成纯噪声样本”会清空标签（只保留 data），从而导致训练报错：
`ValueError: Item:pmp, Value:[]`

因此 pmp 任务请显式关闭该增强：
- `--generate-noise-rate 0`

```bash
python main.py --seed 0 --mode train --model-name SeisMoLLM_pmp --log-base ./logs --log-step 300 --data /path/to/diting2_preprocessed --dataset-name DiTing2 --data-split true --train-size 0.8 --val-size 0.1 --shuffle true --workers 4 --in-samples 8192 --batch-size 32 --augmentation true --generate-noise-rate 0 --epochs 200 --patience 30 --base-lr 0.0005 --max-lr 0.001 --warmup-steps 2500 --down-steps 3000 --label-width 0.5 --label-shape gaussian --norm-mode std
```

---

## 📊 结果评估

### 1. 查看训练日志

```bash
# 日志位置
cd logs/<时间戳>_SeisMoLLM_<task>_diting2/

# 查看训练曲线
tensorboard --logdir .
```

### 2. 测试模型

```bash
python main.py \
  --mode test \
  --model-name SeisMoLLM_dpk \
  --checkpoint logs/<时间戳>_SeisMoLLM_dpk_diting2/checkpoints/model-best.pth \
  --data /path/to/diting2_dpk \
  --dataset-name diting2 \
  --batch-size 32
```

### 3. 对比论文结果

| Task | Metric | 论文 DiTing1.0 | 你的 DiTing2.0 | 状态 |
|------|--------|----------------|----------------|------|
| Phase P | F1 / MAE | 95.69% / 0.785 | ? | 待测试 |
| Phase S | F1 / MAE | 72.82% / 1.277 | ? | 待测试 |
| Magnitude | MAE / R² | 0.166 / 0.947 | ? | 待测试 |
| Back-Azimuth | MAE / R² | 33.551° / 0.676 | ? | 待测试 |
| Distance | MAE / R² | 2.972km / 0.986 | ? | 待测试 |
| Polarity | F1 / Pr / Re | 94.33% / 94.42% / 94.25% | ? | 待测试 |

---

## ⚠️ 常见问题

### Q1: GPU 显存不足

**解决方案**：
```bash
# 减小 batch_size
--batch-size 16  # 或 8

# 或使用梯度累积（需要修改 main.py）
# 或减少 GPT-2 层数（修改 models/SeisMoLLM.py: llm_layers=2）
```

### Q2: 训练速度太慢

**建议**：
- **先小规模验证**：运行 10-20 epochs，估算完整训练时间
- **多卡训练**：如果有多卡，修改 `torchrun --nproc_per_node 2`
- **减少数据增强率**：修改 `--add-noise-rate`、`--drop-channel-rate` 等参数
- **减少 epochs**：先跑 50 epochs 看效果，再决定是否完整训练
- **不同任务速度差异大**：dpk（大数据集）慢，其他任务（小数据集）快

### Q3: pmp 任务效果差

**可能原因**：
- 极性标签 U/C vs R/D 实际比例接近 1:1，但如果你的数据集中某一类特别少，可能导致效果差
- 波形裁窗丢失了关键信息

**解决方案**：
- 检查 class 0 vs class 1 的实际数量（在预处理输出中）
- 如果不平衡，添加 class_weight（修改 config.py）
- 尝试不同的裁窗参数（--pre 3000）

### Q4: Pg_azi 结果很差

**可能原因**：
- Pg_azi 是 azimuth，但你用了 `--azi_offset 0`
- 或反过来

**解决方案**：
- 运行 Step 1 的验证脚本
- 重新预处理数据（用正确的 `--azi_offset`）

---

## 📝 实验记录模板

复制以下模板到 `复现2/实验记录_SeisMoLLM.md`：

```markdown
# SeisMoLLM DiTing2.0 实验记录

## 实验 1: Phase Picking (dpk)

**日期**: 2025-12-17
**数据**: diting2_seismollm_dpk (~960k)
**模型**: SeisMoLLM_dpk
**超参**:
- batch_size: 32
- epochs: 200
- base_lr: 5e-4, max_lr: 1e-3
- warmup: 2500, down: 3000

**结果**:
- Phase P F1: ?
- Phase P MAE: ?
- Phase S F1: ?
- Phase S MAE: ?

**训练时间**: ?
**最佳 epoch**: ?

---

## 实验 2: Magnitude (emg)

...
```

---

## 🎯 下一步

1. **立即执行**: Step 2 数据预处理
2. **验证数据**: 检查预处理输出的统计信息（极性比例、样本数等）
3. **启动训练**: Phase 1 - dpk
4. **小规模验证**: 先跑 10 epochs，估算完整训练时间，检查指标是否合理
5. **完整训练**: 确认无误后运行完整 200 epochs
6. **监控训练**: 实时查看 loss 和 tensorboard
7. **记录结果**: 填写实验记录模板

**验证清单**（每个阶段）：
- [ ] 预处理：极性比例 ≈ 1.07:1，样本数符合预期
- [ ] 第 1 epoch：loss 正常，加载正确的 dataset（diting2）
- [ ] 前 10 epochs：loss 下降，P/S 通道都在学习
- [ ] 完整训练：达到收敛或触发 early stopping

祝复现顺利！🚀

