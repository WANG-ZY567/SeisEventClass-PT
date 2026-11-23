# Windows 运行指南

## 1. 创建独立环境

在项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

仓库按 PyTorch 2.7.1 整理。若 `requirements.txt` 中的默认安装方式与本机 CUDA 驱动不匹配，请先按 PyTorch 官方说明安装对应的 2.7.1 构建，再安装其余依赖。

## 2. 先做无数据自检

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe test_setup.py --skip-gpt2 --skip-data --smoke-model
```

这一步会验证项目导入、数据集/模型注册、随机初始化的小模型前向传播和反向传播，不需要真实波形或 GPT-2 权重。

如需检查完整的“数据加载—训练—验证—保存检查点”链路，可生成仅用于软件测试的合成数据：

```powershell
.\.venv\Scripts\python.exe tools\create_synthetic_evt6.py .\data\synthetic_evt6
```

合成数据不能用于论文指标或科学结论。

## 3. 配置 GPT-2

```powershell
.\.venv\Scripts\python.exe download_gpt2_simple.py
```

也可以设置环境变量：

```powershell
$env:SEISMOLLM_GPT2_PATH = "D:\models\gpt2"
```

路径必须指向包含 `config.json` 和模型权重文件的本地 GPT-2 目录。

## 4. 检查真实 EVT6 数据

```powershell
.\.venv\Scripts\python.exe test_setup.py --data "D:\data\diting2_evt6"
```

目录中至少需要 `meta_evt6.csv`，或严格划分后的 `meta_evt6_train.csv`、`meta_evt6_val.csv`、`meta_evt6_test.csv`。CSV 必须包含 `key`、`part`、`_npy_path`、`_evt6`。

若已有三份划分文件，检查记录重复、事件泄漏和波形路径：

```powershell
.\.venv\Scripts\python.exe tools\validate_evt6_splits.py "D:\data\diting2_evt6" --check-files
```

## 5. 训练和评估

正式论文配置使用 batch size 16；显存较小的机器可先用更小批量完成软件链路检查，但这不等同于复现论文训练设置：

```powershell
.\scripts\train_paper_evt6.ps1 `
  -DataDir "D:\data\diting2_evt6" `
  -Gpt2Path "D:\models\gpt2"
```

评估：

```powershell
.\scripts\eval_paper_evt6.ps1 `
  -DataDir "D:\data\diting2_evt6" `
  -Checkpoint "D:\runs\model-xx.pth" `
  -Gpt2Path "D:\models\gpt2"
```

显存不足时可进行 CPU 或小批量冒烟测试，但正式论文模型的输入必须保持 `8192` 点、隐藏维 `768`、GPT-2 三层，正式训练设置见 `REPRODUCIBILITY.md`。

## 6. 结果是否算“复现成功”

- `test_setup --smoke-model` 通过：说明代码依赖和核心计算图可运行。
- 真实数据检查通过并完成一个训练 step：说明本地数据接口可运行。
- 完整训练得到论文指标：还要求数据成员、事件级划分、预处理和随机设置与原实验一致。

三者不能混为一谈。论文原始数据、检查点和完整划分未随仓库提供时，只能保证前两层验证。
