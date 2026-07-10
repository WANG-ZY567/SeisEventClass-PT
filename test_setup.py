"""
快速测试脚本：验证 GPT-2 和数据集是否配置正确
"""
import torch
import sys
import os
from pathlib import Path

print("=" * 60)
print("SeisMoLLM 环境测试")
print("=" * 60)

# 1. 测试基础依赖
print("\n[1/5] 检查基础依赖...")
try:
    import transformers
    try:
        import peft  # optional when not using LoRA/freeze
        peft_ok = True
    except ImportError:
        peft_ok = False
    import pandas as pd
    import numpy as np
    print(f"✅ transformers 版本: {transformers.__version__}")
    if peft_ok:
        import peft as _peft
        print(f"✅ peft 版本: {_peft.__version__}")
    else:
        print("⚠️  peft 未安装：若训练时关闭 LoRA/冻结（--freeze false）可先跑通流程；否则需要离线安装 peft")
    print(f"✅ pandas 版本: {pd.__version__}")
    print(f"✅ numpy 版本: {np.__version__}")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

# 2. 测试 PyTorch 和 GPU
print("\n[2/5] 检查 PyTorch 和 GPU...")
print(f"✅ PyTorch 版本: {torch.__version__}")
print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU 数量: {torch.cuda.device_count()}")
else:
    print("⚠️  未检测到 GPU，将使用 CPU 训练（速度较慢）")

# 3. 测试 GPT-2 加载
print("\n[3/5] 测试 GPT-2 模型加载...")

def _resolve_gpt2_path() -> str:
    """
    优先级：
    1) 环境变量 SEISMOLLM_GPT2_PATH
    2) gpt2_model_path.txt
    3) 默认 snapshot 相对路径
    """
    env_path = os.environ.get("SEISMOLLM_GPT2_PATH", "").strip()
    if env_path:
        return env_path

    txt = Path(__file__).parent / "gpt2_model_path.txt"
    if txt.exists():
        p = txt.read_text(encoding="utf-8").strip().replace("\\", "/")
        if p:
            return p

    return "./gpt2_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"


# 检查本地模型是否存在
gpt2_model_path = _resolve_gpt2_path()
if not os.path.exists(gpt2_model_path):
    print(f"❌ 本地 GPT-2 模型不存在: {gpt2_model_path}")
    print("\n请先运行: python download_gpt2_simple.py")
    print("或设置环境变量: export SEISMOLLM_GPT2_PATH=/abs/path/to/gpt2/snapshot")
    sys.exit(1)

try:
    import transformers.models.gpt2 as GPT2
    print(f"正在加载本地 GPT-2 模型...")
    llm = GPT2.GPT2Model.from_pretrained(
        gpt2_model_path,
        output_hidden_states=True,
        vocab_size=0,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    print(f"✅ GPT-2 加载成功！参数量: {sum(p.numel() for p in llm.parameters()) / 1e6:.1f}M")
    del llm
except Exception as e:
    print(f"❌ GPT-2 加载失败: {e}")
    print("\n建议：重新运行下载脚本:")
    print("  python download_gpt2_simple.py")
    sys.exit(1)

# 4. 测试数据集注册
print("\n[4/5] 检查数据集注册...")
try:
    from datasets import build_dataset, get_dataset_list
    dataset_list = get_dataset_list()
    print(f"✅ 可用数据集: {dataset_list}")
    
    # 检查 diting2 数据集（不区分大小写）
    diting2_registered = any('diting2' in ds.lower() for ds in dataset_list)
    if diting2_registered:
        print("✅ DiTing2 数据集已注册")
    else:
        print("❌ DiTing2 数据集未注册")
        sys.exit(1)
except Exception as e:
    print(f"❌ 数据集检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试数据加载
print("\n[5/5] 测试数据加载...")
default_data_path = "/path/to/diting2_preprocessed"
data_path = os.environ.get("SEISMOLLM_DITING2_DATA", default_data_path)
meta_csv = os.path.join(data_path, "meta_full5.csv")

if not os.path.exists(meta_csv):
    print(f"❌ 元数据文件不存在: {meta_csv}")
    print("\n请先运行数据预处理（示例）：")
    print("  python tools/prepare_diting2.py \\")
    print("    --h5 /path/to/CENC_DiTingv2_natural_earthquake.hdf5 \\")
    print("    --json /path/to/CENC_DiTingv2_natural_earthquake.json \\")
    print("    --out_dir /path/to/diting2_preprocessed \\")
    print("    --meta_csv meta_full5.csv --in_samples 8192 --pre 2000 --azi_offset 0 --require_full5")
    print("\n或设置环境变量指定数据目录：")
    print(f"  export SEISMOLLM_DITING2_DATA=/abs/path/to/diting2_seismollm_full5")
    sys.exit(1)

try:
    df = pd.read_csv(meta_csv)
    print(f"✅ 元数据加载成功: {len(df)} 条记录")
    print(f"✅ 列名: {list(df.columns)}")
    
    # 检查数据类型
    print("\n列类型检查:")
    for col in ['key', '_pmp_bin', 'baz', 'dis', 'evmag']:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
    
    # 检查 npy 文件
    waves_dir = os.path.join(data_path, "waves")
    # 使用 _npy_path 列（相对路径）或 key 列
    if '_npy_path' in df.columns:
        first_npy = df.iloc[0]['_npy_path']
        npy_path = os.path.join(data_path, first_npy)
    else:
        first_key = df.iloc[0]['key']
        first_part = df.iloc[0]['part']
        npy_path = os.path.join(waves_dir, f"{first_key}_{first_part}.npy")
    
    if os.path.exists(npy_path):
        waveform = np.load(npy_path)
        print(f"\n✅ 波形文件加载成功")
        print(f"  形状: {waveform.shape}")
        print(f"  数据类型: {waveform.dtype}")
    else:
        print(f"❌ 波形文件不存在: {npy_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试完成
print("\n" + "=" * 60)
print("✅ 所有测试通过！环境配置正确，可以开始训练。")
print("=" * 60)
print("\n下一步：运行训练命令（见 DiTing2.0使用指南.md）")

