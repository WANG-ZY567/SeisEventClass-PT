"""
使用 ModelScope（阿里云）下载 GPT-2 模型
"""
import os

print("=" * 60)
print("使用 ModelScope 下载 GPT-2 模型")
print("=" * 60)

# 1. 安装 modelscope（如果未安装）
print("\n[1/3] 检查 modelscope 库...")
try:
    import modelscope
    print(f"✅ modelscope 已安装，版本: {modelscope.__version__}")
except ImportError:
    print("⚠️  modelscope 未安装，正在安装...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'modelscope', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])
    import modelscope
    print(f"✅ modelscope 安装成功，版本: {modelscope.__version__}")

# 2. 下载 GPT-2 模型
print("\n[2/3] 下载 GPT-2 模型（约 500MB）...")
from modelscope import snapshot_download

try:
    model_dir = snapshot_download(
        'AI-ModelScope/gpt2',
        cache_dir='./pretrained_models',
        revision='master'
    )
    print(f"✅ GPT-2 下载成功！")
    print(f"   模型路径: {model_dir}")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)

# 3. 测试加载
print("\n[3/3] 测试模型加载...")
try:
    import transformers.models.gpt2 as GPT2
    llm = GPT2.GPT2Model.from_pretrained(
        model_dir,
        output_hidden_states=True,
        vocab_size=0,
        ignore_mismatched_sizes=True,
        local_files_only=True  # 只使用本地文件
    )
    print(f"✅ GPT-2 加载成功！参数量: {sum(p.numel() for p in llm.parameters()) / 1e6:.1f}M")
    del llm
except Exception as e:
    print(f"❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 保存配置信息
print("\n" + "=" * 60)
print("✅ GPT-2 下载和测试完成！")
print("=" * 60)
print(f"\n模型已保存到: {model_dir}")
print("\n下一步：修改 models/SeisMoLLM.py 使用本地模型路径")
print(f"  将 'gpt2' 改为: r'{model_dir}'")

# 保存路径信息
with open('gpt2_model_path.txt', 'w', encoding='utf-8') as f:
    f.write(model_dir)
print(f"\n模型路径已保存到: gpt2_model_path.txt")

