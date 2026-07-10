"""
使用 huggingface_hub 直接下载 GPT-2 模型（支持镜像站）
"""
import os
import sys

print("=" * 60)
print("下载 GPT-2 模型（简单方法）")
print("=" * 60)

# 1. 检查 huggingface_hub
print("\n[1/3] 检查 huggingface_hub 库...")
try:
    import huggingface_hub
    print(f"✅ huggingface_hub 已安装")
except ImportError:
    print("⚠️  huggingface_hub 未安装，正在安装...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface_hub', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])
    import huggingface_hub
    print(f"✅ huggingface_hub 安装成功")

# 2. 设置环境变量使用镜像
print("\n[2/3] 配置下载镜像...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("✅ 已设置镜像: https://hf-mirror.com")

# 3. 下载 GPT-2
print("\n[3/3] 下载 GPT-2 模型文件（约 500MB）...")
print("提示: 这可能需要几分钟，请耐心等待...")

from huggingface_hub import snapshot_download

try:
    model_path = snapshot_download(
        repo_id="gpt2",
        cache_dir="./gpt2_cache",
        resume_download=True,
        local_files_only=False
    )
    print(f"\n✅ GPT-2 下载成功！")
    print(f"   模型路径: {model_path}")
    
    # 测试加载
    print("\n[测试] 验证模型加载...")
    import transformers.models.gpt2 as GPT2
    llm = GPT2.GPT2Model.from_pretrained(
        model_path,
        output_hidden_states=True,
        vocab_size=0,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    print(f"✅ GPT-2 加载成功！参数量: {sum(p.numel() for p in llm.parameters()) / 1e6:.1f}M")
    del llm
    
    # 保存路径
    with open('gpt2_model_path.txt', 'w', encoding='utf-8') as f:
        f.write(model_path)
    
    print("\n" + "=" * 60)
    print("✅ 完成！模型路径已保存到: gpt2_model_path.txt")
    print("=" * 60)
    print("\n下一步：直接运行 test_setup.py 或训练脚本。")
    print("代码会优先读取 gpt2_model_path.txt，也可通过 SEISMOLLM_GPT2_PATH 指定路径。")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n" + "=" * 60)
    print("备用方案：使用不带预训练的 GPT-2")
    print("=" * 60)
    print("\n虽然效果会差一些，但可以先跑通流程。")
    print("修改方法：")
    print("  在训练命令中设置: --pretrain false")
    sys.exit(1)

