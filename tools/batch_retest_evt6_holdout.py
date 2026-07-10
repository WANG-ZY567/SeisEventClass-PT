#!/usr/bin/env python3
"""
从 EVT6_RESULTS_SUMMARY.md 批量提取已有实验，在 holdout 数据集上重新跑 test，生成对比报告。

用法：
python tools/batch_retest_evt6_holdout.py \
  --summary_md reports/results/EVT6_RESULTS_SUMMARY.md \
  --holdout_data /path/to/diting2_evt6_holdout \
  --out_dir reports/2026-01-13_evt6_holdout_retest \
  --devices cuda:0 cuda:1 \
  --dry_run
"""
import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

def parse_summary_md(md_path: str) -> List[Dict]:
    """从 EVT6_RESULTS_SUMMARY.md 提取实验列表（rank/run/report路径）"""
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    experiments = []
    in_table = False
    for line in lines:
        line = line.strip()
        if line.startswith('| rank | run |'):
            in_table = True
            continue
        if in_table and line.startswith('|---'):
            continue
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6 and parts[1].isdigit():
                rank = int(parts[1])
                run_name = parts[2]
                report_path = parts[5]
                experiments.append({'rank': rank, 'run': run_name, 'report': report_path})
        elif in_table and not line.startswith('|'):
            break
    return experiments

def infer_test_params(report_path: str) -> Dict:
    """从报告路径推断 checkpoint/model_name/TTA 参数"""
    rp = Path(report_path)
    run_dir = rp.parent
    
    # 尝试找 checkpoints/ 或 best.pt
    ckpt = None
    model_name = None
    
    # 常见情况：reports/<timestamp>_<tag>/<run_name>/EVT6_TEST_REPORT.md
    # checkpoint 通常在 <run_name>/checkpoints/
    if (run_dir / 'checkpoints').exists():
        ckpts = sorted((run_dir / 'checkpoints').glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            ckpt = str(ckpts[0])
    elif (run_dir / 'best.pt').exists():
        ckpt = str(run_dir / 'best.pt')
    
    # 从 run_dir 名字推断 model_name
    dir_name = run_dir.name
    if 'SeisMoLLM_evt6' in dir_name or 'SeisMoLLM_evt6' in str(run_dir):
        model_name = 'SeisMoLLM_evt6'
    elif 'SeisMoLLM_evt6_hier_sp' in str(run_dir):
        model_name = 'SeisMoLLM_evt6_hier_sp'
    elif 'ResNet1D_evt6' in dir_name or 'ResNet1D_evt6' in str(run_dir):
        model_name = 'ResNet1D_evt6'
    elif 'InceptionTime_evt6' in dir_name or 'InceptionTime_evt6' in str(run_dir):
        model_name = 'InceptionTime_evt6'
    elif 'CNNsmall' in dir_name:
        model_name = 'CNNsmall_evt6'
    
    # TTA 参数（只能靠启发式/日志推断，这里先给默认值）
    tta = {'times': 1, 'shift': 0, 'noise': 0.0, 'scale': 0.0}
    
    return {
        'checkpoint': ckpt,
        'model_name': model_name,
        'tta': tta,
        'run_dir': str(run_dir)
    }

def generate_test_command(
    exp: Dict,
    params: Dict,
    holdout_data: str,
    out_base: str,
    device: str = 'cuda:0'
) -> Tuple[str, str]:
    """生成单个 test 命令 + 输出子目录名"""
    ckpt = params['checkpoint']
    model_name = params['model_name']
    if not ckpt or not model_name:
        return None, None
    
    # 输出子目录：rank{rank}_{简化run名}
    safe_run = re.sub(r'[^\w\-]+', '_', exp['run'][:60])
    subdir_name = f"rank{exp['rank']:02d}_{safe_run}"
    out_dir = os.path.join(out_base, subdir_name)
    
    # CNNsmall 用独立脚本
    if model_name == 'CNNsmall_evt6':
        cmd = (
            f"python tools/train_evt6_cnn_baseline.py "
            f"--mode test_only --checkpoint {ckpt} "
            f"--data_dir {holdout_data} --out_dir {out_dir} "
            f"--model cnn_small --seed 100"
        )
    else:
        # main.py
        cmd = (
            f"python main.py --mode test --model-name {model_name} "
            f"--device {device} --checkpoint {ckpt} --checkpoint-strict true "
            f"--data {holdout_data} --dataset-name diting2_evt6 "
            f"--shuffle false --workers 0 --in-samples 8192 --batch-size 32 "
            f"--augmentation true --norm-mode max --label-width 0 --label-shape 2 "
            f"--save-test-results true --force-logdir true --log-base {out_dir}"
        )
    
    return cmd, subdir_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary_md', required=True)
    ap.add_argument('--holdout_data', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--devices', nargs='+', default=['cuda:0'])
    ap.add_argument('--dry_run', action='store_true', help='只打印命令不执行')
    args = ap.parse_args()
    
    exps = parse_summary_md(args.summary_md)
    print(f"[OK] 从 {args.summary_md} 解析出 {len(exps)} 条实验")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    tasks = []
    for exp in exps:
        params = infer_test_params(exp['report'])
        if not params['checkpoint']:
            print(f"[SKIP] rank={exp['rank']} {exp['run']}: 无 checkpoint")
            continue
        cmd, subdir = generate_test_command(exp, params, args.holdout_data, args.out_dir, args.devices[0])
        if cmd:
            tasks.append((exp, cmd, subdir))
    
    print(f"[OK] 生成 {len(tasks)} 条可执行任务")
    
    # 写 bash 脚本（两卡并行）
    script_path = os.path.join(args.out_dir, 'run_all_holdout_retest.sh')
    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'# 自动生成：批量 holdout 复测（共 {len(tasks)} 个任务）\n')
        f.write(f'cd {os.path.abspath(".")}\n')
        f.write(f'source /path/to/venv/bin/activate\n\n')
        
        # 按卡分配（简单轮询）
        for i, (exp, cmd, subdir) in enumerate(tasks):
            dev = args.devices[i % len(args.devices)]
            cmd_with_dev = cmd.replace('cuda:0', dev) if 'cuda:0' in cmd else cmd + f' (device={dev})'
            f.write(f'# rank={exp["rank"]} {exp["run"]}\n')
            f.write(f'{cmd} |& tee {args.out_dir}/{subdir}/test.log\n\n')
    
    print(f"[OK] 已生成批量脚本：{script_path}")
    print(f"     运行方式：bash {script_path}")
    
    if not args.dry_run:
        print("[RUN] 开始后台执行...")
        subprocess.Popen(['bash', script_path], cwd=os.getcwd())
        print("[OK] 已启动后台任务")

if __name__ == '__main__':
    main()
