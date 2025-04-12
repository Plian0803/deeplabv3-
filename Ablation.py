#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验脚本：评估Transformer和自注意力机制对模型性能的影响
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from subprocess import Popen, PIPE
from pathlib import Path
import threading
import queue
import re

# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Transformer和自注意力机制消融实验')
    
    # 实验设置
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                        help='实验结果输出目录')
    parser.add_argument('--data_dir', type=str, default='datasets/BCSS-WSSS/',
                        help='数据集目录')
    parser.add_argument('--dataset', type=str, default='bcss',
                        choices=['bcss', 'luad'],
                        help='使用的数据集，可选bcss或luad')
    parser.add_argument('--epochs', type=int, default=5,
                        help='每个模型的训练轮数(减少默认值以加快实验)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='训练的批量大小')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='使用的GPU ID，如"0,1"')
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    
    # 模型设置
    parser.add_argument('--include_baseline', action='store_true', default=True,
                        help='包含基线模型（原始DeepLab）')
    parser.add_argument('--include_sa_only', action='store_true', default=True,
                        help='包含只使用自注意力的模型')
    parser.add_argument('--include_transformer_only', action='store_true', default=True,
                        help='包含只使用Transformer的模型')
    parser.add_argument('--include_full_model', action='store_true', default=True,
                        help='包含完整增强模型（自注意力+Transformer）')
    
    # Transformer参数设置
    parser.add_argument('--transformer_layers', type=int, default=2,
                        help='Transformer层数')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP扩展比例')
    
    # 其他DeepLab参数
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='骨干网络类型')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='输出步长')
    parser.add_argument('--n_class', type=int, default=4,
                        help='类别数量(BCSS和LUAD数据集都是4类)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=0.01,
                        help='早停最小增量')
    
     # 添加PSM相关参数
    parser.add_argument('--include_psm', action='store_true', default=True,
                        help='包含使用PSM方法的模型')
    parser.add_argument('--psm_beta', type=float, default=2.5,
                        help='PSM中原始图像信息的权重')
    parser.add_argument('--psm_k_clusters', type=int, default=3,
                        help='PSM中K-Means聚类的簇数量')
    return parser.parse_args()

def create_experiment_configs(args):
    """根据命令行参数创建实验配置"""
    configs = []
    # 基线模型和自注意力模型使用较大batch_size
    large_batch_size = 64
    # Transformer和完整增强模型使用较小batch_size
    small_batch_size = args.batch_size  # 默认使用命令行参数中的值，如16
    
    # 共享的基本参数
    base_args = [
        f"--dataset={args.dataset}",
        f"--dataroot={args.data_dir}",
        f"--epochs={args.epochs}",
        f"--backbone={args.backbone}",
        f"--out-stride={args.out_stride}",
        f"--n_class={args.n_class}",
        f"--Is_GM=True",  # 使用门控机制
        f"--early_stop_patience={args.early_stop_patience}",
        f"--min_delta={args.min_delta}",
        f"--savepath={args.savepath}",
    ]
    
    # 基线模型 - 原始DeepLab V3+
    if args.include_baseline:
        configs.append({
            'name': '基线模型',
            'desc': '原始DeepLab V3+',
            'batch_size': large_batch_size,
            'script_args': base_args + [
                '--use_self_attention=False',
                '--use_transformer=False'
            ]
        })
    
    # 只使用自注意力的模型
    if args.include_sa_only:
        configs.append({
            'name': '自注意力模型',
            'desc': '带自注意力的ASPP',
            'batch_size': large_batch_size,
            'script_args': base_args + [
                '--use_self_attention=True',
                '--use_transformer=False'
            ]
        })
    
    # 只使用Transformer的模型
    if args.include_transformer_only:
        configs.append({
            'name': 'Transformer模型',
            'desc': '带Transformer解码器',
            'batch_size': small_batch_size,
            'script_args': base_args + [
                '--use_self_attention=False',
                '--use_transformer=True',
                f'--num_transformer_layers={args.transformer_layers}',
                f'--num_attention_heads={args.attention_heads}',
                f'--dropout_rate={args.dropout_rate}',
                f'--mlp_ratio={args.mlp_ratio}'
            ]
        })
    
    # 完整增强模型 - 同时使用Transformer和自注意力
    if args.include_full_model:
        configs.append({
            'name': '完整增强模型',
            'desc': '带自注意力ASPP和Transformer解码器',
            'batch_size': small_batch_size,
            'script_args': base_args + [
                '--use_self_attention=True',
                '--use_transformer=True',
                f'--num_transformer_layers={args.transformer_layers}',
                f'--num_attention_heads={args.attention_heads}',
                f'--dropout_rate={args.dropout_rate}',
                f'--mlp_ratio={args.mlp_ratio}'
            ]
        })
        
    # PSM相关配置
    if args.include_psm:
        # 基础PSM模型 - 不使用Transformer和自注意力
        configs.append({
            'name': 'PSM基础模型',
            'desc': '使用PSM方法生成伪标签',
            'batch_size': large_batch_size,
            'script_args': base_args + [
                '--use_self_attention=False',
                '--use_transformer=False'
            ],
            'psm_args': [
                '--use_psm=True',
                f'--psm_beta={args.psm_beta}',
                f'--psm_k_clusters={args.psm_k_clusters}'
            ],
            'run_psm_generation': True  # 运行PSM生成
        })
        # PSM+自注意力模型
        if args.include_sa_only:
            configs.append({
                'name': 'PSM+自注意力模型',
                'desc': '使用PSM方法生成伪标签，并使用自注意力ASPP',
                'batch_size': large_batch_size,
                'script_args': base_args + [
                    '--use_self_attention=True',
                    '--use_transformer=False'
                ],
                'psm_args': [
                    '--use_psm=True',
                    f'--psm_beta={args.psm_beta}',
                    f'--psm_k_clusters={args.psm_k_clusters}'
                ],
                'run_psm_generation': True  # 运行PSM生成
            })
        # PSM+Transformer模型
        if args.include_transformer_only:
            configs.append({
                'name': 'PSM+Transformer模型',
                'desc': '使用PSM方法生成伪标签，并使用Transformer解码器',
                'batch_size': small_batch_size,
                'script_args': base_args + [
                    '--use_self_attention=False',
                    '--use_transformer=True',
                    f'--num_transformer_layers={args.transformer_layers}',
                    f'--num_attention_heads={args.attention_heads}',
                    f'--dropout_rate={args.dropout_rate}',
                    f'--mlp_ratio={args.mlp_ratio}'
                ],
                'psm_args': [
                    '--use_psm=True',
                    f'--psm_beta={args.psm_beta}',
                    f'--psm_k_clusters={args.psm_k_clusters}'
                ],
                'run_psm_generation': True  # 运行PSM生成
            })
        # PSM+完整增强模型
        if args.include_full_model:
            configs.append({
                'name': 'PSM+完整增强模型',
                'desc': '使用PSM方法生成伪标签，并同时使用自注意力ASPP和Transformer解码器',
                'batch_size': small_batch_size,
                'script_args': base_args + [
                    '--use_self_attention=True',
                    '--use_transformer=True',
                    f'--num_transformer_layers={args.transformer_layers}',
                    f'--num_attention_heads={args.attention_heads}',
                    f'--dropout_rate={args.dropout_rate}',
                    f'--mlp_ratio={args.mlp_ratio}'
                ],
                'psm_args': [
                    '--use_psm=True',
                    f'--psm_beta={args.psm_beta}',
                    f'--psm_k_clusters={args.psm_k_clusters}'
                ],
                'run_psm_generation': True  # 运行PSM生成
            })
    return configs

def setup_experiment_directory(output_dir):
    """设置实验目录结构"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f"ablation_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    return {
        'timestamp': timestamp,
        'root': exp_dir,
        'logs': log_dir,
        'plots': plot_dir,
        'checkpoints': checkpoint_dir
    }

def run_training_process(cmd, model_name, log_file, gpu_id=None, psm_args=None, run_psm=False):
    """运行训练命令并记录输出"""
    print(f"\n{'-'*40}")
    
    # 如果需要运行PSM生成
    if run_psm and psm_args:
        # 为当前实验创建唯一的PSM输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_psm_dir = f"PSM_{model_name.replace(' ', '_').lower()}_{timestamp}"
        full_psm_dir = os.path.join(args.data_dir, 'train_PM', unique_psm_dir)
        
        # 添加输出目录参数
        psm_args_with_dir = psm_args + [f"--unique_output_dir={full_psm_dir}"]
        
        psm_cmd = ["python", "2_generate_PM.py"] + psm_args_with_dir
        print(f"生成PSM伪标签: {' '.join(psm_cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 运行PSM生成脚本
        psm_process = Popen(
            psm_cmd,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
            env=env
        )
        psm_process.wait()
        
        # 将PSM输出目录添加到训练命令中
        cmd.append(f"--pm_dir={full_psm_dir}")
        
    print(f"执行: {' '.join(cmd)}")
    print(f"日志: {log_file}")
    print(f"{'-'*40}\n")
    # 设置环境变量
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"使用GPU: {gpu_id}")
    
    # 打开日志文件
    with open(log_file, 'w') as f_log:
        # 启动进程
        process = Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
            env=env
        )
        
        # 创建队列用于存储输出
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        # 定义读取输出的函数
        def read_output(pipe, queue, prefix, log_file):
            for line in iter(pipe.readline, ''):
                queue.put(line)
                print(f"{prefix}: {line.strip()}")
                log_file.write(line)
                log_file.flush()
                
        # 创建线程来读取输出
        stdout_thread = threading.Thread(
            target=read_output, 
            args=(process.stdout, stdout_queue, f"{model_name}", f_log)
        )
        stderr_thread = threading.Thread(
            target=read_output, 
            args=(process.stderr, stderr_queue, f"{model_name} (错误)", f_log)
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待进程完成
        return_code = process.wait()
        
        # 等待线程结束
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        # 收集所有输出
        stdout_content = []
        while not stdout_queue.empty():
            stdout_content.append(stdout_queue.get())
        
        stderr_content = []
        while not stderr_queue.empty():
            stderr_content.append(stderr_queue.get())
        
        return {
            'return_code': return_code,
            'stdout': ''.join(stdout_content),
            'stderr': ''.join(stderr_content)
        }

def parse_training_results(output, model_id, checkpoint_dir):
    """从训练输出中解析性能指标"""
    metrics = {
        'miou': None,
        'fwiou': None,
        'acc': None,
        'acc_class': None,
        'ious': None,
        'checkpoint_path': os.path.join(checkpoint_dir, f"stage2_checkpoint_{model_id}_trained_on_{args.dataset}.pth")
    }
    
    # 解析最终测试结果
    lines = output['stdout'].split('\n')
    test_section_started = False
    
    for i, line in enumerate(lines):
        if "Test:" in line:
            test_section_started = True
            continue
        
        if test_section_started:
            # 尝试解析主要指标行
            if "Acc:" in line and "mIoU:" in line:
                # 使用正则表达式提取数值
                acc_match = re.search(r'Acc:\s*([\d.]+)', line)
                acc_class_match = re.search(r'Acc_class:\s*([\d.]+)', line)
                miou_match = re.search(r'mIoU:\s*([\d.]+)', line)
                fwiou_match = re.search(r'fwIoU:\s*([\d.]+)', line)
                
                if acc_match: metrics['acc'] = float(acc_match.group(1))
                if acc_class_match: metrics['acc_class'] = float(acc_class_match.group(1))
                if miou_match: metrics['miou'] = float(miou_match.group(1))
                if fwiou_match: metrics['fwiou'] = float(fwiou_match.group(1))
            
            # 尝试解析IoUs行
            if "IoUs:" in line:
                try:
                    # 提取IoUs后面的数组内容
                    ious_str = line.split("IoUs:")[1].strip()
                    # 安全解析数组
                    metrics['ious'] = eval(ious_str)
                except Exception as e:
                    print(f"解析IoUs时出错: {str(e)}")
    
    # 如果没有找到有效结果，尝试查找验证集上的结果
    if metrics['miou'] is None:
        val_section_started = False
        for i, line in enumerate(lines):
            if "Validation:" in line:
                val_section_started = True
                continue
            
            if val_section_started:
                if "Acc:" in line and "mIoU:" in line:
                    # 与上面相同的正则表达式
                    acc_match = re.search(r'Acc:\s*([\d.]+)', line)
                    acc_class_match = re.search(r'Acc_class:\s*([\d.]+)', line)
                    miou_match = re.search(r'mIoU:\s*([\d.]+)', line)
                    fwiou_match = re.search(r'fwIoU:\s*([\d.]+)', line)
                    
                    if acc_match: metrics['acc'] = float(acc_match.group(1))
                    if acc_class_match: metrics['acc_class'] = float(acc_class_match.group(1))
                    if miou_match: metrics['miou'] = float(miou_match.group(1))
                    if fwiou_match: metrics['fwiou'] = float(fwiou_match.group(1))
                
                if "IoUs:" in line:
                    try:
                        ious_str = line.split("IoUs:")[1].strip()
                        metrics['ious'] = eval(ious_str)
                    except Exception as e:
                        print(f"解析验证集IoUs时出错: {str(e)}")
    
    return metrics

def create_visualizations(results, exp_dirs):
    """创建实验结果的可视化图表"""
    # 配置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    except:
        print("警告: 无法设置中文字体，图表中文可能显示为方块")
        
    if not results:
        print("没有结果可视化")
        return
    
    valid_results = [r for r in results if r['metrics']['miou'] is not None]
    if not valid_results:
        print("没有有效的结果可视化")
        return
    
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 提取数据
    models = [r['name'] for r in valid_results]
    mious = [r['metrics']['miou'] for r in valid_results]
    fwious = [r['metrics']['fwiou'] for r in valid_results]
    
    x = range(len(models))
    width = 0.35
    
    # 绘制条形图
    plt.bar([i - width/2 for i in x], mious, width, label='mIoU', color='#3498db')
    plt.bar([i + width/2 for i in x], fwious, width, label='fwIoU', color='#e74c3c')
    
    # 添加数据标签
    for i, v in enumerate(mious):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(fwious):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('模型配置')
    plt.ylabel('性能指标')
    plt.title(f'{args.dataset}数据集上的不同模型配置性能对比')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, max(max(mious), max(fwious)) * 1.15)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(exp_dirs['plots'] / "performance_comparison.png", dpi=300)
    plt.savefig(exp_dirs['plots'] / "performance_comparison.pdf")
    plt.close()
    
    # 如果有多个结果，创建雷达图比较
    if len(valid_results) > 1:
        # 创建雷达图比较不同指标
        categories = ['mIoU', 'fwIoU', 'Acc', 'Acc_class']
        values = []
        for r in valid_results:
            values.append([
                r['metrics']['miou'],
                r['metrics']['fwiou'],
                r['metrics']['acc'],
                r['metrics']['acc_class']
            ])
        
        # 创建雷达图
        plt.figure(figsize=(10, 8))
        
        # 设置角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 绘制雷达图
        ax = plt.subplot(111, polar=True)
        
        # 为每个模型绘制雷达图
        for i, vals in enumerate(values):
            vals = vals + vals[:1]  # 闭合数据
            ax.plot(angles, vals, linewidth=2, label=valid_results[i]['name'])
            ax.fill(angles, vals, alpha=0.1)
        
        # 设置标签和图例
        plt.xticks(angles[:-1], categories)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f'{args.dataset}数据集上各模型在不同指标上的性能对比')
        plt.tight_layout()
        
        # 保存雷达图
        plt.savefig(exp_dirs['plots'] / "radar_comparison.png", dpi=300)
        plt.close()
        
        # 创建类别IoU比较图(如果有IoU数据)
        if all(r['metrics']['ious'] is not None for r in valid_results):
            plt.figure(figsize=(12, 6))
            
            # 获取每个类别的IoU
            class_names = [f'类别{i}' for i in range(len(valid_results[0]['metrics']['ious']))]
            
            # 为每个模型绘制类别IoU条形图
            for i, r in enumerate(valid_results):
                plt.bar(
                    [x + i*width/len(valid_results) for x in range(len(class_names))], 
                    r['metrics']['ious'], 
                    width/len(valid_results), 
                    label=r['name']
                )
            
            plt.xlabel('类别')
            plt.ylabel('IoU')
            plt.title(f'{args.dataset}数据集上不同模型的类别IoU对比')
            plt.xticks(range(len(class_names)), class_names)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(exp_dirs['plots'] / "class_iou_comparison.png", dpi=300)
            plt.close()

def generate_report(results, exp_dirs, args):
    """生成实验报告"""
    if not results:
        print("没有结果可用，无法生成报告")
        return
    
    valid_results = [r for r in results if r['metrics']['miou'] is not None]
    if not valid_results:
        print("没有有效的结果，无法生成报告")
        return
    
    # 找出性能最好的模型
    best_model = max(valid_results, key=lambda x: x['metrics']['miou'])
    
    # 生成报告
    report_path = exp_dirs['root'] / "experiment_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Transformer和自注意力机制消融实验报告\n\n")
        f.write(f"实验日期: {exp_dirs['timestamp']}\n\n")
        
        f.write("## 实验概述\n\n")
        f.write("这项消融实验旨在评估Transformer和自注意力机制对DeepLab V3+模型性能的影响。\n\n")
        
        f.write("### 实验配置\n\n")
        f.write(f"- 数据集: {args.dataset}\n")
        f.write(f"- 训练轮数: {args.epochs}\n")
        f.write(f"- 批量大小: {args.batch_size}\n")
        f.write(f"- 骨干网络: {args.backbone}\n")
        f.write(f"- 输出步长: {args.out_stride}\n")
        f.write(f"- 类别数量: {args.n_class}\n\n")
        
        if args.include_transformer_only:
            f.write("### Transformer配置\n\n")
            f.write(f"- Transformer层数: {args.transformer_layers}\n")
            f.write(f"- 注意力头数: {args.attention_heads}\n")
            f.write(f"- Dropout率: {args.dropout_rate}\n")
            f.write(f"- MLP比例: {args.mlp_ratio}\n\n")
        
        f.write("## 实验结果\n\n")
        
        # 添加表格
        f.write("### 性能指标比较\n\n")
        f.write("| 模型 | 描述 | mIoU | fwIoU | Acc | Acc_class |\n")
        f.write("|------|------|------|-------|-----|----------|\n")
        
        for r in valid_results:
            f.write(f"| {r['name']} | {r['desc']} | {r['metrics']['miou']:.4f} | {r['metrics']['fwiou']:.4f} | "
                    f"{r['metrics']['acc']:.4f} | {r['metrics']['acc_class']:.4f} |\n")
        
        f.write("\n### 最佳模型\n\n")
        f.write(f"- 模型: **{best_model['name']}**\n")
        f.write(f"- 描述: {best_model['desc']}\n")
        f.write(f"- mIoU: {best_model['metrics']['miou']:.4f}\n")
        f.write(f"- fwIoU: {best_model['metrics']['fwiou']:.4f}\n")
        
        f.write("\n## 结论\n\n")
        
        # 基于结果得出简单结论
        baseline = next((r for r in valid_results if r['name'] == '基线模型'), None)
        sa_model = next((r for r in valid_results if r['name'] == '自注意力模型'), None)
        transformer_model = next((r for r in valid_results if r['name'] == 'Transformer模型'), None)
        full_model = next((r for r in valid_results if r['name'] == '完整增强模型'), None)
        
        if baseline and full_model:
            improvement = (full_model['metrics']['miou'] - baseline['metrics']['miou']) / baseline['metrics']['miou'] * 100
            f.write(f"1. 与基线模型相比，完整增强模型（同时使用自注意力和Transformer）提高了性能 **{improvement:.2f}%**。\n\n")
        
        if baseline and sa_model:
            sa_improvement = (sa_model['metrics']['miou'] - baseline['metrics']['miou']) / baseline['metrics']['miou'] * 100
            f.write(f"2. 仅添加自注意力机制提高了性能 **{sa_improvement:.2f}%**。\n\n")
        
        if baseline and transformer_model:
            trans_improvement = (transformer_model['metrics']['miou'] - baseline['metrics']['miou']) / baseline['metrics']['miou'] * 100
            f.write(f"3. 仅添加Transformer解码器提高了性能 **{trans_improvement:.2f}%**。\n\n")
        
        if sa_model and transformer_model and full_model:
            f.write("4. 结合自注意力和Transformer的效果" + 
                  ("优于" if full_model['metrics']['miou'] > max(sa_model['metrics']['miou'], transformer_model['metrics']['miou']) else "不如") + 
                  "单独使用其中一种机制。\n\n")
        
        # 添加图表
        f.write("\n## 可视化结果\n\n")
        
        f.write("### 性能对比\n\n")
        f.write(f"![性能对比](./plots/performance_comparison.png)\n\n")
        
        if len(valid_results) > 1:
            f.write("### 雷达图对比\n\n")
            f.write(f"![雷达图对比](./plots/radar_comparison.png)\n\n")
            
            if all(r['metrics']['ious'] is not None for r in valid_results):
                f.write("### 类别IoU对比\n\n")
                f.write(f"![类别IoU对比](./plots/class_iou_comparison.png)\n\n")
        
        # 添加推荐和建议
        f.write("\n## 推荐和建议\n\n")
        
        # 根据实验结果给出建议
        best_model_name = best_model['name']
        f.write(f"基于实验结果，我们推荐在{args.dataset}数据集上使用**{best_model_name}**配置，它在mIoU和其他指标上表现最佳。\n\n")
        
        if full_model and full_model['name'] == best_model_name:
            f.write("结合自注意力和Transformer机制能够显著提升模型性能，建议在计算资源允许的情况下采用完整增强模型。\n")
        elif sa_model and sa_model['name'] == best_model_name:
            f.write("仅使用自注意力机制就能获得良好的性能提升，计算开销相对较小，适合资源受限的场景。\n")
        elif transformer_model and transformer_model['name'] == best_model_name:
            f.write("Transformer解码器提供了较大的性能提升，建议在需要高精度分割的场景中使用。\n")
    
    print(f"\n实验报告已生成: {report_path}")
    return report_path

def main():
    # 全局定义args以便在各函数中访问
    global args
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置实验目录
    exp_dirs = setup_experiment_directory(args.output_dir)
    
    # 记录实验配置
    with open(exp_dirs['root'] / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建实验配置
    configs = create_experiment_configs(args)
    with open(exp_dirs['root'] / "configs.json", 'w') as f:
        json.dump(configs, f, indent=2)
    
    # 输出实验信息
    print(f"{'='*80}")
    print(f"开始消融实验: {exp_dirs['timestamp']}")
    print(f"实验目录: {exp_dirs['root']}")
    print(f"数据集: {args.dataset}")
    print(f"GPU IDs: {args.gpu_ids}")
    print(f"{'='*80}")
    print(f"将运行 {len(configs)} 个实验配置:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config['name']} - {config['desc']}")
    print(f"{'='*80}")
    
    # 确保每个实验使用一个GPU
    gpus = args.gpu_ids.split(',')
    
    # 运行实验
    results = []
    for i, config in enumerate(configs):
        # 为当前配置选择GPU
        gpu_id = gpus[i % len(gpus)] if gpus else None
        
        model_name = config['name']
        log_file = exp_dirs['logs'] / f"{model_name.replace(' ', '_')}.log"
        
        print(f"\n{'='*40}")
        print(f"开始实验 {i+1}/{len(configs)}: {model_name}")
        print(f"{'='*40}")
        
        # 构建训练命令
        
        # 为每个模型创建唯一标识符
        model_id = f"{model_name.replace(' ', '_').lower()}_{exp_dirs['timestamp']}"
        
        # 为每个实验创建专用的checkpoint目录
        specific_checkpoint_dir = exp_dirs['checkpoints'] / model_id
        specific_checkpoint_dir.mkdir(exist_ok=True)
        
        # 添加PSM参数（如果有）
        psm_args = None
        if 'psm_args' in config:
            psm_args = config['psm_args'] + [
                f"--dataroot={args.data_dir}",
                f"--dataset={args.dataset}",
                "--network=network.resnet38_cls",
                f"--weights=checkpoints/stage1_checkpoint_trained_on_{args.dataset}.pth"
            ]
        
        # 构建命令，添加模型ID参数
        cmd = ["python", "3_train_stage2.py"] + config['script_args'] + [
            f"--savepath={str(specific_checkpoint_dir)}",
            f"--model_id={model_id}",
            f"--batch_size={config['batch_size']}" 
        ]

        
        # 执行训练
        start_time = time.time()
        # 根据配置决定是否运行PSM生成
        run_psm = 'run_psm_generation' in config and config['run_psm_generation']
        output = run_training_process(cmd, model_name, log_file, gpu_id, psm_args, run_psm)
        end_time = time.time()
        
        # 解析结果
        metrics = parse_training_results(output, model_id, str(specific_checkpoint_dir))
        
        # 记录结果
        result = {
            'name': model_name,
            'desc': config['desc'],
            'cmd': cmd,
            'config': config,
            'metrics': metrics,
            'checkpoint_path': metrics['checkpoint_path'],
            'training_time': end_time - start_time,
            'return_code': output['return_code'],
            'batch_size': config['batch_size']  # 明确记录batch_size
        }
        
        results.append(result)
        
        # 保存当前结果
        with open(exp_dirs['root'] / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)  # 使用default=str处理不可序列化的对象
        
        # 输出结果摘要
        print(f"\n{'='*40}")
        print(f"实验 {i+1} ({model_name}) 完成")
        if metrics['miou'] is not None:
            print(f"mIoU: {metrics['miou']:.4f}, fwIoU: {metrics['fwiou']:.4f}")
            print(f"训练时间: {(end_time - start_time)/3600:.2f}小时")
        else:
            print("实验未成功完成，请检查日志")
        print(f"{'='*40}\n")
    
    # 创建可视化
    create_visualizations(results, exp_dirs)
    
    # 生成报告
    generate_report(results, exp_dirs, args)
    
    print(f"\n{'='*80}")
    print(f"消融实验完成！")
    print(f"结果保存在: {exp_dirs['root']}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()