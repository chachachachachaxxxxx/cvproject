#!/usr/bin/env python3
"""
Vision Transformer MNIST 高效消融实验启动脚本
使用实验设计方法减少实验数量从243个到31个
"""

import os
import sys
import argparse
from efficient_ablation_runner import EfficientAblationRunner


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer MNIST 高效消融实验')
    parser.add_argument('--config', type=str, default='efficient_ablation_config.json',
                       help='实验配置文件路径')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式（只运行基线和部分单因子实验）')
    parser.add_argument('--dry-run', action='store_true',
                       help='空运行模式（只显示实验配置，不实际训练）')
    
    args = parser.parse_args()
    
    print("Vision Transformer MNIST 高效消融实验")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"快速模式: {'是' if args.quick else '否'}")
    print(f"空运行模式: {'是' if args.dry_run else '否'}")
    print()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在！")
        print("请确保配置文件存在，或使用默认配置文件 'efficient_ablation_config.json'")
        return 1
    
    # 如果是快速模式，修改配置
    if args.quick:
        print("快速测试模式：将只运行部分关键实验")
        create_quick_config(args.config)
        args.config = 'quick_ablation_config.json'
    
    try:
        # 创建实验运行器
        runner = EfficientAblationRunner(args.config)
        
        if args.dry_run:
            # 空运行模式：只显示配置信息
            print_experiment_plan(runner)
        else:
            # 正常运行模式
            runner.run_efficient_ablation()
            print(f"\n实验完成！结果保存在: {runner.experiment_dir}")
            
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        return 1
    except Exception as e:
        print(f"\n实验运行出错: {str(e)}")
        return 1
    
    return 0


def create_quick_config(original_config):
    """创建快速测试配置"""
    import json
    
    # 读取原始配置
    with open(original_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 修改为快速配置
    config['experiment_name'] = "快速消融实验测试"
    config['description'] = "快速测试版本，只运行关键实验"
    
    # 只保留基线和部分单因子实验
    quick_groups = {
        'baseline': config['experiment_groups']['baseline'],
        'patch_size_ablation': {
            'description': "Patch Size单因子消融 - 快速版",
            'base_config': {
                'num_layers': 6,
                'num_heads': 4,
                'embed_dim': 64,
                'batch_size': 64,
                'dropout': 0.1
            },
            'variable': 'patch_size',
            'values': [4, 7, 14]  # 保持3个值
        },
        'depth_ablation': {
            'description': "网络深度单因子消融 - 快速版",
            'base_config': {
                'patch_size': 7,
                'num_heads': 4,
                'embed_dim': 64,
                'batch_size': 64,
                'dropout': 0.1
            },
            'variable': 'num_layers',
            'values': [3, 6]  # 只保留2个值
        },
        'extreme_configs': {
            'description': "极端配置实验 - 快速版",
            'experiments': [
                config['experiment_groups']['extreme_configs']['experiments'][0],  # minimal
                config['experiment_groups']['extreme_configs']['experiments'][2]   # balanced
            ]
        }
    }
    
    config['experiment_groups'] = quick_groups
    
    # 减少训练轮数
    config['train_config']['epochs'] = 15
    config['train_config']['early_stopping_patience'] = 3
    
    # 更新实验数量估计
    config['estimated_experiments'] = {
        'baseline': 1,
        'patch_size_ablation': 3,
        'depth_ablation': 2,
        'extreme_configs': 2,
        'total': 8,
        'original_full_factorial': 243,
        'reduction_ratio': 0.033
    }
    
    # 保存快速配置
    with open('quick_ablation_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"快速配置已创建: quick_ablation_config.json")


def print_experiment_plan(runner):
    """打印实验计划"""
    print("\n实验计划概览:")
    print("-" * 40)
    
    config = runner.config
    
    print(f"实验名称: {config['experiment_name']}")
    print(f"实验描述: {config['description']}")
    print(f"预计实验总数: {config['estimated_experiments']['total']}")
    print(f"原始全因子实验数: {config['estimated_experiments']['original_full_factorial']}")
    print(f"压缩比例: {config['estimated_experiments']['reduction_ratio']:.1%}")
    print()
    
    print("实验组详情:")
    for group_name, group_config in config['experiment_groups'].items():
        print(f"\n{group_name}:")
        print(f"  描述: {group_config['description']}")
        
        if 'experiments' in group_config:
            print(f"  实验数: {len(group_config['experiments'])}")
            for i, exp in enumerate(group_config['experiments']):
                name = exp.get('name', f"{group_name}_{i+1}")
                print(f"    - {name}")
        elif 'variable' in group_config:
            variable = group_config['variable']
            values = group_config['values']
            print(f"  变量: {variable}")
            print(f"  取值: {values}")
            print(f"  实验数: {len(values)}")
    
    print(f"\n训练配置:")
    train_config = config['train_config']
    print(f"  学习率: {train_config['learning_rate']}")
    print(f"  权重衰减: {train_config['weight_decay']}")
    print(f"  训练轮数: {train_config['epochs']}")
    print(f"  早停耐心: {train_config['early_stopping_patience']}")
    
    print(f"\n设备信息:")
    print(f"  计算设备: {runner.device}")
    print(f"  实验目录: {runner.experiment_dir}")


if __name__ == "__main__":
    exit(main()) 