#!/usr/bin/env python3
"""
测试学习率消融实验配置
"""

import json

def test_lr_ablation_config():
    """测试学习率消融实验配置"""
    
    # 加载配置文件
    with open('efficient_ablation_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("学习率消融实验配置测试")
    print("=" * 50)
    
    # 检查学习率消融组
    if 'learning_rate_ablation' in config['experiment_groups']:
        lr_group = config['experiment_groups']['learning_rate_ablation']
        print(f"描述: {lr_group['description']}")
        print(f"变量: {lr_group['variable']}")
        print(f"学习率值: {lr_group['values']}")
        print()
        
        # 显示基础配置
        print("基础配置:")
        for key, value in lr_group['base_config'].items():
            print(f"  {key}: {value}")
        print()
        
        # 模拟生成实验配置
        print("生成的实验配置:")
        for i, lr_value in enumerate(lr_group['values']):
            exp_config = lr_group['base_config'].copy()
            exp_config[lr_group['variable']] = lr_value
            exp_name = f"{lr_group['variable']}_{lr_value}"
            
            print(f"实验 {i+1}: {exp_name}")
            print(f"  学习率: {lr_value:.6f}")
            print(f"  batch_size: {exp_config['batch_size']}")
            print(f"  其他参数: patch_size={exp_config['patch_size']}, "
                  f"num_layers={exp_config['num_layers']}, "
                  f"num_heads={exp_config['num_heads']}, "
                  f"embed_dim={exp_config['embed_dim']}")
            print()
    else:
        print("错误: 未找到 learning_rate_ablation 组")
    
    # 检查正交设计中的学习率
    print("正交设计中的学习率分布:")
    if 'orthogonal_design' in config['experiment_groups']:
        orthogonal_group = config['experiment_groups']['orthogonal_design']
        lr_values = []
        for exp in orthogonal_group['experiments']:
            if 'learning_rate' in exp:
                lr_values.append(exp['learning_rate'])
        
        if lr_values:
            unique_lrs = sorted(set(lr_values))
            print(f"使用的学习率: {unique_lrs}")
            for lr in unique_lrs:
                count = lr_values.count(lr)
                print(f"  {lr:.6f}: {count} 次")
        else:
            print("正交设计中未包含学习率")
    
    print()
    print("实验统计:")
    stats = config['estimated_experiments']
    print(f"总实验数: {stats['total']}")
    print(f"单因子消融: {stats['single_factor_ablations']}")
    print(f"节省比例: {(1-stats['reduction_ratio']):.1%}")

if __name__ == "__main__":
    test_lr_ablation_config() 