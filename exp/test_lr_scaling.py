#!/usr/bin/env python3
"""
测试学习率根据batch size调整的逻辑
"""

import json

def test_lr_scaling():
    """测试学习率缩放逻辑"""
    
    # 模拟配置
    train_config = {
        'learning_rate': 3e-4,
        'base_batch_size': 64,
        'lr_scaling_factor': 0.5,
        'max_learning_rate': 1e-3
    }
    
    # 测试不同的batch size
    test_batch_sizes = [64, 256, 1024]
    
    print("学习率调整测试")
    print("=" * 50)
    print(f"基础学习率: {train_config['learning_rate']:.6f}")
    print(f"基础batch size: {train_config['base_batch_size']}")
    print(f"缩放因子: {train_config['lr_scaling_factor']}")
    print(f"最大学习率: {train_config['max_learning_rate']:.6f}")
    print()
    
    for batch_size in test_batch_sizes:
        # 计算调整后的学习率
        base_lr = train_config['learning_rate']
        base_batch_size = train_config['base_batch_size']
        lr_scaling_factor = train_config['lr_scaling_factor']
        max_lr = train_config['max_learning_rate']
        
        # 线性缩放学习率
        adjusted_lr = base_lr * (batch_size / base_batch_size) ** lr_scaling_factor
        adjusted_lr = min(adjusted_lr, max_lr)
        
        scaling_ratio = batch_size / base_batch_size
        actual_scaling = adjusted_lr / base_lr
        
        print(f"Batch Size: {batch_size:4d}")
        print(f"  理论缩放比例: {scaling_ratio:.3f}")
        print(f"  实际缩放比例: {actual_scaling:.3f}")
        print(f"  调整后学习率: {adjusted_lr:.6f}")
        print(f"  是否达到上限: {'是' if adjusted_lr >= max_lr else '否'}")
        print()

if __name__ == "__main__":
    test_lr_scaling() 