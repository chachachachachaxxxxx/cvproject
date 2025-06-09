#!/usr/bin/env python3
"""
测试实验框架的基本功能
快速验证修改后的代码是否能正常运行
"""

import sys
import os
import json
import tempfile
import shutil

# 添加上级目录到Python路径，以便导入vit_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import():
    """测试导入模块"""
    print("测试1: 导入模块...")
    try:
        from experiment_runner_fixed import ExperimentRunner
        print("✓ 成功导入 ExperimentRunner")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n测试2: 配置文件加载...")
    try:
        # 创建临时配置文件
        config = {
            "patch_sizes": [7],
            "num_layers_list": [3],
            "num_heads_list": [2],
            "embed_dims": [64],
            "dropout_rates": [0.1],
            "base_config": {
                "img_size": 28,
                "in_channels": 1,
                "num_classes": 10,
                "embed_dim": 64,
                "mlp_dim": 128,
                "dropout": 0.1
            },
            "train_config": {
                "batch_size": 32,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "epochs": 2,
                "num_workers": 0,
                "early_stopping_patience": 5,
                "min_delta": 0.001
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            from experiment_runner_fixed import ExperimentRunner
            runner = ExperimentRunner(config_file=config_file)
            print("✓ 成功加载配置文件")
            return True
        finally:
            os.unlink(config_file)
            
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试3: 模型创建...")
    try:
        from vit_model import create_vit_model
        
        config = {
            'img_size': 28,
            'patch_size': 7,
            'in_channels': 1,
            'num_classes': 10,
            'embed_dim': 64,
            'num_layers': 3,
            'num_heads': 2,
            'mlp_dim': 128,
            'dropout': 0.1
        }
        
        model = create_vit_model(config)
        print("✓ 成功创建ViT模型")
        
        # 测试模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数数量: {total_params:,}")
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_runner_initialization():
    """测试运行器初始化"""
    print("\n测试4: 运行器初始化...")
    try:
        from experiment_runner_fixed import ExperimentRunner
        
        # 创建临时输出目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            runner = ExperimentRunner(base_output_dir=temp_dir)
            print("✓ 成功初始化ExperimentRunner")
            print(f"  设备: {runner.device}")
            print(f"  实验目录: {runner.experiment_dir}")
            
            # 测试配置保存
            runner.save_experiment_config()
            config_file = os.path.join(runner.experiment_dir, 'experiment_config.json')
            if os.path.exists(config_file):
                print("✓ 成功保存实验配置")
            else:
                print("✗ 实验配置保存失败")
                return False
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ 运行器初始化失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n测试5: 数据加载...")
    try:
        from experiment_runner_fixed import ExperimentRunner
        
        runner = ExperimentRunner()
        runner.train_config['batch_size'] = 32  # 使用小batch size测试
        
        train_loader, test_loader = runner.get_data_loaders()
        print("✓ 成功创建数据加载器")
        print(f"  训练集batch数: {len(train_loader)}")
        print(f"  测试集batch数: {len(test_loader)}")
        
        # 测试一个batch
        for data, target in train_loader:
            print(f"  数据形状: {data.shape}")
            print(f"  标签形状: {target.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_flops_estimation():
    """测试FLOPs估算"""
    print("\n测试6: FLOPs估算...")
    try:
        from experiment_runner_fixed import ExperimentRunner
        
        runner = ExperimentRunner()
        
        config = {
            'img_size': 28,
            'patch_size': 7,
            'embed_dim': 64,
            'num_layers': 3,
            'num_heads': 2,
            'mlp_dim': 128
        }
        
        flops = runner._estimate_flops(config)
        print(f"✓ 成功估算FLOPs: {flops:,}")
        return True
        
    except Exception as e:
        print(f"✗ FLOPs估算失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("Vision Transformer 实验框架测试")
    print("=" * 40)
    
    tests = [
        test_import,
        test_config_loading,
        test_model_creation,
        test_runner_initialization,
        test_data_loading,
        test_flops_estimation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("测试失败，请检查代码")
    
    print("\n" + "=" * 40)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！实验框架可以正常使用。")
        print("\n建议运行方式:")
        print("1. 快速测试: python experiment_runner_fixed.py --quick")
        print("2. 完整实验: python run_experiments.py")
        return 0
    else:
        print("✗ 部分测试失败，请检查依赖和代码。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 