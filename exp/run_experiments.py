#!/usr/bin/env python3
"""
Vision Transformer MNIST 实验运行脚本

使用方法:
1. 默认运行: python run_experiments.py
2. 快速模式: python run_experiments.py --quick
3. 使用配置文件: python run_experiments.py --config experiment_config_example.json
4. 自定义输出目录: python run_experiments.py --output-dir my_experiments
"""

import sys
import os
import subprocess

def main():
    """运行实验的主函数"""
    
    print("Vision Transformer MNIST 实验启动器")
    print("=" * 50)
    
    # 检查是否存在必要文件
    required_files = ['../vit_model.py', 'experiment_runner_fixed.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("错误: 缺少必要文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保所有必要文件都在当前目录中。")
        return 1
    
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        print("选择运行模式:")
        print("1. 快速模式 (推荐新手)")
        print("2. 完整模式")
        print("3. 使用配置文件")
        print("4. 退出")
        
        while True:
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == '1':
                # 快速模式
                cmd = [sys.executable, 'experiment_runner_fixed.py', '--quick']
                break
            elif choice == '2':
                # 完整模式
                cmd = [sys.executable, 'experiment_runner_fixed.py']
                break
            elif choice == '3':
                # 配置文件模式
                if os.path.exists('experiment_config_example.json'):
                    cmd = [sys.executable, 'experiment_runner_fixed.py', 
                           '--config', 'experiment_config_example.json']
                    break
                else:
                    print("错误: 未找到配置文件 experiment_config_example.json")
                    continue
            elif choice == '4':
                print("退出程序")
                return 0
            else:
                print("无效选择，请重新输入")
    else:
        # 直接传递命令行参数
        cmd = [sys.executable, 'experiment_runner_fixed.py'] + sys.argv[1:]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("实验开始...\n")
    
    # 运行实验
    try:
        result = subprocess.run(cmd, check=True)
        print("\n实验完成!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n实验失败，错误代码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n用户中断实验")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 