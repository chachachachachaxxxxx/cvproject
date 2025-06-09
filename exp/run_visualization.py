#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行实验可视化的简单脚本
"""

import os
import sys
import argparse
from experiment_visualizer import ExperimentVisualizer

def main():
    parser = argparse.ArgumentParser(description='Generate experiment visualizations')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to experiment directory containing all_results.json')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations (default: experiment_dir/visualizations)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory '{args.experiment_dir}' does not exist")
        return 1
    
    results_file = os.path.join(args.experiment_dir, 'all_results.json')
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found")
        return 1
    
    print(f"Loading results from: {results_file}")
    
    # 创建可视化器
    visualizer = ExperimentVisualizer(experiment_dir=args.experiment_dir)
    
    # 如果指定了输出目录，覆盖默认设置
    if args.output_dir:
        visualizer.output_dir = args.output_dir
        os.makedirs(visualizer.output_dir, exist_ok=True)
    
    print(f"Generating visualizations...")
    
    # 生成所有可视化
    try:
        visualizer.create_all_visualizations()
        print(f"\nAll visualizations saved to: {visualizer.output_dir}")
        print(f"Open {os.path.join(visualizer.output_dir, 'index.html')} to view the report")
        return 0
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 