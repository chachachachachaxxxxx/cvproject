#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果可视化工具
从efficient_ablation_runner.py中剥离的独立可视化模块
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from datetime import datetime

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, experiment_dir=None, results_file=None):
        """
        初始化可视化器
        Args:
            experiment_dir: 实验目录路径
            results_file: 直接指定结果文件路径
        """
        self.experiment_dir = experiment_dir
        self.results_file = results_file
        self.experiment_results = []
        self.group_results = defaultdict(list)
        
        # 加载实验结果
        self._load_results()
        
        # 创建输出目录
        if experiment_dir:
            self.output_dir = os.path.join(experiment_dir, 'visualizations')
        else:
            self.output_dir = 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loaded {len(self.experiment_results)} experiment results")
        print(f"Visualizations will be saved to: {self.output_dir}")
    
    def _load_results(self):
        """加载实验结果"""
        if self.results_file and os.path.exists(self.results_file):
            # 直接从指定文件加载
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.experiment_results = json.load(f)
        elif self.experiment_dir:
            # 从实验目录加载
            results_path = os.path.join(self.experiment_dir, 'all_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    self.experiment_results = json.load(f)
        
        # 按组分类结果
        for result in self.experiment_results:
            group_name = result.get('group_name', 'unknown')
            self.group_results[group_name].append(result)
    
    def create_overview_dashboard(self):
        """创建总览仪表板"""
        if len(self.experiment_results) < 2:
            print("Not enough results for overview dashboard")
            return
        
        # 创建多个分析图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment Results Overview', fontsize=16, fontweight='bold')
        
        # 准备数据
        data = []
        for result in self.experiment_results:
            config = result['experiment_config']
            data.append({
                'patch_size': config.get('patch_size', 0),
                'num_layers': config.get('num_layers', 0),
                'num_heads': config.get('num_heads', 0),
                'embed_dim': config.get('embed_dim', 0),
                'batch_size': config.get('batch_size', 0),
                'dropout': config.get('dropout', 0),
                'accuracy': result['best_accuracy'],
                'params': result['total_params'],
                'time': result['training_time'],
                'param_efficiency': result.get('parameter_efficiency', 0),
                'comp_efficiency': result.get('computational_efficiency', 0),
                'group': result['group_name']
            })
        
        df = pd.DataFrame(data)
        
        # 1. Group-wise accuracy distribution
        groups = df['group'].unique()
        group_accs = [df[df['group'] == g]['accuracy'].values for g in groups]
        bp = axes[0, 0].boxplot(group_accs, labels=groups, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0, 0].set_title('Accuracy Distribution by Group')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameters vs Accuracy (colored by training time)
        scatter = axes[0, 1].scatter(df['params'], df['accuracy'], 
                                   c=df['time'], cmap='viridis', alpha=0.7, s=60)
        axes[0, 1].set_title('Parameters vs Accuracy')
        axes[0, 1].set_xlabel('Number of Parameters')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('Training Time (s)')
        
        # 3. Training time distribution
        axes[0, 2].hist(df['time'], bins=min(10, len(df)//2), alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[0, 2].set_title('Training Time Distribution')
        axes[0, 2].set_xlabel('Training Time (s)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Parameter efficiency vs Computational efficiency
        axes[1, 0].scatter(df['param_efficiency'], df['comp_efficiency'], 
                          alpha=0.7, s=60, color='coral')
        axes[1, 0].set_title('Parameter vs Computational Efficiency')
        axes[1, 0].set_xlabel('Parameter Efficiency')
        axes[1, 0].set_ylabel('Computational Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Top 10 experiments
        top_results = df.nlargest(min(10, len(df)), 'accuracy')
        y_pos = np.arange(len(top_results))
        bars = axes[1, 1].barh(y_pos, top_results['accuracy'], color='lightgreen', alpha=0.8)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"Exp {i+1}" for i in range(len(top_results))])
        axes[1, 1].set_xlabel('Accuracy (%)')
        axes[1, 1].set_title('Top 10 Experiments')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加准确率数值标签在柱形右边
        for i, (bar, acc) in enumerate(zip(bars, top_results['accuracy'])):
            axes[1, 1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                           f'{acc:.2f}%', va='center', ha='left', fontsize=9)
        
        # 6. Accuracy vs Time scatter with size = parameters
        axes[1, 2].scatter(df['time'], df['accuracy'], 
                          s=df['params']/1000, alpha=0.6, color='purple')
        axes[1, 2].set_title('Training Time vs Accuracy\n(Size = Parameters)')
        axes[1, 2].set_xlabel('Training Time (s)')
        axes[1, 2].set_ylabel('Accuracy (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overview_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created overview dashboard")
    
    def create_ablation_analysis(self):
        """创建消融分析图表"""
        ablation_groups = ['patch_size_ablation', 'depth_ablation', 'heads_ablation', 
                          'embed_dim_ablation', 'batch_size_ablation', 'learning_rate_ablation']
        
        # 计算需要的子图数量
        available_groups = [g for g in ablation_groups if g in self.group_results]
        if not available_groups:
            print("No ablation groups found")
            return
        
        n_groups = len(available_groups)
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
        
        for i, group_name in enumerate(available_groups):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            results = self.group_results[group_name]
            if len(results) > 1:
                # 提取变量和准确率
                variable = group_name.replace('_ablation', '')
                if variable == 'depth':
                    variable = 'num_layers'
                elif variable == 'heads':
                    variable = 'num_heads'
                
                values = []
                accuracies = []
                times = []
                params = []
                
                for result in results:
                    config = result['experiment_config']
                    if variable in config:
                        values.append(config[variable])
                        accuracies.append(result['best_accuracy'])
                        times.append(result['training_time'])
                        params.append(result['total_params'])
                    elif variable == 'learning_rate':
                        # 特殊处理学习率
                        lr_config = result.get('training_config', {})
                        lr = lr_config.get('adjusted_learning_rate', 0)
                        if lr > 0:
                            values.append(lr)
                            accuracies.append(result['best_accuracy'])
                            times.append(result['training_time'])
                            params.append(result['total_params'])
                
                if values and accuracies:
                    # 按值排序
                    sorted_data = sorted(zip(values, accuracies, times, params))
                    values, accuracies, times, params = zip(*sorted_data)
                    
                    # 主线图：准确率
                    line1 = ax.plot(values, accuracies, 'o-', linewidth=2, 
                                   markersize=8, color='blue', label='Accuracy')
                    ax.set_ylabel('Accuracy (%)', color='blue')
                    ax.tick_params(axis='y', labelcolor='blue')
                    
                    # 次坐标轴：训练时间
                    ax2 = ax.twinx()
                    line2 = ax2.plot(values, times, 's--', linewidth=2, 
                                    markersize=6, color='red', alpha=0.7, label='Time')
                    ax2.set_ylabel('Training Time (s)', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    # 设置标题和标签
                    ax.set_title(f'{variable.replace("_", " ").title()} Ablation')
                    ax.set_xlabel(variable.replace('_', ' ').title())
                    ax.grid(True, alpha=0.3)
                    
                    # 如果是学习率，使用对数刻度
                    if variable == 'learning_rate':
                        ax.set_xscale('log')
                    
                    # 添加图例
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
                else:
                    ax.text(0.5, 0.5, f'No {variable} data available', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'Insufficient data for {group_name}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 隐藏多余的子图
        for i in range(n_groups, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ablation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created ablation analysis")
    
    def create_correlation_heatmap(self):
        """创建相关性热图"""
        if len(self.experiment_results) < 3:
            print("Not enough results for correlation analysis")
            return
        
        # 准备数据
        data = []
        for result in self.experiment_results:
            config = result['experiment_config']
            data.append({
                'patch_size': config.get('patch_size', 0),
                'num_layers': config.get('num_layers', 0),
                'num_heads': config.get('num_heads', 0),
                'embed_dim': config.get('embed_dim', 0),
                'batch_size': config.get('batch_size', 0),
                'dropout': config.get('dropout', 0),
                'accuracy': result['best_accuracy'],
                'params': result['total_params'],
                'time': result['training_time'],
                'param_efficiency': result.get('parameter_efficiency', 0),
                'comp_efficiency': result.get('computational_efficiency', 0)
            })
        
        df = pd.DataFrame(data)
        
        # 计算相关性矩阵
        correlation_matrix = df.corr()
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                             cmap='RdBu_r', center=0, square=True, 
                             linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Configuration and Performance Correlation Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存相关性矩阵到CSV
        correlation_matrix.to_csv(os.path.join(self.output_dir, 'correlation_matrix.csv'))
        print("Created correlation heatmap")
    
    def create_performance_analysis(self):
        """创建性能分析图表"""
        if len(self.experiment_results) < 2:
            print("Not enough results for performance analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
        
        # 准备数据
        data = []
        for result in self.experiment_results:
            data.append({
                'accuracy': result['best_accuracy'],
                'params': result['total_params'],
                'time': result['training_time'],
                'param_efficiency': result.get('parameter_efficiency', 0),
                'comp_efficiency': result.get('computational_efficiency', 0),
                'flops': result.get('estimated_flops', 0),
                'group': result['group_name']
            })
        
        df = pd.DataFrame(data)
        
        # 1. Pareto frontier: Accuracy vs Parameters
        axes[0, 0].scatter(df['params'], df['accuracy'], alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Number of Parameters')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy vs Model Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加Pareto前沿
        pareto_mask = self._get_pareto_frontier(df['params'].values, df['accuracy'].values)
        pareto_points = df[pareto_mask].sort_values('params')
        axes[0, 0].plot(pareto_points['params'], pareto_points['accuracy'], 
                       'r-', linewidth=2, alpha=0.7, label='Pareto Frontier')
        axes[0, 0].legend()
        
        # 2. Efficiency comparison
        efficiency_data = df[['param_efficiency', 'comp_efficiency']].values
        axes[0, 1].scatter(df['param_efficiency'], df['comp_efficiency'], 
                          alpha=0.7, s=60, c=df['accuracy'], cmap='viridis')
        axes[0, 1].set_xlabel('Parameter Efficiency')
        axes[0, 1].set_ylabel('Computational Efficiency')
        axes[0, 1].set_title('Efficiency Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training time vs Accuracy
        axes[1, 0].scatter(df['time'], df['accuracy'], alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Training Time (s)')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Training Time vs Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model complexity analysis
        if 'flops' in df.columns and df['flops'].sum() > 0:
            axes[1, 1].scatter(df['flops'], df['accuracy'], alpha=0.7, s=60)
            axes[1, 1].set_xlabel('FLOPs')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title('Computational Complexity vs Accuracy')
            axes[1, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        else:
            # 如果没有FLOPs数据，显示参数vs时间
            axes[1, 1].scatter(df['params'], df['time'], alpha=0.7, s=60)
            axes[1, 1].set_xlabel('Number of Parameters')
            axes[1, 1].set_ylabel('Training Time (s)')
            axes[1, 1].set_title('Model Size vs Training Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created performance analysis")
    
    def _get_pareto_frontier(self, x, y, maximize_both=False):
        """计算Pareto前沿点"""
        # 对于模型选择，我们通常想要最小化参数数量，最大化准确率
        # 所以对x取负值
        if not maximize_both:
            x = -np.array(x)
        
        points = np.column_stack((x, y))
        pareto_mask = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            dominated = False
            for j in range(len(points)):
                if i != j:
                    if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                        dominated = True
                        break
            if not dominated:
                pareto_mask[i] = True
        
        return pareto_mask
    
    def create_training_curves(self):
        """创建训练曲线图"""
        # 选择几个最好的实验显示训练曲线
        top_experiments = sorted(self.experiment_results, 
                               key=lambda x: x['best_accuracy'], reverse=True)[:6]
        
        if not top_experiments:
            print("No experiments with training history found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Curves - Top 6 Experiments', fontsize=16, fontweight='bold')
        
        for i, result in enumerate(top_experiments):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            history = result.get('train_history', {})
            if history and 'test_accs' in history:
                epochs = range(1, len(history['test_accs']) + 1)
                
                # 绘制训练和测试准确率
                ax.plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
                ax.plot(epochs, history['test_accs'], 'r-', label='Test Acc', linewidth=2)
                
                # 标记最佳点
                best_epoch = result['best_epoch']
                best_acc = result['best_accuracy']
                ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
                ax.scatter([best_epoch], [best_acc], color='green', s=100, zorder=5)
                
                ax.set_title(f"{result['experiment_name']}\nBest: {best_acc:.2f}% @ Epoch {best_epoch}")
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No training history available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created training curves")
    
    def generate_summary_report(self):
        """生成总结报告"""
        if not self.experiment_results:
            print("No experiment results to summarize")
            return
        
        # 找到最佳结果
        best_result = max(self.experiment_results, key=lambda x: x['best_accuracy'])
        
        # 创建总结图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experiment Summary Report', fontsize=16, fontweight='bold')
        
        # 1. 最佳结果配置雷达图
        ax = axes[0, 0]
        config = best_result['experiment_config']
        
        # 标准化配置值用于雷达图
        radar_data = {
            'Patch Size': config.get('patch_size', 4) / 8,  # 假设最大为8
            'Layers': config.get('num_layers', 6) / 12,     # 假设最大为12
            'Heads': config.get('num_heads', 8) / 16,       # 假设最大为16
            'Embed Dim': config.get('embed_dim', 256) / 512, # 假设最大为512
            'Batch Size': min(config.get('batch_size', 64) / 128, 1), # 假设最大为128
            'Dropout': config.get('dropout', 0.1) / 0.5     # 假设最大为0.5
        }
        
        categories = list(radar_data.keys())
        values = list(radar_data.values())
        
        # 简化的条形图代替雷达图
        bars = ax.barh(categories, values, color='skyblue', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_title(f'Best Configuration\n{best_result["experiment_name"]}\nAccuracy: {best_result["best_accuracy"]:.2f}%')
        ax.set_xlabel('Normalized Value')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center')
        
        # 2. 实验组比较
        ax = axes[0, 1]
        group_stats = {}
        for group_name, results in self.group_results.items():
            if results:
                accs = [r['best_accuracy'] for r in results]
                group_stats[group_name] = {
                    'mean': np.mean(accs),
                    'std': np.std(accs),
                    'max': np.max(accs),
                    'count': len(accs)
                }
        
        if group_stats:
            groups = list(group_stats.keys())
            means = [group_stats[g]['mean'] for g in groups]
            stds = [group_stats[g]['std'] for g in groups]
            
            bars = ax.bar(range(len(groups)), means, yerr=stds, 
                         capsize=5, alpha=0.7, color='lightcoral')
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45, ha='right')
            ax.set_title('Group Performance Comparison')
            ax.set_ylabel('Accuracy (%) - Mean ± Std')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5, 
                       f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 效率分析
        ax = axes[1, 0]
        accuracies = [r['best_accuracy'] for r in self.experiment_results]
        param_effs = [r.get('parameter_efficiency', 0) for r in self.experiment_results]
        
        scatter = ax.scatter(param_effs, accuracies, alpha=0.7, s=60)
        ax.set_xlabel('Parameter Efficiency')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Parameter Efficiency vs Accuracy')
        ax.grid(True, alpha=0.3)
        
        # 4. 实验统计
        ax = axes[1, 1]
        
        # 创建统计文本
        total_experiments = len(self.experiment_results)
        total_groups = len(self.group_results)
        best_acc = max(accuracies)
        worst_acc = min(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        stats_text = f"""Experiment Statistics:
        
Total Experiments: {total_experiments}
Total Groups: {total_groups}

Accuracy Statistics:
Best: {best_acc:.2f}%
Worst: {worst_acc:.2f}%
Mean: {mean_acc:.2f}%
Std: {std_acc:.2f}%

Best Experiment:
{best_result['experiment_name']}
Group: {best_result['group_name']}
Parameters: {best_result['total_params']:,}
Training Time: {best_result['training_time']:.1f}s
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Experiment Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'summary_report.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created summary report")
    
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        print("Creating all visualizations...")
        
        self.create_overview_dashboard()
        self.create_ablation_analysis()
        self.create_correlation_heatmap()
        self.create_performance_analysis()
        self.create_training_curves()
        self.generate_summary_report()
        
        print(f"All visualizations saved to: {self.output_dir}")
        
        # 创建索引文件
        self._create_index_html()
    
    def _create_index_html(self):
        """创建HTML索引文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; }}
        .card img {{ width: 100%; height: auto; }}
        .card h3 {{ margin-top: 0; color: #333; }}
        .card p {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Experiment Visualization Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>Overview Dashboard</h3>
            <img src="overview_dashboard.png" alt="Overview Dashboard">
            <p>Comprehensive overview of all experiment results including accuracy distributions, parameter analysis, and efficiency metrics.</p>
        </div>
        
        <div class="card">
            <h3>Ablation Analysis</h3>
            <img src="ablation_analysis.png" alt="Ablation Analysis">
            <p>Detailed analysis of individual hyperparameter effects on model performance.</p>
        </div>
        
        <div class="card">
            <h3>Correlation Heatmap</h3>
            <img src="correlation_heatmap.png" alt="Correlation Heatmap">
            <p>Correlation matrix showing relationships between configuration parameters and performance metrics.</p>
        </div>
        
        <div class="card">
            <h3>Performance Analysis</h3>
            <img src="performance_analysis.png" alt="Performance Analysis">
            <p>Pareto frontier analysis and efficiency comparisons across different model configurations.</p>
        </div>
        
        <div class="card">
            <h3>Training Curves</h3>
            <img src="training_curves.png" alt="Training Curves">
            <p>Training progress visualization for the top performing experiments.</p>
        </div>
        
        <div class="card">
            <h3>Summary Report</h3>
            <img src="summary_report.png" alt="Summary Report">
            <p>High-level summary with best configurations, group comparisons, and experiment statistics.</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(os.path.join(self.output_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Created visualization index: {os.path.join(self.output_dir, 'index.html')}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Results Visualizer')
    parser.add_argument('--experiment-dir', type=str, 
                       help='Path to experiment directory')
    parser.add_argument('--results-file', type=str, 
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if not args.experiment_dir and not args.results_file:
        print("Please specify either --experiment-dir or --results-file")
        return
    
    # 创建可视化器
    visualizer = ExperimentVisualizer(
        experiment_dir=args.experiment_dir,
        results_file=args.results_file
    )
    
    # 如果指定了输出目录，覆盖默认设置
    if args.output_dir != 'visualizations':
        visualizer.output_dir = args.output_dir
        os.makedirs(visualizer.output_dir, exist_ok=True)
    
    # 生成所有可视化
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main() 