import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json
import time
from datetime import datetime
from vit_model import create_vit_model

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class QuickExperiment:
    """快速实验类，用于测试少量配置"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"quick_test_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 测试配置 - 只测试3个配置
        self.test_configs = [
            {'patch_size': 4, 'num_layers': 3, 'num_heads': 2},
            {'patch_size': 7, 'num_layers': 6, 'num_heads': 4},
            {'patch_size': 14, 'num_layers': 3, 'num_heads': 2}
        ]
        
        # 基础配置
        self.base_config = {
            'img_size': 28,
            'in_channels': 1,
            'num_classes': 10,
            'embed_dim': 64,
            'mlp_dim': 128,
            'dropout': 0.1
        }
        
        # 训练配置 - 减少epoch数
        self.base_batch_size = 64  # 基准batch size
        self.base_learning_rate = 3e-4  # 基准学习率
        
        self.train_config = {
            'batch_size': 16384,
            'learning_rate': self.calculate_lr(16384),  # 自适应学习率
            'weight_decay': 1e-4,
            'epochs': 5,  # 只训练5个epoch用于测试
            'num_workers': 0  # 避免多进程问题
        }
        
        self.results = []
    
    def calculate_lr(self, batch_size, scaling_rule='sqrt'):
        """
        根据batch size计算自适应学习率
        
        Args:
            batch_size: 当前batch size
            scaling_rule: 缩放规则 ('linear' 或 'sqrt')
        """
        if scaling_rule == 'linear':
            # 线性缩放: lr ∝ batch_size
            scale_factor = batch_size / self.base_batch_size
        elif scaling_rule == 'sqrt':
            # 平方根缩放: lr ∝ √batch_size (更保守)
            scale_factor = (batch_size / self.base_batch_size) ** 0.5
        else:
            scale_factor = 1.0
        
        new_lr = self.base_learning_rate * scale_factor
        print(f"Batch size: {batch_size}, LR scaling: {scale_factor:.3f}, New LR: {new_lr:.6f}")
        return new_lr
    
    def get_data_loaders(self):
        """创建数据加载器"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.train_config['batch_size'], 
            shuffle=True, num_workers=self.train_config['num_workers']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.train_config['batch_size'], 
            shuffle=False, num_workers=self.train_config['num_workers']
        )
        
        return train_loader, test_loader
    
    def train_and_evaluate(self, model_config, exp_name):
        """训练和评估单个模型"""
        print(f"\n开始实验: {exp_name}")
        print(f"配置: {model_config}")
        
        # 创建模型
        model = create_vit_model(model_config).to(self.device)
        
        # 获取数据
        train_loader, test_loader = self.get_data_loaders()
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.train_config['learning_rate'], 
            weight_decay=self.train_config['weight_decay']
        )
        
        # 训练历史
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, self.train_config['epochs'] + 1):
            # 训练
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output, _ = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # 评估
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output, _ = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            test_loss /= len(test_loader)
            test_acc = 100. * correct / total
            
            # 记录历史
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        training_time = time.time() - start_time
        
        # 生成注意力可视化
        self.visualize_attention(model, test_loader, exp_name)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 保存结果
        result = {
            'experiment_name': exp_name,
            'model_config': model_config,
            'best_accuracy': max(test_accs),
            'final_accuracy': test_accs[-1],
            'training_time': training_time,
            'total_params': total_params,
            'train_history': {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_accs': test_accs
            }
        }
        
        self.results.append(result)
        
        print(f"实验 {exp_name} 完成!")
        print(f"最佳准确率: {max(test_accs):.2f}%")
        print(f"训练时间: {training_time:.1f}秒")
        print(f"模型参数: {total_params:,}")
        
        return result
    
    def visualize_attention(self, model, test_loader, exp_name):
        """可视化注意力图"""
        model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, attention_maps = model(data)
                break
        
        # 选择第一个样本
        sample_image = data[0].cpu().squeeze()
        if len(attention_maps) > 0:
            sample_attention = attention_maps[-1][0].cpu()  # 最后一层的注意力图
            
            num_heads = sample_attention.shape[0]
            patch_size = model.patch_size
            num_patches = (28 // patch_size)
            
            if num_heads == 0:
                return
            
            # 动态确定子图数量
            max_heads_to_show = min(num_heads, 3)
            total_plots = max_heads_to_show + 1  # 原图 + 注意力头
            
            # 创建子图
            fig, axes = plt.subplots(1, total_plots, figsize=(4 * total_plots, 4))
            if total_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # 原始图像
            axes[plot_idx].imshow(sample_image, cmap='gray')
            axes[plot_idx].set_title('Original Image')
            axes[plot_idx].axis('off')
            plot_idx += 1
            
            # 可视化注意力头
            for i in range(max_heads_to_show):
                if plot_idx < len(axes):
                    try:
                        # CLS token对其他token的注意力
                        cls_attention = sample_attention[i, 0, 1:].reshape(num_patches, num_patches)
                        
                        im = axes[plot_idx].imshow(cls_attention, cmap='hot')
                        axes[plot_idx].set_title(f'Attention Head {i+1}')
                        axes[plot_idx].axis('off')
                        plt.colorbar(im, ax=axes[plot_idx])
                        plot_idx += 1
                    except Exception as e:
                        print(f"警告: 无法可视化注意力头 {i+1}: {str(e)}")
                        axes[plot_idx].axis('off')
                        plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{exp_name}_attention.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_batch_size_experiments(self):
        """运行batch size对比实验"""
        print("开始Batch Size对比实验...")
        
        # 测试不同的batch size
        batch_sizes = [64, 256, 1024, 4096, 16384]
        
        # 使用固定的模型配置进行对比
        fixed_model_config = self.base_config.copy()
        fixed_model_config.update({'patch_size': 7, 'num_layers': 6, 'num_heads': 4})
        
        for batch_size in batch_sizes:
            # 更新batch size和对应的学习率
            original_batch_size = self.train_config['batch_size']
            original_lr = self.train_config['learning_rate']
            
            self.train_config['batch_size'] = batch_size
            self.train_config['learning_rate'] = self.calculate_lr(batch_size)
            
            exp_name = f"batch_{batch_size}_lr_{self.train_config['learning_rate']:.6f}"
            
            print(f"\n测试 Batch Size: {batch_size}")
            
            try:
                self.train_and_evaluate(fixed_model_config, exp_name)
            except Exception as e:
                print(f"实验 {exp_name} 失败: {str(e)}")
                continue
        
        # 恢复原始设置
        self.train_config['batch_size'] = original_batch_size
        self.train_config['learning_rate'] = original_lr
        
        # 生成报告
        self.generate_batch_size_report()
    
    def run_experiments(self):
        """运行所有实验"""
        print("开始快速实验测试...")
        print(f"测试配置数量: {len(self.test_configs)}")
        
        for i, config in enumerate(self.test_configs, 1):
            # 创建完整的模型配置
            model_config = self.base_config.copy()
            model_config.update(config)
            
            exp_name = f"test_{i}_patch{config['patch_size']}_layers{config['num_layers']}_heads{config['num_heads']}"
            
            print(f"\n进度: {i}/{len(self.test_configs)}")
            
            try:
                self.train_and_evaluate(model_config, exp_name)
            except Exception as e:
                print(f"实验 {exp_name} 失败: {str(e)}")
                continue
        
        # 生成简单报告
        self.generate_report()
    
    def generate_report(self):
        """生成实验报告"""
        if not self.results:
            print("没有实验结果可以报告")
            return
        
        # 保存结果
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 生成对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 准确率对比
        names = [r['experiment_name'] for r in self.results]
        accs = [r['best_accuracy'] for r in self.results]
        times = [r['training_time'] for r in self.results]
        
        ax1.bar(range(len(names)), accs)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.replace('test_', '') for n in names], rotation=45)
        ax1.set_title('Best Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        
        # 训练时间对比
        ax2.bar(range(len(names)), times)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n.replace('test_', '') for n in names], rotation=45)
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成文本报告
        with open(os.path.join(self.output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write("快速实验测试报告\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("实验配置:\n")
            for i, result in enumerate(self.results, 1):
                config = result['model_config']
                f.write(f"{i}. {result['experiment_name']}\n")
                f.write(f"   Patch size: {config['patch_size']}\n")
                f.write(f"   Layers: {config['num_layers']}\n")
                f.write(f"   Heads: {config['num_heads']}\n")
                f.write(f"   Best accuracy: {result['best_accuracy']:.2f}%\n")
                f.write(f"   Training time: {result['training_time']:.1f}s\n")
                f.write(f"   Parameters: {result['total_params']:,}\n\n")
            
            f.write("总结:\n")
            f.write(f"最高准确率: {max(r['best_accuracy'] for r in self.results):.2f}%\n")
            f.write(f"平均准确率: {np.mean([r['best_accuracy'] for r in self.results]):.2f}%\n")
            f.write(f"平均训练时间: {np.mean([r['training_time'] for r in self.results]):.1f}s\n")
        
        print(f"\n快速实验完成！结果保存在: {self.output_dir}")
    
    def generate_batch_size_report(self):
        """生成batch size对比报告"""
        if not self.results:
            print("没有实验结果可以报告")
            return
        
        # 筛选batch size实验
        batch_results = [r for r in self.results if r['experiment_name'].startswith('batch_')]
        
        if not batch_results:
            print("没有batch size实验结果")
            return
        
        # 生成对比图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 提取数据
        batch_sizes = []
        learning_rates = []
        accuracies = []
        times = []
        
        for result in batch_results:
            # 从实验名称中提取batch size
            name_parts = result['experiment_name'].split('_')
            batch_size = int(name_parts[1])
            lr = float(name_parts[3])
            
            batch_sizes.append(batch_size)
            learning_rates.append(lr)
            accuracies.append(result['best_accuracy'])
            times.append(result['training_time'])
        
        # 准确率 vs Batch Size
        ax1.plot(batch_sizes, accuracies, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Best Accuracy (%)')
        ax1.set_title('Accuracy vs Batch Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # 学习率 vs Batch Size
        ax2.plot(batch_sizes, learning_rates, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Scaling')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 训练时间 vs Batch Size
        ax3.plot(batch_sizes, times, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time vs Batch Size')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # 效率 (准确率/时间) vs Batch Size
        efficiency = [acc/time for acc, time in zip(accuracies, times)]
        ax4.plot(batch_sizes, efficiency, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Efficiency (Accuracy/Time)')
        ax4.set_title('Training Efficiency')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'batch_size_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成详细报告
        with open(os.path.join(self.output_dir, 'batch_size_report.txt'), 'w', encoding='utf-8') as f:
            f.write("Batch Size vs Learning Rate 实验报告\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("实验结果:\n")
            f.write(f"{'Batch Size':<12} {'Learning Rate':<15} {'Accuracy':<12} {'Time':<10} {'Efficiency':<12}\n")
            f.write("-" * 70 + "\n")
            
            for i, result in enumerate(batch_results):
                name_parts = result['experiment_name'].split('_')
                batch_size = int(name_parts[1])
                lr = float(name_parts[3])
                acc = result['best_accuracy']
                time_taken = result['training_time']
                eff = acc / time_taken
                
                f.write(f"{batch_size:<12} {lr:<15.6f} {acc:<12.2f} {time_taken:<10.1f} {eff:<12.3f}\n")
            
            f.write(f"\n分析:\n")
            best_acc_idx = accuracies.index(max(accuracies))
            best_eff_idx = efficiency.index(max(efficiency))
            
            f.write(f"最高准确率: {max(accuracies):.2f}% (Batch Size: {batch_sizes[best_acc_idx]})\n")
            f.write(f"最高效率: {max(efficiency):.3f} (Batch Size: {batch_sizes[best_eff_idx]})\n")
            f.write(f"平均准确率: {np.mean(accuracies):.2f}%\n")
            
            f.write(f"\n建议:\n")
            if accuracies[0] > accuracies[-1]:  # 小batch size效果更好
                f.write("- 较小的batch size (64-256) 可能有更好的泛化性能\n")
                f.write("- 可以考虑使用gradient accumulation来模拟大batch size的效果\n")
            else:
                f.write("- 大batch size配合学习率缩放效果良好\n")
                f.write("- 可以进一步增大batch size来提高训练效率\n")
        
        print(f"Batch size分析报告已保存到: {self.output_dir}/batch_size_analysis.png")
        print(f"详细报告: {self.output_dir}/batch_size_report.txt")


def main():
    """主函数"""
    print("Vision Transformer MNIST 快速实验测试")
    print("=" * 40)
    
    # 创建实验对象
    experiment = QuickExperiment()
    
    # 让用户选择实验类型
    print("\n选择实验类型:")
    print("1. 模型架构对比实验")
    print("2. Batch Size vs Learning Rate 对比实验")
    print("3. 运行所有实验")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        experiment.run_experiments()
    elif choice == '2':
        experiment.run_batch_size_experiments()
    elif choice == '3':
        print("\n首先运行模型架构实验...")
        experiment.run_experiments()
        print("\n然后运行Batch Size实验...")
        experiment.run_batch_size_experiments()
    else:
        print("无效选择，运行默认的模型架构实验...")
        experiment.run_experiments()


if __name__ == "__main__":
    main() 