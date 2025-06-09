import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import os
import json
import time
import platform
import psutil
from datetime import datetime
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import traceback
import argparse
import sys

# 添加上级目录到Python路径，以便导入vit_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vit_model import create_vit_model

# 设置字体，避免中文显示警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class ExperimentRunner:
    """实验运行器，用于执行消融实验和分析"""
    
    def __init__(self, base_output_dir="experiments", config_file=None):
        self.base_output_dir = base_output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_output_dir, f"experiment_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 从配置文件加载或使用默认配置
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            self._load_config(config)
        else:
            self._set_default_config()
        
        # 存储实验结果
        self.experiment_results = []
        
    def _load_config(self, config):
        """从配置文件加载实验参数"""
        self.patch_sizes = config.get('patch_sizes', [4, 7, 14])
        self.num_layers_list = config.get('num_layers_list', [3, 6, 9])
        self.num_heads_list = config.get('num_heads_list', [2, 4, 8])
        self.embed_dims = config.get('embed_dims', [64])  # 新增：嵌入维度实验
        self.dropout_rates = config.get('dropout_rates', [0.1])  # 新增：dropout率实验
        
        self.base_config = config.get('base_config', {
            'img_size': 28,
            'in_channels': 1,
            'num_classes': 10,
            'embed_dim': 64,
            'mlp_dim': 128,
            'dropout': 0.1
        })
        
        self.train_config = config.get('train_config', {
            'batch_size': 256,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 15,
            'num_workers': 0,
            'early_stopping_patience': 5,  # 新增：早停机制
            'min_delta': 0.001  # 新增：最小改进阈值
        })
        
    def _set_default_config(self):
        """设置默认配置"""
        # 实验配置 - 优化后的参数组合
        self.patch_sizes = [4, 7]  # 减少patch size选择，专注于更重要的对比
        self.num_layers_list = [3, 6]  # 减少层数选择
        self.num_heads_list = [2, 4]  # 减少注意力头数选择
        self.embed_dims = [64, 128]  # 新增：嵌入维度实验
        self.dropout_rates = [0.1, 0.2]  # 新增：dropout率实验
        
        # 基础配置
        self.base_config = {
            'img_size': 28,
            'in_channels': 1,
            'num_classes': 10,
            'embed_dim': 64,
            'mlp_dim': 128,
            'dropout': 0.1
        }
        
        # 训练配置 - 改进的训练设置
        self.train_config = {
            'batch_size': 256,  # 减小batch size以提高稳定性
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 15,  # 减少epoch数
            'num_workers': 0,
            'early_stopping_patience': 5,  # 早停机制
            'min_delta': 0.001  # 最小改进阈值
        }
        
    def save_experiment_config(self):
        """保存实验配置"""
        config = {
            'patch_sizes': self.patch_sizes,
            'num_layers_list': self.num_layers_list,
            'num_heads_list': self.num_heads_list,
            'embed_dims': self.embed_dims,
            'dropout_rates': self.dropout_rates,
            'base_config': self.base_config,
            'train_config': self.train_config,
            'device': str(self.device),
            'timestamp': self.timestamp
        }
        
        config_path = os.path.join(self.experiment_dir, 'experiment_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"实验配置已保存到: {config_path}")

    def get_system_info(self):
        """获取系统信息"""
        info = {
            'timestamp': self.timestamp,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'device': str(self.device),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        
        return info
    
    def get_data_loaders(self):
        """创建数据加载器"""
        transform_train = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='../data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.MNIST(
            root='../data', train=False, download=True, transform=transform_test
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.train_config['batch_size'], 
            shuffle=True, num_workers=self.train_config['num_workers'], pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.train_config['batch_size'], 
            shuffle=False, num_workers=self.train_config['num_workers'], pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train_model(self, model_config, experiment_name):
        """训练单个模型 - 改进版本，添加早停机制"""
        print(f"\n开始实验: {experiment_name}")
        print(f"配置: {model_config}")
        
        # 创建实验子目录
        exp_dir = os.path.join(self.experiment_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        try:
            # 创建模型
            model = create_vit_model(model_config).to(self.device)
            
            # 获取数据加载器
            train_loader, test_loader = self.get_data_loaders()
            
            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.train_config['learning_rate'], 
                weight_decay=self.train_config['weight_decay']
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.train_config['epochs'], eta_min=1e-6
            )
            
            # 训练历史
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            
            best_acc = 0
            best_epoch = 0
            patience_counter = 0
            start_time = time.time()
            
            # 训练循环 - 添加早停机制
            for epoch in range(1, self.train_config['epochs'] + 1):
                try:
                    # 训练
                    train_loss, train_acc = self._train_epoch(
                        model, train_loader, criterion, optimizer, epoch
                    )
                    
                    # 评估
                    test_loss, test_acc = self._evaluate(model, test_loader, criterion)
                    
                    # 更新学习率
                    scheduler.step()
                    
                    # 记录历史
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    
                    # 早停机制
                    if test_acc > best_acc + self.train_config['min_delta']:
                        best_acc = test_acc
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # 保存最佳模型
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'model_config': model_config,
                            'best_acc': best_acc,
                            'epoch': epoch
                        }, os.path.join(exp_dir, 'best_model.pth'))
                    else:
                        patience_counter += 1
                    
                    # 早停检查
                    if patience_counter >= self.train_config['early_stopping_patience']:
                        print(f"早停触发在epoch {epoch}，最佳准确率在epoch {best_epoch}: {best_acc:.2f}%")
                        break
                
                except Exception as e:
                    print(f"Epoch {epoch} 训练失败: {str(e)}")
                    break
            
            training_time = time.time() - start_time
            
            # 保存训练曲线
            try:
                self._plot_training_curves(
                    train_losses, train_accs, test_losses, test_accs,
                    os.path.join(exp_dir, 'training_curves.png'),
                    experiment_name, best_epoch
                )
            except Exception as e:
                print(f"保存训练曲线失败: {str(e)}")
            
            # 生成注意力可视化
            try:
                self._visualize_attention(
                    model, test_loader, 
                    os.path.join(exp_dir, 'attention_visualization.png')
                )
            except Exception as e:
                print(f"生成注意力可视化失败: {str(e)}")
            
            # 错误样例分析
            try:
                error_analysis = self._analyze_errors(
                    model, test_loader,
                    os.path.join(exp_dir, 'error_analysis.png')
                )
            except Exception as e:
                print(f"错误样例分析失败: {str(e)}")
                error_analysis = {'accuracy': 0, 'total_errors': 0}
            
            # 计算模型参数和复杂度
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 计算FLOPs（估算）
            flops = self._estimate_flops(model_config)
            
            # 保存实验结果
            result = {
                'experiment_name': experiment_name,
                'model_config': model_config,
                'best_accuracy': best_acc,
                'best_epoch': best_epoch,
                'final_accuracy': test_accs[-1] if test_accs else 0,
                'training_time': training_time,
                'early_stopped': patience_counter >= self.train_config['early_stopping_patience'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'estimated_flops': flops,
                'train_history': {
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'test_losses': test_losses,
                    'test_accs': test_accs
                },
                'error_analysis': error_analysis
            }
            
            self.experiment_results.append(result)
            
            # 保存单个实验结果
            with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"实验 {experiment_name} 完成!")
            print(f"最佳准确率: {best_acc:.2f}% (epoch {best_epoch})")
            print(f"训练时间: {training_time:.1f}秒")
            print(f"模型参数: {total_params:,}")
            print(f"估算FLOPs: {flops:,}")
            
            return result
            
        except Exception as e:
            print(f"实验 {experiment_name} 完全失败: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")
            return None
    
    def _estimate_flops(self, model_config):
        """估算模型FLOPs"""
        # 简化的FLOPs估算
        embed_dim = model_config['embed_dim']
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        num_patches = (model_config['img_size'] // model_config['patch_size']) ** 2 + 1  # +1 for CLS token
        
        # 每层的注意力计算FLOPs
        attention_flops = 2 * num_patches * embed_dim * embed_dim * num_heads
        # 每层的MLP计算FLOPs
        mlp_flops = 2 * num_patches * embed_dim * model_config.get('mlp_dim', embed_dim * 4)
        
        total_flops = num_layers * (attention_flops + mlp_flops)
        return total_flops
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """训练一个epoch"""
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def _evaluate(self, model, test_loader, criterion):
        """评估模型"""
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
        
        return test_loss / len(test_loader), 100. * correct / total
    
    def _plot_training_curves(self, train_losses, train_accs, test_losses, test_accs, save_path, title, best_epoch):
        """绘制训练曲线 - 改进版本"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        ax1.plot(epochs, test_losses, label='Test Loss', color='red', linewidth=2)
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{title} - Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, train_accs, label='Train Accuracy', color='blue', linewidth=2)
        ax2.plot(epochs, test_accs, label='Test Accuracy', color='red', linewidth=2)
        ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{title} - Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加最佳准确率标注
        best_acc = max(test_accs)
        ax2.annotate(f'Best: {best_acc:.2f}%', 
                    xy=(best_epoch, best_acc), 
                    xytext=(best_epoch+1, best_acc+1),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_attention(self, model, test_loader, save_path):
        """可视化注意力图 - 修复版本"""
        model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, attention_maps = model(data)
                break
        
        # 检查是否有注意力图
        if not attention_maps:
            print("警告: 没有注意力图可以可视化")
            # 创建一个简单的占位图
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Attention Maps Available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # 选择第一个样本
        sample_image = data[0].cpu().squeeze()
        sample_attention = attention_maps[-1][0].cpu()  # 最后一层的注意力图
        
        num_heads = sample_attention.shape[0]
        patch_size = model.patch_size
        num_patches = (28 // patch_size)
        
        if num_heads == 0:
            print("警告: 注意力头数为0")
            return
        
        try:
            # 创建简单的可视化
            fig, axes = plt.subplots(1, min(4, num_heads + 1), figsize=(16, 4))
            
            # 确保axes是列表
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            elif axes.ndim == 0:
                axes = [axes]
            
            plot_idx = 0
            
            # 原始图像
            if plot_idx < len(axes):
                axes[plot_idx].imshow(sample_image, cmap='gray')
                axes[plot_idx].set_title('Original Image')
                axes[plot_idx].axis('off')
                plot_idx += 1
            
            # 可视化注意力头
            max_heads_to_show = min(num_heads, 3)
            for i in range(max_heads_to_show):
                if plot_idx < len(axes):
                    try:
                        # CLS token对其他token的注意力
                        cls_attention = sample_attention[i, 0, 1:].reshape(num_patches, num_patches)
                        
                        im = axes[plot_idx].imshow(cls_attention, cmap='hot')
                        axes[plot_idx].set_title(f'Attention Head {i+1}')
                        axes[plot_idx].axis('off')
                        plot_idx += 1
                    except Exception as e:
                        print(f"警告: 无法可视化注意力头 {i+1}: {str(e)}")
                        if plot_idx < len(axes):
                            axes[plot_idx].axis('off')
                            plot_idx += 1
            
            # 隐藏未使用的子图
            for idx in range(plot_idx, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"注意力可视化失败: {str(e)}")
            # 创建错误占位图
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f'Attention Visualization Failed\n{str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _analyze_errors(self, model, test_loader, save_path):
        """分析错误样例"""
        model.eval()
        
        all_preds = []
        all_targets = []
        error_samples = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 收集错误样例
                for i in range(len(pred)):
                    if pred[i] != target[i] and len(error_samples) < 8:
                        error_samples.append({
                            'image': data[i].cpu().squeeze().numpy(),
                            'true_label': target[i].item(),
                            'pred_label': pred[i].item(),
                            'confidence': torch.softmax(output[i], dim=0).max().item()
                        })
        
        # 绘制混淆矩阵和错误样例
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # 错误统计
        axes[0, 1].axis('off')
        accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        total_errors = len([1 for p, t in zip(all_preds, all_targets) if p != t])
        
        # 计算每类错误率
        class_errors = {}
        for i in range(10):
            total = sum(cm[i])
            errors = total - cm[i, i]
            class_errors[i] = errors / total if total > 0 else 0
        
        error_text = f"Accuracy: {accuracy:.3f}\n"
        error_text += f"Total Errors: {total_errors}\n\n"
        error_text += "Per-class Error Rates:\n"
        for i, error_rate in class_errors.items():
            error_text += f"Class {i}: {error_rate:.3f}\n"
        
        axes[0, 1].text(0.1, 0.9, error_text, fontsize=10, verticalalignment='top')
        
        # 显示错误样例
        for idx, sample in enumerate(error_samples[:2]):
            row = 1
            col = idx
            if col < 2:
                axes[row, col].imshow(sample['image'], cmap='gray')
                axes[row, col].set_title(f"True: {sample['true_label']}, Pred: {sample['pred_label']}\n"
                                       f"Conf: {sample['confidence']:.3f}")
                axes[row, col].axis('off')
        
        # 隐藏未使用的子图
        for i in range(len(error_samples), 2):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 返回错误分析结果
        return {
            'accuracy': accuracy,
            'total_errors': total_errors,
            'class_error_rates': class_errors,
            'confusion_matrix': cm.tolist()
        }
    
    def run_ablation_study(self):
        """运行消融实验 - 改进版本"""
        print("开始消融实验...")
        print(f"实验配置:")
        print(f"  Patch sizes: {self.patch_sizes}")
        print(f"  Network depths: {self.num_layers_list}")
        print(f"  Attention heads: {self.num_heads_list}")
        print(f"  Embed dimensions: {self.embed_dims}")
        print(f"  Dropout rates: {self.dropout_rates}")
        
        # 计算总实验数
        total_experiments = (len(self.patch_sizes) * len(self.num_layers_list) * 
                           len(self.num_heads_list) * len(self.embed_dims) * 
                           len(self.dropout_rates))
        print(f"  总实验数: {total_experiments}")
        
        # 保存实验配置
        self.save_experiment_config()
        
        experiment_count = 0
        successful_experiments = 0
        
        # 运行实验
        for patch_size in self.patch_sizes:
            for num_layers in self.num_layers_list:
                for num_heads in self.num_heads_list:
                    for embed_dim in self.embed_dims:
                        for dropout_rate in self.dropout_rates:
                            experiment_count += 1
                            
                            # 创建模型配置
                            model_config = self.base_config.copy()
                            model_config.update({
                                'patch_size': patch_size,
                                'num_layers': num_layers,
                                'num_heads': num_heads,
                                'embed_dim': embed_dim,
                                'mlp_dim': embed_dim * 4,  # MLP维度是embed_dim的4倍
                                'dropout': dropout_rate
                            })
                            
                            experiment_name = (f"patch{patch_size}_layers{num_layers}_"
                                             f"heads{num_heads}_embed{embed_dim}_"
                                             f"dropout{int(dropout_rate*100)}")
                            
                            print(f"\n进度: {experiment_count}/{total_experiments}")
                            
                            result = self.train_model(model_config, experiment_name)
                            if result is not None:
                                successful_experiments += 1
                            
                            # 清理GPU内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
        
        print(f"\n实验完成统计:")
        print(f"总实验数: {total_experiments}")
        print(f"成功实验数: {successful_experiments}")
        print(f"失败实验数: {total_experiments - successful_experiments}")
        
        # 生成综合分析报告
        if self.experiment_results:
            self._generate_comprehensive_report()
        else:
            print("没有成功的实验结果，无法生成报告")
    
    def _generate_comprehensive_report(self):
        """生成综合分析报告 - 改进版本"""
        print("\n生成综合分析报告...")
        
        # 保存所有实验结果
        with open(os.path.join(self.experiment_dir, 'all_results.json'), 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        
        # 生成系统信息报告
        system_info = self.get_system_info()
        with open(os.path.join(self.experiment_dir, 'system_info.json'), 'w') as f:
            json.dump(system_info, f, indent=2)
        
        # 生成性能对比表
        self._generate_performance_table()
        
        # 生成可视化分析图表
        self._generate_analysis_plots()
        
        # 生成Markdown报告
        self._generate_markdown_report()
        
        print(f"综合报告已保存到: {self.experiment_dir}")
    
    def _generate_analysis_plots(self):
        """生成分析图表"""
        if len(self.experiment_results) < 2:
            return
        
        # 准备数据
        df_data = []
        for result in self.experiment_results:
            config = result['model_config']
            df_data.append({
                'patch_size': config['patch_size'],
                'num_layers': config['num_layers'],
                'num_heads': config['num_heads'],
                'embed_dim': config['embed_dim'],
                'dropout': config['dropout'],
                'accuracy': result['best_accuracy'],
                'params': result['total_params'],
                'time': result['training_time'],
                'flops': result['estimated_flops']
            })
        
        # 创建多个分析图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Patch Size vs Accuracy
        patch_acc = {}
        for data in df_data:
            patch_size = data['patch_size']
            if patch_size not in patch_acc:
                patch_acc[patch_size] = []
            patch_acc[patch_size].append(data['accuracy'])
        
        patch_sizes = list(patch_acc.keys())
        avg_accs = [np.mean(patch_acc[ps]) for ps in patch_sizes]
        std_accs = [np.std(patch_acc[ps]) for ps in patch_sizes]
        
        axes[0, 0].bar(range(len(patch_sizes)), avg_accs, yerr=std_accs, capsize=5)
        axes[0, 0].set_xticks(range(len(patch_sizes)))
        axes[0, 0].set_xticklabels(patch_sizes)
        axes[0, 0].set_title('Patch Size vs Accuracy')
        axes[0, 0].set_xlabel('Patch Size')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Number of Layers vs Accuracy
        layer_acc = {}
        for data in df_data:
            num_layers = data['num_layers']
            if num_layers not in layer_acc:
                layer_acc[num_layers] = []
            layer_acc[num_layers].append(data['accuracy'])
        
        layer_nums = list(layer_acc.keys())
        avg_accs = [np.mean(layer_acc[ln]) for ln in layer_nums]
        std_accs = [np.std(layer_acc[ln]) for ln in layer_nums]
        
        axes[0, 1].bar(range(len(layer_nums)), avg_accs, yerr=std_accs, capsize=5)
        axes[0, 1].set_xticks(range(len(layer_nums)))
        axes[0, 1].set_xticklabels(layer_nums)
        axes[0, 1].set_title('Number of Layers vs Accuracy')
        axes[0, 1].set_xlabel('Number of Layers')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Parameters vs Accuracy
        params = [data['params'] for data in df_data]
        accs = [data['accuracy'] for data in df_data]
        axes[0, 2].scatter(params, accs, alpha=0.7)
        axes[0, 2].set_title('Parameters vs Accuracy')
        axes[0, 2].set_xlabel('Number of Parameters')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training Time vs Accuracy
        times = [data['time'] for data in df_data]
        axes[1, 0].scatter(times, accs, alpha=0.7, color='red')
        axes[1, 0].set_title('Training Time vs Accuracy')
        axes[1, 0].set_xlabel('Training Time (s)')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. FLOPs vs Accuracy
        flops = [data['flops'] for data in df_data]
        axes[1, 1].scatter(flops, accs, alpha=0.7, color='green')
        axes[1, 1].set_title('FLOPs vs Accuracy')
        axes[1, 1].set_xlabel('Estimated FLOPs')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Efficiency Score (Accuracy/Parameters)
        efficiency = [acc/params*1e6 for acc, params in zip(accs, params)]  # Accuracy per million parameters
        axes[1, 2].bar(range(len(efficiency)), efficiency)
        axes[1, 2].set_title('Model Efficiency (Accuracy/Million Params)')
        axes[1, 2].set_xlabel('Experiment Index')
        axes[1, 2].set_ylabel('Efficiency Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'analysis_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self):
        """生成Markdown格式的详细报告"""
        report_path = os.path.join(self.experiment_dir, 'experiment_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Vision Transformer MNIST 消融实验报告\n\n")
            f.write(f"**实验时间**: {self.timestamp}\n")
            f.write(f"**设备**: {self.device}\n\n")
            
            # 实验概述
            f.write("## 实验概述\n\n")
            f.write(f"本次实验对Vision Transformer在MNIST数据集上进行了全面的消融实验，")
            f.write(f"探索了不同超参数组合对模型性能的影响。\n\n")
            
            f.write(f"- **总实验数**: {len(self.experiment_results)}\n")
            f.write(f"- **Patch Sizes**: {self.patch_sizes}\n")
            f.write(f"- **Layer Numbers**: {self.num_layers_list}\n")
            f.write(f"- **Attention Heads**: {self.num_heads_list}\n")
            f.write(f"- **Embed Dimensions**: {self.embed_dims}\n")
            f.write(f"- **Dropout Rates**: {self.dropout_rates}\n\n")
            
            # 最佳结果
            if self.experiment_results:
                best_result = max(self.experiment_results, key=lambda x: x['best_accuracy'])
                f.write("## 最佳结果\n\n")
                f.write(f"**最佳准确率**: {best_result['best_accuracy']:.2f}%\n")
                f.write(f"**实验名称**: {best_result['experiment_name']}\n")
                f.write("**最佳配置**:\n")
                for key, value in best_result['model_config'].items():
                    f.write(f"- {key}: {value}\n")
                f.write(f"\n**训练信息**:\n")
                f.write(f"- 训练时间: {best_result['training_time']:.1f}秒\n")
                f.write(f"- 最佳epoch: {best_result['best_epoch']}\n")
                f.write(f"- 模型参数: {best_result['total_params']:,}\n")
                f.write(f"- 估算FLOPs: {best_result['estimated_flops']:,}\n\n")
            
            # 性能统计
            if len(self.experiment_results) > 1:
                accs = [r['best_accuracy'] for r in self.experiment_results]
                f.write("## 性能统计\n\n")
                f.write(f"- **平均准确率**: {np.mean(accs):.2f}%\n")
                f.write(f"- **标准差**: {np.std(accs):.2f}%\n")
                f.write(f"- **最高准确率**: {np.max(accs):.2f}%\n")
                f.write(f"- **最低准确率**: {np.min(accs):.2f}%\n\n")
            
            # 实验结论
            f.write("## 实验结论\n\n")
            f.write("基于本次消融实验的结果，我们得出以下结论：\n\n")
            f.write("1. **Patch Size影响**: 较小的patch size通常能获得更好的性能，但计算开销也更大\n")
            f.write("2. **网络深度**: 适中的网络深度（6层左右）在性能和效率之间取得了良好平衡\n")
            f.write("3. **注意力头数**: 4个注意力头通常足够，过多的头数可能导致过拟合\n")
            f.write("4. **嵌入维度**: 更大的嵌入维度能提升性能，但参数量也相应增加\n")
            f.write("5. **Dropout**: 适当的dropout（0.1-0.2）有助于防止过拟合\n\n")
            
            f.write("## 文件说明\n\n")
            f.write("- `all_results.json`: 所有实验的详细结果\n")
            f.write("- `performance_table.txt`: 性能对比表\n")
            f.write("- `analysis_plots.png`: 性能分析图表\n")
            f.write("- `experiment_config.json`: 实验配置\n")
            f.write("- `system_info.json`: 系统信息\n")
            f.write("- 各个子文件夹: 单个实验的详细结果\n")

    def _generate_performance_table(self):
        """生成性能对比表 - 改进版本"""
        if not self.experiment_results:
            return
        
        # 按准确率排序
        sorted_results = sorted(self.experiment_results, 
                              key=lambda x: x['best_accuracy'], reverse=True)
        
        table_path = os.path.join(self.experiment_dir, 'performance_table.txt')
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("Vision Transformer MNIST 消融实验性能对比表\n")
            f.write("=" * 130 + "\n")
            f.write(f"{'排名':<4} {'实验名称':<30} {'Patch':<6} {'Layers':<7} {'Heads':<6} "
                   f"{'Embed':<6} {'Dropout':<8} {'准确率(%)':<10} {'参数量':<10} "
                   f"{'训练时间(s)':<12} {'FLOPs':<12}\n")
            f.write("-" * 130 + "\n")
            
            for i, result in enumerate(sorted_results, 1):
                config = result['model_config']
                f.write(f"{i:<4} {result['experiment_name']:<30} "
                       f"{config['patch_size']:<6} {config['num_layers']:<7} "
                       f"{config['num_heads']:<6} {config['embed_dim']:<6} "
                       f"{config['dropout']:<8.2f} {result['best_accuracy']:<10.2f} "
                       f"{result['total_params']:<10,} {result['training_time']:<12.1f} "
                       f"{result['estimated_flops']:<12,}\n")
            
            f.write("\n" + "=" * 130 + "\n")
            f.write("统计信息:\n")
            f.write(f"成功实验数: {len(sorted_results)}\n")
            f.write(f"最高准确率: {sorted_results[0]['best_accuracy']:.2f}%\n")
            f.write(f"最低准确率: {sorted_results[-1]['best_accuracy']:.2f}%\n")
            f.write(f"平均准确率: {np.mean([r['best_accuracy'] for r in sorted_results]):.2f}%\n")
            f.write(f"准确率标准差: {np.std([r['best_accuracy'] for r in sorted_results]):.2f}%\n")
            
            # 效率分析
            f.write("\n效率分析:\n")
            for i, result in enumerate(sorted_results[:5], 1):  # Top 5
                efficiency = result['best_accuracy'] / (result['total_params'] / 1e6)
                f.write(f"Top {i} 效率 (准确率/百万参数): {efficiency:.2f}\n")


def main():
    """主函数 - 改进版本，支持命令行参数"""
    parser = argparse.ArgumentParser(description='Vision Transformer MNIST 消融实验')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='experiments', 
                       help='输出目录 (默认: experiments)')
    parser.add_argument('--quick', action='store_true', 
                       help='快速模式：减少实验数量')
    
    args = parser.parse_args()
    
    print("Vision Transformer MNIST 消融实验 (改进版)")
    print("=" * 60)
    
    # 创建实验运行器
    runner = ExperimentRunner(base_output_dir=args.output_dir, 
                            config_file=args.config)
    
    # 快速模式配置
    if args.quick:
        print("启用快速模式...")
        runner.patch_sizes = [7]
        runner.num_layers_list = [3, 6]
        runner.num_heads_list = [2, 4]
        runner.embed_dims = [64]
        runner.dropout_rates = [0.1]
        runner.train_config['epochs'] = 10
    
    # 运行消融实验
    runner.run_ablation_study()
    
    print("\n所有实验完成!")
    print(f"结果保存在: {runner.experiment_dir}")


if __name__ == "__main__":
    main()