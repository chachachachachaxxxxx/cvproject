import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
import json
import time
import platform
import psutil
from datetime import datetime
from collections import defaultdict
import pandas as pd
import sys
sys.path.append('../')
from vit_model import create_vit_model


class EfficientAblationRunner:
    """高效消融实验运行器"""
    
    def __init__(self, config_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 加载实验配置
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 创建实验目录
        self.experiment_dir = os.path.join("experiments", f"efficient_ablation_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 存储结果
        self.experiment_results = []
        self.group_results = defaultdict(list)
        
        print(f"实验配置: {self.config['experiment_name']}")
        print(f"实验目录: {self.experiment_dir}")
        print(f"预计实验数量: {self.config['estimated_experiments']['total']}")
        
    def get_data_loaders(self, batch_size):
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
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True
        )
        
        return train_loader, test_loader
    
    def create_model_config(self, experiment_config):
        """根据实验配置创建模型配置"""
        model_config = self.config['base_config'].copy()
        
        # 计算 MLP 维度
        mlp_ratio = self.config['base_config'].get('mlp_ratio', 2)
        mlp_dim = experiment_config['embed_dim'] * mlp_ratio
        
        # 更新模型参数
        model_config.update({
            'patch_size': experiment_config['patch_size'],
            'num_layers': experiment_config['num_layers'],
            'num_heads': experiment_config['num_heads'],
            'embed_dim': experiment_config['embed_dim'],
            'mlp_dim': mlp_dim,
            'dropout': experiment_config['dropout']
        })
        
        # 移除不需要的参数
        if 'mlp_ratio' in model_config:
            del model_config['mlp_ratio']
        
        return model_config
    
    def train_model(self, experiment_config, experiment_name, group_name):
        """训练单个模型"""
        print(f"\n开始实验: {experiment_name} (组: {group_name})")
        print(f"配置: {experiment_config}")
        
        # 创建实验子目录
        exp_dir = os.path.join(self.experiment_dir, group_name, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        try:
            # 创建模型配置
            model_config = self.create_model_config(experiment_config)
            model = create_vit_model(model_config).to(self.device)
            
            # 获取数据加载器
            batch_size = experiment_config['batch_size']
            train_loader, test_loader = self.get_data_loaders(batch_size)
            
            # 训练配置
            train_config = self.config['train_config']
            
            # 处理学习率设置
            if 'learning_rate' in experiment_config:
                # 如果实验配置中指定了学习率，直接使用（学习率消融实验）
                adjusted_lr = experiment_config['learning_rate']
                lr_source = "experiment_config"
                print(f"使用实验指定学习率: {adjusted_lr:.6f}")
            else:
                # 否则根据batch size调整学习率
                base_lr = train_config['learning_rate']
                base_batch_size = train_config.get('base_batch_size', 64)
                lr_scaling_factor = train_config.get('lr_scaling_factor', 0.5)
                
                # 线性缩放学习率，但加上上限保护
                adjusted_lr = base_lr * (batch_size / base_batch_size) ** lr_scaling_factor
                max_lr = train_config.get('max_learning_rate', base_lr * 4)
                adjusted_lr = min(adjusted_lr, max_lr)
                lr_source = "batch_size_scaling"
                print(f"Batch Size: {batch_size}, 基础学习率: {base_lr:.6f}, 调整后学习率: {adjusted_lr:.6f}")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=adjusted_lr, 
                weight_decay=train_config['weight_decay']
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_config['epochs'], eta_min=1e-6
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
            
            # 训练循环
            for epoch in range(1, train_config['epochs'] + 1):
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
                
                # 早停检查
                if test_acc > best_acc + train_config['min_delta']:
                    best_acc = test_acc
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # 保存最佳模型
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': model_config,
                        'best_acc': best_acc
                    }, os.path.join(exp_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                
                if patience_counter >= train_config['early_stopping_patience']:
                    print(f"早停在epoch {epoch}, 最佳准确率: {best_acc:.2f}%")
                    break
            
            training_time = time.time() - start_time
            
            # 计算模型统计信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            flops = self._estimate_flops(model_config)
            
            # 保存实验结果
            result = {
                'experiment_name': experiment_name,
                'group_name': group_name,
                'experiment_config': experiment_config,
                'model_config': model_config,
                'training_config': {
                    'base_learning_rate': train_config['learning_rate'],
                    'adjusted_learning_rate': adjusted_lr,
                    'batch_size': batch_size,
                    'lr_source': lr_source,
                    'lr_scaling_applied': lr_source == "batch_size_scaling"
                },
                'best_accuracy': best_acc,
                'best_epoch': best_epoch,
                'final_accuracy': test_accs[-1] if test_accs else 0,
                'training_time': training_time,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'estimated_flops': flops,
                'parameter_efficiency': best_acc / total_params * 1e6,  # Accuracy per million parameters
                'computational_efficiency': best_acc / flops * 1e9,  # Accuracy per GFLOP
                'train_history': {
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'test_losses': test_losses,
                    'test_accs': test_accs
                }
            }
            
            self.experiment_results.append(result)
            self.group_results[group_name].append(result)
            
            # 保存单个实验结果
            with open(os.path.join(exp_dir, 'result.json'), 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"实验 {experiment_name} 完成!")
            print(f"最佳准确率: {best_acc:.2f}% (epoch {best_epoch})")
            print(f"训练时间: {training_time:.1f}秒")
            print(f"模型参数: {total_params:,}")
            print(f"参数效率: {result['parameter_efficiency']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"实验 {experiment_name} 失败: {str(e)}")
            return None
    
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
    
    def _estimate_flops(self, model_config):
        """估算模型FLOPs"""
        # 简化的FLOPs估算
        embed_dim = model_config['embed_dim']
        num_layers = model_config['num_layers']
        patch_size = model_config['patch_size']
        img_size = model_config['img_size']
        
        num_patches = (img_size // patch_size) ** 2
        seq_len = num_patches + 1  # +1 for CLS token
        
        # 主要计算：注意力机制和MLP
        attention_flops = num_layers * seq_len * seq_len * embed_dim
        mlp_flops = num_layers * seq_len * embed_dim * model_config['mlp_dim'] * 2
        
        return attention_flops + mlp_flops
    
    def run_efficient_ablation(self):
        """运行高效消融实验"""
        print("\n开始高效消融实验...")
        
        experiment_groups = self.config['experiment_groups']
        total_count = 0
        successful_count = 0
        
        for group_name, group_config in experiment_groups.items():
            print(f"\n=== 实验组: {group_name} ===")
            print(f"描述: {group_config['description']}")
            
            if 'experiments' in group_config:
                # 直接定义的实验列表
                for i, exp_config in enumerate(group_config['experiments']):
                    total_count += 1
                    exp_name = exp_config.get('name', f"{group_name}_{i+1}")
                    
                    result = self.train_model(exp_config, exp_name, group_name)
                    if result is not None:
                        successful_count += 1
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            elif 'variable' in group_config:
                # 单因子消融实验
                base_config = group_config['base_config']
                variable = group_config['variable']
                values = group_config['values']
                
                for value in values:
                    total_count += 1
                    exp_config = base_config.copy()
                    exp_config[variable] = value
                    exp_name = f"{group_name}_{variable}_{value}"
                    
                    result = self.train_model(exp_config, exp_name, group_name)
                    if result is not None:
                        successful_count += 1
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print(f"\n实验完成统计:")
        print(f"总实验数: {total_count}")
        print(f"成功实验数: {successful_count}")
        print(f"失败实验数: {total_count - successful_count}")
        
        # 保存实验结果
        if self.experiment_results:
            self._save_results()
        else:
            print("没有成功的实验结果")
    
    def _save_results(self):
        """保存实验结果和基本分析"""
        print("\n保存实验结果...")
        
        # 保存所有实验结果
        with open(os.path.join(self.experiment_dir, 'all_results.json'), 'w') as f:
            json.dump(self.experiment_results, f, indent=2, ensure_ascii=False)
        
        # 生成基本分析数据
        self._create_analysis_data()
        
        # 生成文本报告
        self._generate_text_report()
        
        print(f"实验结果已保存到: {self.experiment_dir}")
        print("使用 experiment_visualizer.py 生成可视化图表")
    
    def _create_analysis_data(self):
        """创建分析数据文件"""
        if not self.experiment_results:
            return
        
        # 创建DataFrame用于分析
        data = []
        for result in self.experiment_results:
            config = result['experiment_config']
            data.append({
                'experiment_name': result['experiment_name'],
                'group_name': result['group_name'],
                'patch_size': config['patch_size'],
                'num_layers': config['num_layers'],
                'num_heads': config['num_heads'],
                'embed_dim': config['embed_dim'],
                'batch_size': config['batch_size'],
                'dropout': config['dropout'],
                'accuracy': result['best_accuracy'],
                'params': result['total_params'],
                'time': result['training_time'],
                'flops': result['estimated_flops'],
                'param_efficiency': result['parameter_efficiency'],
                'comp_efficiency': result['computational_efficiency']
            })
        
        df = pd.DataFrame(data)
        
        # 保存DataFrame
        df.to_csv(os.path.join(self.experiment_dir, 'experiment_data.csv'), index=False)
        
        # 分析单因子效应
        factor_effects = {}
        ablation_groups = ['patch_size_ablation', 'depth_ablation', 'heads_ablation', 
                          'embed_dim_ablation', 'batch_size_ablation', 'learning_rate_ablation']
        
        for group_name in ablation_groups:
            if group_name in self.group_results:
                results = self.group_results[group_name]
                if len(results) > 1:
                    variable = group_name.replace('_ablation', '')
                    if variable == 'depth':
                        variable = 'num_layers'
                    elif variable == 'heads':
                        variable = 'num_heads'
                    
                    values = []
                    accuracies = []
                    
                    for result in results:
                        config = result['experiment_config']
                        if variable in config:
                            values.append(config[variable])
                            accuracies.append(result['best_accuracy'])
                        elif variable == 'learning_rate':
                            lr_config = result.get('training_config', {})
                            lr = lr_config.get('adjusted_learning_rate', 0)
                            if lr > 0:
                                values.append(lr)
                                accuracies.append(result['best_accuracy'])
                    
                    if values and accuracies:
                        effect_size = max(accuracies) - min(accuracies)
                        best_value = values[accuracies.index(max(accuracies))]
                        
                        factor_effects[variable] = {
                            'effect_size': effect_size,
                            'best_value': best_value,
                            'all_values': values,
                            'all_accuracies': accuracies
                        }
        
        # 保存因子效应分析
        with open(os.path.join(self.experiment_dir, 'factor_effects.json'), 'w') as f:
            json.dump(factor_effects, f, indent=2)
    
    def _generate_text_report(self):
        """生成文本报告"""
        report_path = os.path.join(self.experiment_dir, 'experiment_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Vision Transformer MNIST Efficient Ablation Study Report\n\n")
            f.write(f"**Experiment Time**: {self.timestamp}\n")
            f.write(f"**Device**: {self.device}\n")
            f.write(f"**Strategy**: Efficient ablation design\n\n")
            
            # 实验概览
            f.write("## Experiment Overview\n\n")
            f.write(f"- **Total Experiments**: {len(self.experiment_results)}\n")
            f.write(f"- **Experiment Groups**: {len(self.group_results)}\n")
            
            # 最佳结果
            if self.experiment_results:
                best_result = max(self.experiment_results, key=lambda x: x['best_accuracy'])
                f.write("\n## Best Result\n\n")
                f.write(f"- **Experiment Name**: {best_result['experiment_name']}\n")
                f.write(f"- **Group**: {best_result['group_name']}\n")
                f.write(f"- **Best Accuracy**: {best_result['best_accuracy']:.2f}%\n")
                f.write(f"- **Parameters**: {best_result['total_params']:,}\n")
                f.write(f"- **Training Time**: {best_result['training_time']:.1f}s\n")
                f.write(f"- **Parameter Efficiency**: {best_result['parameter_efficiency']:.3f}\n\n")
                
                # 最佳配置
                f.write("### Best Configuration\n\n")
                config = best_result['experiment_config']
                f.write("```json\n")
                f.write(json.dumps(config, indent=2))
                f.write("\n```\n\n")
            
            # 分组结果
            f.write("## Group Results\n\n")
            for group_name, results in self.group_results.items():
                if results:
                    f.write(f"### {group_name}\n\n")
                    f.write(f"Number of experiments: {len(results)}\n\n")
                    
                    # 创建结果表格
                    f.write("| Experiment | Accuracy(%) | Parameters | Time(s) | Learning Rate | Param Efficiency |\n")
                    f.write("|------------|-------------|------------|---------|---------------|------------------|\n")
                    
                    for result in sorted(results, key=lambda x: x['best_accuracy'], reverse=True):
                        lr_info = result.get('training_config', {}).get('adjusted_learning_rate', 'N/A')
                        lr_str = f"{lr_info:.6f}" if isinstance(lr_info, float) else str(lr_info)
                        f.write(f"| {result['experiment_name']} | "
                               f"{result['best_accuracy']:.2f} | "
                               f"{result['total_params']:,} | "
                               f"{result['training_time']:.1f} | "
                               f"{lr_str} | "
                               f"{result['parameter_efficiency']:.3f} |\n")
                    f.write("\n")
            
            # 推荐配置
            f.write("## Recommendations\n\n")
            if self.experiment_results:
                best_acc = max(self.experiment_results, key=lambda x: x['best_accuracy'])
                best_eff = max(self.experiment_results, key=lambda x: x['parameter_efficiency'])
                fastest = min(self.experiment_results, key=lambda x: x['training_time'])
                
                f.write("Based on different objectives:\n\n")
                f.write(f"- **Highest Accuracy**: {best_acc['experiment_name']} ({best_acc['best_accuracy']:.2f}%)\n")
                f.write(f"- **Best Parameter Efficiency**: {best_eff['experiment_name']} ({best_eff['parameter_efficiency']:.3f})\n")
                f.write(f"- **Fastest Training**: {fastest['experiment_name']} ({fastest['training_time']:.1f}s)\n\n")
                
                f.write("## Visualization\n\n")
                f.write("To generate visualizations, run:\n")
                f.write("```bash\n")
                f.write(f"python experiment_visualizer.py --experiment-dir {self.experiment_dir}\n")
                f.write("```\n")


def main():
    """主函数"""
    config_file = "efficient_ablation_config.json"
    
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在！")
        return
    
    print("Vision Transformer MNIST 高效消融实验")
    print("=" * 50)
    
    # 创建实验运行器
    runner = EfficientAblationRunner(config_file)
    
    # 运行实验
    runner.run_efficient_ablation()
    
    print("\n所有实验完成!")
    print(f"结果保存在: {runner.experiment_dir}")


if __name__ == "__main__":
    main()