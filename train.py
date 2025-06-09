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
from datetime import datetime
from vit_model import create_vit_model

# 设置字体，避免中文显示警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def get_data_loaders(batch_size=64, num_workers=4):
    """
    创建MNIST数据加载器
    """
    # 数据预处理和增强
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=10),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def visualize_attention(model, test_loader, device, save_path='attention_visualization.png'):
    """
    可视化注意力图
    """
    model.eval()
    
    # 获取一个批次的数据
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, attention_maps = model(data)
            break
    
    # 选择第一个样本进行可视化
    sample_image = data[0].cpu().squeeze()  # (28, 28)
    sample_attention = attention_maps[-1][0].cpu()  # 最后一层的注意力图
    
    # 获取注意力头数量
    num_heads = sample_attention.shape[0]
    
    # 动态创建子图布局
    if num_heads <= 3:
        fig, axes = plt.subplots(1, num_heads + 1, figsize=(5 * (num_heads + 1), 5))
        if num_heads == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
    else:
        # 如果注意力头数量较多，使用2行布局
        cols = min(4, num_heads + 1)
        rows = 2 if num_heads > 3 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
    
    # 将axes转为二维数组便于索引
    axes = np.array(axes).reshape(-1) if axes.ndim == 1 else axes
    
    # 原始图像
    ax_idx = 0
    if axes.ndim == 2:
        axes[0, 0].imshow(sample_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        ax_idx = 1
    else:
        axes[0].imshow(sample_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        ax_idx = 1
    
    # 可视化注意力头
    for i in range(min(num_heads, 5)):  # 最多显示5个注意力头
        # 获取CLS token对其他token的注意力
        cls_attention = sample_attention[i, 0, 1:].reshape(4, 4)  # 假设有16个patches (4x4)
        
        if axes.ndim == 2:
            # 2D layout
            row = (ax_idx + i) // axes.shape[1]
            col = (ax_idx + i) % axes.shape[1]
            if row < axes.shape[0] and col < axes.shape[1]:
                im = axes[row, col].imshow(cls_attention, cmap='hot')
                axes[row, col].set_title(f'Attention Head {i+1}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])
        else:
            # 1D layout
            if ax_idx + i < len(axes):
                im = axes[ax_idx + i].imshow(cls_attention, cmap='hot')
                axes[ax_idx + i].set_title(f'Attention Head {i+1}')
                axes[ax_idx + i].axis('off')
                plt.colorbar(im, ax=axes[ax_idx + i])
    
    # 隐藏未使用的子图
    total_plots = min(num_heads + 1, 6)  # 原图 + 最多5个注意力头
    if axes.ndim == 2:
        for idx in range(total_plots, axes.size):
            row = idx // axes.shape[1]
            col = idx % axes.shape[1]
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
    else:
        for idx in range(total_plots, len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"注意力可视化已保存到: {save_path}")


def plot_training_curves(train_losses, train_accs, test_losses, test_accs, save_path='training_curves.png'):
    """
    绘制训练曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(test_accs, label='Test Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_output_dir = f"train/train_{timestamp}"
    os.makedirs(train_output_dir, exist_ok=True)
    print(f"训练输出将保存到: {train_output_dir}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 超参数
    config = {
        'batch_size': 64,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'epochs': 100,
        'num_workers': 4 if device.type == 'cuda' else 0
    }
    
    # 模型配置
    model_config = {
        'img_size': 28,
        'patch_size': 7,
        'in_channels': 1,
        'num_classes': 10,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 6,
        'mlp_dim': 128,
        'dropout': 0.1
    }
    
    # 创建数据加载器
    print("正在加载数据...")
    train_loader, test_loader = get_data_loaders(
        batch_size=config['batch_size'], 
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("正在创建模型...")
    model = create_vit_model(model_config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    
    # 训练历史
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_acc = 0
    
    print("开始训练...")
    for epoch in range(1, config['epochs'] + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印结果
        print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(train_output_dir, 'best_vit_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_config': model_config
            }, best_model_path)
            print(f'保存最佳模型，测试准确率: {best_acc:.2f}%')
    
    print(f'\n训练完成！最佳测试准确率: {best_acc:.2f}%')
    
    # 绘制训练曲线
    training_curves_path = os.path.join(train_output_dir, 'training_curves.png')
    plot_training_curves(train_losses, train_accs, test_losses, test_accs, training_curves_path)
    
    # 可视化注意力
    print("正在生成注意力可视化...")
    attention_viz_path = os.path.join(train_output_dir, 'attention_visualization.png')
    visualize_attention(model, test_loader, device, attention_viz_path)
    
    # 保存最终模型
    final_model_path = os.path.join(train_output_dir, 'final_vit_model.pth')
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_acc': test_accs[-1],
        'model_config': model_config,
        'train_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs
        }
    }, final_model_path)
    
    # 保存训练配置和结果摘要
    summary_path = os.path.join(train_output_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Vision Transformer Training Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"设备: {device}\n")
        f.write(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"训练轮数: {config['epochs']}\n")
        f.write(f"批量大小: {config['batch_size']}\n")
        f.write(f"学习率: {config['learning_rate']}\n")
        f.write(f"权重衰减: {config['weight_decay']}\n")
        f.write(f"最佳测试准确率: {best_acc:.2f}%\n")
        f.write(f"最终测试准确率: {test_accs[-1]:.2f}%\n")
        f.write(f"\n模型配置:\n")
        for key, value in model_config.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"训练完成！所有文件已保存到: {train_output_dir}")
    print(f"生成的文件:")
    print(f"  - best_vit_model.pth: 最佳模型权重")
    print(f"  - final_vit_model.pth: 最终模型权重")
    print(f"  - training_curves.png: 训练曲线图")
    print(f"  - attention_visualization.png: 注意力可视化")
    print(f"  - training_summary.txt: 训练摘要")


if __name__ == '__main__':
    main() 