import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from vit_model import create_vit_model
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime
import glob

# 设置字体，避免中文显示警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_model(model_path, device):
    """
    加载训练好的模型
    """
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = create_vit_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_single_image(model, image_tensor, device):
    """
    对单张图像进行预测
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加batch维度
        output, attention_maps = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence, attention_maps


def visualize_prediction(image, predicted_class, confidence, attention_maps=None, true_label=None):
    """
    可视化预测结果
    """
    if attention_maps is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]
    
    # 显示原始图像
    axes[0].imshow(image.squeeze(), cmap='gray')
    title = f'Prediction: {predicted_class} (Confidence: {confidence:.2f})'
    if true_label is not None:
        title += f'\nTrue Label: {true_label}'
    axes[0].set_title(title)
    axes[0].axis('off')
    
    if attention_maps is not None and len(axes) >= 3:
        # 显示注意力图
        last_attention = attention_maps[-1][0].cpu()  # 最后一层的注意力
        
        # 平均所有注意力头
        avg_attention = last_attention.mean(dim=0)  # (seq_len, seq_len)
        cls_attention = avg_attention[0, 1:].reshape(4, 4)  # CLS token对patches的注意力
        
        im = axes[1].imshow(cls_attention, cmap='hot')
        axes[1].set_title('Average Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加显示
        # 将注意力图插值到原图尺寸
        attention_resized = F.interpolate(
            cls_attention.clone().detach().unsqueeze(0).unsqueeze(0), 
            size=(28, 28), mode='bilinear'
        ).squeeze().numpy()
        
        axes[2].imshow(image.squeeze(), cmap='gray', alpha=0.7)
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def evaluate_model_comprehensive(model, test_loader, device):
    """
    全面评估模型性能
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("正在评估模型...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # 计算准确率
    accuracy = (all_predictions == all_targets).mean() * 100
    
    # 分类报告
    class_report = classification_report(
        all_targets, all_predictions, 
        target_names=[str(i) for i in range(10)],
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    return accuracy, class_report, cm, all_probabilities


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")


def analyze_errors(model, test_loader, device, num_errors=10):
    """
    分析错误预测的样本
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, attention_maps = model(data)
            predictions = output.argmax(dim=1)
            
            # 找到错误预测
            for i in range(len(data)):
                if predictions[i] != target[i]:
                    probabilities = F.softmax(output[i], dim=0)
                    errors.append({
                        'image': data[i].cpu(),
                        'true_label': target[i].item(),
                        'predicted_label': predictions[i].item(),
                        'confidence': probabilities[predictions[i]].item(),
                        'attention_maps': [att[i:i+1] for att in attention_maps]
                    })
                    
                    if len(errors) >= num_errors:
                        break
            
            if len(errors) >= num_errors:
                break
    
    return errors


def visualize_error_analysis(errors, save_path='error_analysis.png'):
    """
    可视化错误分析
    """
    num_errors = len(errors)
    if num_errors == 0:
        print("没有找到错误预测的样本！")
        return
    
    cols = min(5, num_errors)
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 8))
    if cols == 1:
        axes = np.array(axes).reshape(2, 1)
    
    for i, error in enumerate(errors[:cols]):  # 最多显示5个错误
        # 原始图像
        axes[0, i].imshow(error['image'].squeeze(), cmap='gray')
        axes[0, i].set_title(
            f'True: {error["true_label"]}\nPred: {error["predicted_label"]}\nConf: {error["confidence"]:.2f}'
        )
        axes[0, i].axis('off')
        
        # 注意力图
        if error['attention_maps']:
            last_attention = error['attention_maps'][-1][0].cpu()
            avg_attention = last_attention.mean(dim=0)
            cls_attention = avg_attention[0, 1:].reshape(4, 4)
            im = axes[1, i].imshow(cls_attention, cmap='hot')
            axes[1, i].set_title('Attention Map')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i])
        else:
            axes[1, i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_errors, 5):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"错误分析已保存到: {save_path}")


def find_latest_model():
    """
    查找最新的训练模型
    """
    # 查找train文件夹下的所有训练目录
    train_dirs = glob.glob("train/train_*")
    if not train_dirs:
        return None
    
    # 按时间戳排序，获取最新的
    latest_dir = sorted(train_dirs)[-1]
    model_path = os.path.join(latest_dir, 'best_vit_model.pth')
    
    if os.path.exists(model_path):
        return model_path, latest_dir
    else:
        return None


def main():
    # 创建测试输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = f"test/test_{timestamp}"
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"测试输出将保存到: {test_output_dir}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    # 查找并加载模型
    model_info = find_latest_model()
    if model_info is None:
        print("未找到训练好的模型，请先运行 python train.py 训练模型。")
        return
    
    model_path, train_dir = model_info
    try:
        model, checkpoint = load_model(model_path, device)
        print(f"成功加载模型: {model_path}")
        print(f"模型最佳准确率: {checkpoint['best_acc']:.2f}%")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 全面评估
    accuracy, class_report, cm, probabilities = evaluate_model_comprehensive(
        model, test_loader, device
    )
    
    print(f"\n模型评估结果:")
    print(f"总体准确率: {accuracy:.2f}%")
    print(f"\n各类别性能:")
    for class_id in range(10):
        precision = class_report[str(class_id)]['precision']
        recall = class_report[str(class_id)]['recall']
        f1 = class_report[str(class_id)]['f1-score']
        print(f"数字 {class_id}: 精确率={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}")
    
    # 绘制混淆矩阵
    confusion_matrix_path = os.path.join(test_output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, confusion_matrix_path)
    
    # 错误分析
    print("\n正在进行错误分析...")
    errors = analyze_errors(model, test_loader, device, num_errors=10)
    print(f"找到 {len(errors)} 个错误预测样本")
    
    if errors:
        error_analysis_path = os.path.join(test_output_dir, 'error_analysis.png')
        visualize_error_analysis(errors, error_analysis_path)
    
    # 随机预测几张图片
    print("\n随机预测示例:")
    random_indices = np.random.choice(len(test_dataset), 5, replace=False)
    
    for i, idx in enumerate(random_indices):
        image, true_label = test_dataset[idx]
        predicted_class, confidence, attention_maps = predict_single_image(
            model, image, device
        )
        
        print(f"样本 {i+1}: 真实标签={true_label}, 预测={predicted_class}, 置信度={confidence:.3f}")
        
        # 可视化第一个样本
        if i == 0:
            fig = visualize_prediction(
                image, predicted_class, confidence, attention_maps, true_label
            )
            prediction_example_path = os.path.join(test_output_dir, 'random_prediction_example.png')
            plt.savefig(prediction_example_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"随机预测示例已保存到: {prediction_example_path}")
    
    # 保存测试结果摘要
    summary_path = os.path.join(test_output_dir, 'test_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Vision Transformer Test Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"使用模型: {model_path}\n")
        f.write(f"训练来源: {train_dir}\n")
        f.write(f"设备: {device}\n")
        f.write(f"总体准确率: {accuracy:.2f}%\n")
        f.write(f"错误样本数: {len(errors)}\n")
        f.write(f"\n各类别性能:\n")
        for class_id in range(10):
            precision = class_report[str(class_id)]['precision']
            recall = class_report[str(class_id)]['recall']
            f1 = class_report[str(class_id)]['f1-score']
            f.write(f"  数字 {class_id}: 精确率={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}\n")
    
    print(f"\n推理分析完成！所有文件已保存到: {test_output_dir}")
    print(f"生成的文件:")
    print(f"  - confusion_matrix.png: 混淆矩阵")
    print(f"  - error_analysis.png: 错误分析")
    print(f"  - random_prediction_example.png: 随机预测示例")
    print(f"  - test_summary.txt: 测试摘要")


if __name__ == '__main__':
    main() 