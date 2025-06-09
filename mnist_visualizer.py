#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST数据集解析和可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import gzip
import os
from pathlib import Path

class MNISTLoader:
    """MNIST数据集加载器"""
    
    def __init__(self, data_path="data/MNIST/raw"):
        self.data_path = Path(data_path)
        
    def load_images(self, filename):
        """加载MNIST图像文件"""
        filepath = self.data_path / filename
        
        # 如果是压缩文件，先解压
        if filename.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()
        
        # 解析IDX文件格式
        # 前16字节是头部信息
        magic, num_images, rows, cols = struct.unpack('>IIII', data[:16])
        
        # 验证魔数
        if magic != 0x00000803:
            raise ValueError(f"无效的图像文件魔数: {magic}")
        
        # 读取图像数据
        images = np.frombuffer(data[16:], dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
        print(f"加载图像: {filename}")
        print(f"  - 图像数量: {num_images}")
        print(f"  - 图像尺寸: {rows}x{cols}")
        print(f"  - 数据形状: {images.shape}")
        
        return images
    
    def load_labels(self, filename):
        """加载MNIST标签文件"""
        filepath = self.data_path / filename
        
        # 如果是压缩文件，先解压
        if filename.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()
        
        # 解析IDX文件格式
        # 前8字节是头部信息
        magic, num_labels = struct.unpack('>II', data[:8])
        
        # 验证魔数
        if magic != 0x00000801:
            raise ValueError(f"无效的标签文件魔数: {magic}")
        
        # 读取标签数据
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        
        print(f"加载标签: {filename}")
        print(f"  - 标签数量: {num_labels}")
        print(f"  - 标签范围: {labels.min()} - {labels.max()}")
        
        return labels
    
    def load_dataset(self, train=True):
        """加载完整的训练集或测试集"""
        if train:
            images_file = "train-images-idx3-ubyte"
            labels_file = "train-labels-idx1-ubyte"
        else:
            images_file = "t10k-images-idx3-ubyte"
            labels_file = "t10k-labels-idx1-ubyte"
        
        # 尝试加载未压缩文件，如果不存在则加载压缩文件
        try:
            images = self.load_images(images_file)
        except FileNotFoundError:
            images = self.load_images(images_file + ".gz")
        
        try:
            labels = self.load_labels(labels_file)
        except FileNotFoundError:
            labels = self.load_labels(labels_file + ".gz")
        
        return images, labels

STAT_DIR = "stat"
os.makedirs(STAT_DIR, exist_ok=True)

def analyze_dataset(images, labels, dataset_name="Dataset", text_file=None):
    """分析数据集统计信息，并写入文本文件"""
    lines = []
    lines.append(f"\n=== {dataset_name} Analysis ===")
    lines.append(f"Number of images: {len(images)}")
    lines.append(f"Image size: {images.shape[1]}x{images.shape[2]}")
    lines.append(f"Pixel value range: {images.min()} - {images.max()}")
    lines.append(f"Number of labels: {len(labels)}")
    lines.append(f"Number of classes: {len(np.unique(labels))}")
    
    # 统计每个类别的样本数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    lines.append("\nSamples per class:")
    for label, count in zip(unique_labels, counts):
        lines.append(f"  Digit {label}: {count} samples")
    
    # 输出到终端
    for l in lines:
        print(l)
    # 写入文本文件
    if text_file is not None:
        with open(text_file, 'a', encoding='utf-8') as f:
            for l in lines:
                f.write(l + '\n')
    
    # 可视化类别分布
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'{dataset_name} Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(unique_labels)
    plt.grid(axis='y', alpha=0.3)
    for i, count in enumerate(counts):
        plt.text(i, count + 50, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(STAT_DIR, f'{dataset_name.lower().replace(" ", "_")}_class_distribution.png'))
    plt.close()

def visualize_samples(images, labels, num_samples=20, title="MNIST Samples", filename=None):
    """可视化MNIST样例并保存图片"""
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    indices = np.random.choice(len(images), num_samples, replace=False)
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(images[idx], cmap='gray')
        axes[row, col].set_title(f'Label: {labels[idx]}', fontsize=12)
        axes[row, col].axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(STAT_DIR, filename))
    plt.close()

def visualize_pixel_distribution(images, title="Pixel Value Distribution", filename=None):
    """可视化像素值分布并保存图片"""
    plt.figure(figsize=(10, 6))
    pixel_values = images.flatten()
    plt.hist(pixel_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Pixel Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'Mean+Std: {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'Mean-Std: {mean_val - std_val:.2f}')
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(STAT_DIR, filename))
    plt.close()

def main():
    print("MNIST Dataset Analysis and Visualization")
    print("=" * 50)
    loader = MNISTLoader()
    analysis_txt = os.path.join(STAT_DIR, 'analysis.txt')
    # 清空分析文本
    with open(analysis_txt, 'w', encoding='utf-8') as f:
        f.write('')
    try:
        print("\nLoading training set...")
        train_images, train_labels = loader.load_dataset(train=True)
        print("\nLoading test set...")
        test_images, test_labels = loader.load_dataset(train=False)
        analyze_dataset(train_images, train_labels, "Training Set", text_file=analysis_txt)
        analyze_dataset(test_images, test_labels, "Test Set", text_file=analysis_txt)
        print("\nGenerating visualizations...")
        visualize_samples(train_images, train_labels, title="Training Set Samples", filename="training_set_samples.png")
        visualize_samples(test_images, test_labels, title="Test Set Samples", filename="test_set_samples.png")
        visualize_pixel_distribution(train_images, "Training Set Pixel Distribution", filename="training_set_pixel_distribution.png")
        print("\nShowing examples of each digit...")
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Examples of Each Digit', fontsize=16, fontweight='bold')
        for digit in range(10):
            idx = np.where(train_labels == digit)[0][0]
            row = digit // 5
            col = digit % 5
            axes[row, col].imshow(train_images[idx], cmap='gray')
            axes[row, col].set_title(f'Digit {digit}', fontsize=12)
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(STAT_DIR, 'examples_of_each_digit.png'))
        plt.close()
        print("\nDataset analysis and visualization completed!")
        print(f"All images and analysis text are saved in the '{STAT_DIR}' folder.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 