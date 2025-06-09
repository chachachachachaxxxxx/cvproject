# Vision Transformer 注意力机制可视化分析报告

## 分析概述

- **模型配置**: 最优ViT模型 (embed_dim=128, num_heads=4, num_layers=6)
- **分析样本数**: 8
- **设备**: cuda
- **输出目录**: best_model_attention_analysis_efficient_ablation_20250609_101146

## 生成的可视化文件

### 1. 注意力模式分析
- `attention_pattern_sample_*.png`: 每个样本的层级注意力热力图
- 显示了CLS token对图像patches的注意力分布
- 包含原图叠加和纯注意力热力图

### 2. 注意力头专业化分析
- `attention_heads_sample_*.png`: 不同注意力头的注意力模式
- 揭示了多头注意力的分工机制
- 展示了每个头关注的不同特征区域

### 3. 层级注意力演化
- `layer_comparison_sample_*.png`: 跨层注意力模式对比
- 显示了从浅层到深层的注意力演化过程
- 体现了分层特征提取的过程

### 4. 统计分析
- `attention_statistics.png`: 注意力模式的统计特征
- 包含最大注意力、平均注意力和注意力熵的分布
- 帮助理解模型的注意力集中程度和分散程度

## 主要发现

1. **层级特征提取**: 浅层关注局部边缘特征，深层关注全局结构
2. **注意力头专业化**: 不同头部专注于不同类型的特征
3. **数字特征识别**: 模型能够自动关注数字的关键笔画区域
4. **注意力集中度**: 随着层数增加，注意力逐渐集中到关键区域

## 使用方法

```python
# 初始化可视化器
visualizer = AttentionVisualizer('path/to/best_model.pth')

# 运行完整分析
visualizer.run_comprehensive_analysis('path/to/best_model.pth')
```
