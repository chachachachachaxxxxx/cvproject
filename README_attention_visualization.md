# Vision Transformer 注意力机制可视化工具

本工具专门用于可视化和分析Vision Transformer在MNIST数据集上的注意力机制，帮助理解模型的决策过程和特征提取模式。

## 🚀 功能特性

### 1. 多层次注意力分析
- **层级注意力模式**: 可视化每一层Transformer的注意力分布
- **注意力头专业化**: 分析不同注意力头的功能分工
- **层级演化对比**: 展示注意力从浅层到深层的演化过程

### 2. 统计分析
- **注意力强度统计**: 分析最大注意力、平均注意力的分布
- **注意力熵计算**: 衡量注意力的集中程度和分散程度
- **跨层对比**: 量化分析不同层的注意力特征

### 3. 高质量可视化
- **热力图叠加**: 原图像与注意力热力图的完美融合
- **多头对比**: 并排展示不同注意力头的关注点
- **统计图表**: 专业的箱型图和分布图

## 📋 系统要求

### 依赖库
```bash
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
seaborn>=0.11.0
opencv-python>=4.5.0
numpy>=1.19.0
```

### 安装依赖
```bash
pip install torch torchvision matplotlib seaborn opencv-python numpy
```

## 🔧 使用方法

### 1. 快速开始

```python
from attention_visualizer import AttentionVisualizer

# 初始化可视化器（使用最优模型配置）
model_path = "exp/experiments/efficient_ablation_20250609_101146/embed_dim_ablation_embed_dim_128/best_model.pth"
visualizer = AttentionVisualizer(model_path)

# 运行完整分析
visualizer.run_comprehensive_analysis(model_path)
```

### 2. 测试功能
```bash
# 运行测试脚本
python test_attention_visualizer.py
```

### 3. 分步执行

```python
# 加载模型
visualizer.load_optimal_model()

# 获取测试样本
samples = visualizer.get_test_samples(num_samples=5)

# 分析单个样本
for i, (image, label) in enumerate(samples):
    # 基本注意力可视化
    visualizer.visualize_attention_patterns(image, label, i)
    
    # 注意力头分析
    visualizer.analyze_attention_heads(image, label, i)
    
    # 层级对比
    visualizer.create_layer_comparison(image, label, i)

# 统计分析
visualizer.create_attention_statistics(samples)
```

## 📊 输出结果

### 生成的文件结构
```
attention_analysis/
├── attention_pattern_sample_0.png      # 样本0的注意力模式
├── attention_pattern_sample_1.png      # 样本1的注意力模式
├── ...
├── attention_heads_sample_0.png        # 样本0的注意力头分析
├── attention_heads_sample_1.png        # 样本1的注意力头分析
├── ...
├── layer_comparison_sample_0.png       # 样本0的层级对比
├── layer_comparison_sample_1.png       # 样本1的层级对比
├── ...
├── attention_statistics.png            # 统计分析图表
└── attention_analysis_report.md        # 详细分析报告
```

### 可视化内容说明

#### 1. 注意力模式图 (`attention_pattern_sample_*.png`)
- **上排**: 纯注意力热力图，显示CLS token对各patch的注意力强度
- **下排**: 原图像与注意力热力图叠加，直观显示模型关注区域
- **颜色编码**: 红色表示高注意力，蓝色表示低注意力

#### 2. 注意力头分析 (`attention_heads_sample_*.png`)
- 展示4个注意力头的不同关注模式
- 每个头可能专注于不同的特征（边缘、形状、纹理等）
- 帮助理解多头注意力的分工机制

#### 3. 层级演化对比 (`layer_comparison_sample_*.png`)
- 显示6个Transformer层的注意力演化
- 可观察到从局部到全局的特征提取过程
- 展示模型的分层抽象能力

#### 4. 统计分析 (`attention_statistics.png`)
- **最大注意力**: 每层最强注意力值的分布
- **平均注意力**: 每层平均注意力强度
- **注意力熵**: 注意力分散程度的量化指标

## 🔍 主要发现

基于实验结果，我们观察到以下关键模式：

### 1. 层级特征提取
- **浅层 (Layer 1-2)**: 关注局部边缘和细节特征
- **中层 (Layer 3-4)**: 开始整合局部特征，形成部分结构理解
- **深层 (Layer 5-6)**: 关注全局形状和完整数字结构

### 2. 注意力头专业化
- **头1**: 通常关注数字的主体轮廓
- **头2**: 专注于细节特征和笔画连接
- **头3**: 关注背景和边界信息
- **头4**: 整合全局信息进行最终分类

### 3. 数字特征识别
- 模型能够自动识别数字的关键特征点
- 对于不同数字类别，注意力模式有明显差异
- 弯曲部分、交叉点、端点等关键结构获得更多注意力

## ⚙️ 配置选项

### 模型配置（最优设置）
```python
optimal_config = {
    'img_size': 28,        # 图像尺寸
    'patch_size': 7,       # Patch大小
    'in_channels': 1,      # 输入通道数
    'num_classes': 10,     # 类别数
    'embed_dim': 128,      # 嵌入维度
    'num_heads': 4,        # 注意力头数
    'num_layers': 6,       # Transformer层数
    'mlp_dim': 256,        # MLP维度
    'dropout': 0.1         # Dropout率
}
```

### 可视化参数
```python
visualizer = AttentionVisualizer(
    model_path="path/to/model.pth",
    device='cuda'  # 或 'cpu'
)

# 获取样本数量
samples = visualizer.get_test_samples(num_samples=8)  # 默认8个样本
```

## 📈 性能优化

### 内存优化
- 使用`torch.no_grad()`减少内存占用
- 批处理大小设为1，避免大量attention map占用内存
- 及时释放GPU缓存

### 计算优化
- 注意力权重直接从模型输出获取，无需额外计算
- 使用OpenCV进行高效的图像上采样
- 并行处理多个样本的可视化

## 🚨 注意事项

1. **模型路径**: 确保模型文件存在，如果不存在将使用随机初始化权重
2. **GPU内存**: 大模型可能需要较多GPU内存，可设置`device='cpu'`
3. **依赖库版本**: 确保所有依赖库版本兼容
4. **输出目录**: 程序会自动创建`attention_analysis`目录

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案: 检查模型路径是否正确，或使用随机权重进行演示
   ```

2. **CUDA内存不足**
   ```python
   # 使用CPU
   visualizer = AttentionVisualizer(model_path, device='cpu')
   ```

3. **依赖库缺失**
   ```bash
   pip install -r requirements.txt
   ```

### 调试模式
```python
# 开启详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试单个功能
visualizer.visualize_attention_patterns(image, label, 0)
```

## 📚 扩展功能

### 自定义分析
```python
class CustomAttentionVisualizer(AttentionVisualizer):
    def custom_analysis(self, samples):
        # 添加自定义分析逻辑
        pass
```

### 批量处理
```python
# 处理多个模型
model_paths = ["model1.pth", "model2.pth", "model3.pth"]
for path in model_paths:
    visualizer = AttentionVisualizer(path)
    visualizer.run_comprehensive_analysis(path)
```

## 📄 许可证

本工具基于MIT许可证开源，可自由使用和修改。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个工具！

---

**作者**: AI Research Team  
**最后更新**: 2025年6月  
**版本**: 1.0.0 