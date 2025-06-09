# Vision Transformer (ViT) for MNIST Hand-written Digit Recognition

本项目使用Vision Transformer实现MNIST手写数字识别任务，展示了Transformer架构在计算机视觉领域的应用。

## 项目结构

```
cvproject/
├── docs/
│   ├── requirements.md                    # 作业要求
│   └── vision_transformer_mnist.md       # 详细的原理介绍和实现细节
├── vit_model.py                          # Vision Transformer模型实现
├── train.py                              # 训练脚本
├── inference.py                          # 推理和评估脚本
├── requirements.txt                      # 项目依赖
└── README.md                            # 项目说明
```

## 功能特性

- **完整的ViT实现**：从零开始实现Vision Transformer所有组件
- **MNIST适配**：针对MNIST数据集优化的模型配置
- **注意力可视化**：可视化模型的注意力机制
- **全面评估**：包含准确率、混淆矩阵、错误分析等
- **训练监控**：实时显示训练进度和性能指标

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练脚本会：
- 自动下载MNIST数据集
- 训练Vision Transformer模型
- 创建时间戳文件夹保存所有输出
- 保存最佳模型和训练历史
- 生成训练曲线和注意力可视化
- 生成训练摘要报告

### 2. 模型推理

```bash
python inference.py
```

推理脚本会：
- 自动查找最新训练的模型
- 创建时间戳文件夹保存测试结果
- 全面评估模型性能
- 生成混淆矩阵
- 分析错误预测样本
- 展示随机预测示例
- 生成测试摘要报告

### 3. 测试模型结构

```bash
python vit_model.py
```

## 模型架构

### 核心组件

1. **图像块嵌入 (Patch Embedding)**
   - 将28×28图像分割为7×7的patches（共16个patches）
   - 线性投影到64维嵌入空间

2. **位置编码 (Position Embedding)**
   - 可学习的位置嵌入
   - 为每个patch和分类token提供位置信息

3. **Transformer编码器**
   - 6层Transformer blocks
   - 每层包含多头自注意力和MLP
   - 4个注意力头，隐藏层维度128

4. **分类头**
   - 使用[CLS] token的最终表示
   - 单层线性分类器

### 模型配置

```python
model_config = {
    'img_size': 28,        # 图像尺寸
    'patch_size': 7,       # Patch大小
    'in_channels': 1,      # 输入通道数
    'num_classes': 10,     # 类别数
    'embed_dim': 64,       # 嵌入维度
    'num_heads': 4,        # 注意力头数
    'num_layers': 6,       # Transformer层数
    'mlp_dim': 128,        # MLP隐藏层维度
    'dropout': 0.1         # Dropout率
}
```

## 训练配置

- **优化器**: AdamW (lr=3e-4, weight_decay=1e-4)
- **学习率调度**: 余弦退火
- **批量大小**: 64
- **训练轮数**: 100 epochs
- **数据增强**: 随机旋转、平移

## 预期性能

- **准确率**: >98%
- **参数量**: ~100K
- **训练时间**: 约30-60分钟 (GPU)

## 文件组织结构

项目采用时间戳文件夹来组织训练和测试输出：

```
cvproject/
├── train/                              # 训练输出目录
│   └── train_YYYYMMDD_HHMMSS/         # 按时间戳命名的训练文件夹
│       ├── best_vit_model.pth         # 最佳模型权重
│       ├── final_vit_model.pth        # 最终模型权重
│       ├── training_curves.png        # 训练曲线图
│       ├── attention_visualization.png # 注意力可视化
│       └── training_summary.txt       # 训练摘要
├── test/                               # 测试输出目录
│   └── test_YYYYMMDD_HHMMSS/          # 按时间戳命名的测试文件夹
│       ├── confusion_matrix.png       # 混淆矩阵
│       ├── error_analysis.png         # 错误分析
│       ├── random_prediction_example.png # 随机预测示例
│       └── test_summary.txt           # 测试摘要
└── ...
```

## 技术亮点

1. **纯PyTorch实现**: 完全使用PyTorch从零实现ViT
2. **模块化设计**: 代码结构清晰，易于理解和修改
3. **可视化分析**: 丰富的可视化功能帮助理解模型行为
4. **全面评估**: 多维度的模型性能评估
5. **中文注释**: 详细的中文注释便于学习

## 原理文档

详细的原理介绍和实现细节请参考：`docs/vision_transformer_mnist.md`

该文档包含：
- Vision Transformer原理详解
- 数学公式推导
- 与CNN的对比分析
- MNIST数据集适配方案
- 实验结果分析
- 未来发展方向

## 参考资料

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
2. [Meta-Transformer: A Unified Framework for Multimodal Learning](https://www.arxiv.org/abs/2504.13181)

## 许可证

本项目仅用于学习和研究目的。 