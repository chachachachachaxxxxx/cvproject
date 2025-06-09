# Vision Transformer MNIST 消融实验框架

这是一个改进的Vision Transformer消融实验框架，提供了全面的超参数优化和性能分析功能。

## 主要特性

### 🚀 核心功能
- **多维度消融实验**: Patch Size、网络深度、注意力头数、嵌入维度、Dropout率
- **早停机制**: 自动检测收敛，避免过度训练
- **梯度裁剪**: 提高训练稳定性
- **自动化报告**: 生成详细的性能分析报告

### 📊 分析功能
- **性能对比表**: 按准确率排序的详细对比
- **可视化分析图表**: 6种不同角度的性能分析图
- **Markdown报告**: 完整的实验报告文档
- **效率分析**: 准确率/参数量效率指标

### ⚙️ 配置选项
- **配置文件支持**: JSON格式的灵活配置
- **命令行参数**: 支持快速模式和自定义参数
- **模块化设计**: 易于扩展和修改

## 快速开始

### 1. 基本运行

```bash
# 交互式启动器（推荐新手）
python run_experiments.py

# 快速模式（实验数量较少）
python experiment_runner_fixed.py --quick

# 完整模式
python experiment_runner_fixed.py
```

### 2. 使用配置文件

```bash
# 使用示例配置文件
python experiment_runner_fixed.py --config experiment_config_example.json

# 自定义输出目录
python experiment_runner_fixed.py --output-dir my_experiments
```

### 3. 配置文件格式

创建自定义配置文件 `my_config.json`：

```json
{
  "patch_sizes": [4, 7],
  "num_layers_list": [3, 6],
  "num_heads_list": [2, 4],
  "embed_dims": [64, 128],
  "dropout_rates": [0.1, 0.2],
  
  "base_config": {
    "img_size": 28,
    "in_channels": 1,
    "num_classes": 10,
    "embed_dim": 64,
    "mlp_dim": 128,
    "dropout": 0.1
  },
  
  "train_config": {
    "batch_size": 256,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 15,
    "num_workers": 0,
    "early_stopping_patience": 5,
    "min_delta": 0.001
  }
}
```

## 输出文件说明

实验完成后，会在 `experiments/experiment_YYYYMMDD_HHMMSS/` 目录下生成：

### 📁 主目录文件
- `all_results.json`: 所有实验的详细结果数据
- `performance_table.txt`: 性能对比表（按准确率排序）
- `analysis_plots.png`: 6个维度的性能分析图表
- `experiment_report.md`: 完整的Markdown格式报告
- `experiment_config.json`: 本次实验的配置信息
- `system_info.json`: 系统环境信息

### 📁 单个实验子目录
每个实验都有独立的子目录，包含：
- `best_model.pth`: 最佳模型权重
- `result.json`: 实验详细结果
- `training_curves.png`: 训练曲线图
- `attention_visualization.png`: 注意力可视化
- `error_analysis.png`: 错误分析图

## 性能分析图表

生成的 `analysis_plots.png` 包含6个分析图：

1. **Patch Size vs Accuracy**: 不同patch大小对准确率的影响
2. **Number of Layers vs Accuracy**: 网络深度对性能的影响
3. **Parameters vs Accuracy**: 参数量与准确率的关系
4. **Training Time vs Accuracy**: 训练时间与准确率的平衡
5. **FLOPs vs Accuracy**: 计算复杂度与性能的关系
6. **Model Efficiency**: 模型效率评分（准确率/百万参数）

## 实验参数说明

### Patch Size
- **作用**: 决定图像被分割成多少个patch
- **影响**: 较小的patch size通常提供更好的性能，但计算开销更大
- **推荐**: 4, 7, 14

### Network Layers
- **作用**: Transformer的层数
- **影响**: 更深的网络能学习更复杂的特征，但也增加了过拟合风险
- **推荐**: 3, 6, 9

### Attention Heads
- **作用**: 多头注意力的头数
- **影响**: 更多头数提供更丰富的注意力模式，但也增加计算成本
- **推荐**: 2, 4, 8

### Embed Dimension
- **作用**: 特征嵌入的维度
- **影响**: 更大的维度提供更强的表达能力，但参数量增加
- **推荐**: 64, 128, 256

### Dropout Rate
- **作用**: 正则化强度
- **影响**: 适当的dropout防止过拟合，过高会影响学习能力
- **推荐**: 0.1, 0.2

## 早停机制

框架实现了智能早停机制：
- **patience**: 连续多少个epoch没有改进就停止
- **min_delta**: 最小改进阈值
- **自动保存**: 保存最佳epoch的模型

## 性能优化建议

### 💻 硬件配置
- **GPU**: 推荐使用GPU加速训练
- **内存**: 建议至少8GB RAM
- **存储**: 预留至少2GB空间存储结果

### ⚡ 运行优化
- **batch_size**: GPU内存充足时可以增大
- **num_workers**: CPU核数允许时可以增大
- **epochs**: 使用早停机制，可以设置较大值

### 🎯 实验设计
- **快速测试**: 先用 `--quick` 模式验证框架
- **参数筛选**: 先做粗粒度搜索，再细化最优区域
- **分批实验**: 避免一次运行过多实验，可分批进行

## 常见问题

### Q: 如何减少实验时间？
A: 使用 `--quick` 模式，或在配置文件中减少参数组合数量。

### Q: 实验中断怎么办？
A: 每个实验都独立保存，可以从断点继续（手动排除已完成的实验）。

### Q: 如何查看实验进度？
A: 终端会显示当前进度，也可以查看输出目录中的文件。

### Q: 内存不足怎么办？
A: 减小batch_size，或减少并行实验数量。

## 依赖要求

```txt
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
tqdm>=4.62.0
seaborn>=0.11.0
scikit-learn>=0.24.0
psutil>=5.8.0
```

## 许可证

本项目用于学术研究和教育目的。 