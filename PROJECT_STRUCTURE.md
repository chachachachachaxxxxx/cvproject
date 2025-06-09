# Vision Transformer MNIST 项目结构

这是一个完整的Vision Transformer在MNIST数据集上的实现和实验项目。

## 📁 项目结构

```
cvproject/
├── 📁 exp/                          # 实验文件夹
│   ├── experiment_runner_fixed.py   # 消融实验框架
│   ├── run_experiments.py           # 交互式实验启动器
│   ├── quick_experiment.py          # Batch Size对比实验
│   ├── test_experiment.py           # 测试验证脚本
│   ├── experiment_config_example.json # 配置文件示例
│   ├── README_experiment.md         # 实验框架详细文档
│   ├── README.md                    # 实验文件夹说明
│   └── experiments/                 # 实验输出目录
│
├── 📁 docs/                         # 文档目录
│   └── vision_transformer_mnist.md  # ViT原理详细文档
│
├── 📁 data/                         # 数据目录
│   └── MNIST/                       # MNIST数据集
│
├── 📁 train/                        # 训练输出目录
│   └── train_YYYYMMDD_HHMMSS/       # 按时间戳组织的训练结果
│
├── 📁 test/                         # 测试输出目录
│   └── test_YYYYMMDD_HHMMSS/        # 按时间戳组织的测试结果
│
├── 📁 experiments/                  # 实验结果目录
│   └── experiment_YYYYMMDD_HHMMSS/  # 按时间戳组织的实验结果
│
├── 📄 vit_model.py                  # ViT模型定义
├── 📄 train.py                      # 训练脚本
├── 📄 inference.py                  # 推理和评估脚本
├── 📄 mnist_visualizer.py           # MNIST数据可视化工具
├── 📄 requirements.txt              # 项目依赖
├── 📄 README.md                     # 项目主文档
└── 📄 PROJECT_STRUCTURE.md          # 本文件
```

## 🚀 快速开始

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 基础训练和推理
```bash
# 训练模型
python train.py

# 推理评估
python inference.py
```

### 3. 实验研究
```bash
# 进入实验文件夹
cd exp

# 测试实验框架
python test_experiment.py

# 运行消融实验
python run_experiments.py

# 或直接运行快速模式
python experiment_runner_fixed.py --quick
```

## 📊 主要功能

### 🤖 模型实现
- **完整的ViT架构**: 从零实现Vision Transformer
- **MNIST适配**: 针对28×28灰度图像优化
- **注意力可视化**: 可视化模型注意力机制

### 🔬 实验功能
- **消融实验**: 系统性超参数调优
- **性能分析**: 多维度性能评估
- **自动报告**: 详细的分析报告生成
- **可视化**: 丰富的图表和可视化

### 📈 分析工具
- **训练曲线**: 损失和准确率曲线
- **混淆矩阵**: 分类性能分析
- **错误分析**: 错误样例分析
- **效率评估**: 准确率/参数量效率

## 📖 文档说明

### 核心文档
- **`README.md`**: 项目主要说明
- **`docs/vision_transformer_mnist.md`**: ViT原理详解（6000+字）
- **`exp/README_experiment.md`**: 实验框架详细文档

### 使用文档
- **`exp/README.md`**: 实验文件夹快速入门
- **`PROJECT_STRUCTURE.md`**: 本项目结构说明

## 🔧 核心文件说明

### 模型相关
- **`vit_model.py`**: ViT模型的完整实现
  - PatchEmbedding: 图像patch嵌入
  - MultiHeadSelfAttention: 多头自注意力
  - TransformerBlock: Transformer块
  - VisionTransformer: 完整ViT模型

### 训练相关
- **`train.py`**: 标准训练脚本
  - 数据加载和预处理
  - 训练循环和验证
  - 模型保存和可视化

### 推理相关
- **`inference.py`**: 推理和评估脚本
  - 模型加载和评估
  - 性能分析和可视化
  - 错误分析和注意力可视化

### 实验相关
- **`exp/experiment_runner_fixed.py`**: 消融实验主框架
- **`exp/quick_experiment.py`**: Batch Size对比实验
- **`exp/run_experiments.py`**: 交互式启动器

## 🎯 使用场景

### 学习研究
1. **理解ViT**: 阅读文档和代码理解Vision Transformer
2. **动手实践**: 运行训练和推理了解模型行为
3. **深入分析**: 使用实验框架进行超参数调优

### 项目开发
1. **模型基础**: 基于现有ViT实现进行扩展
2. **实验框架**: 使用实验框架进行系统性研究
3. **结果分析**: 利用自动生成的报告进行分析

### 教学演示
1. **课程材料**: 完整的文档和代码适合教学
2. **实验演示**: 实时运行展示模型训练过程
3. **结果展示**: 丰富的可视化适合结果展示

## 💡 特色功能

### 🔄 自动化程度高
- 自动数据下载和预处理
- 自动模型保存和加载
- 自动生成详细分析报告

### 📊 可视化丰富
- 训练曲线实时显示
- 注意力机制可视化
- 多维度性能分析图表

### 🛠️ 易于扩展
- 模块化设计便于修改
- 配置文件支持灵活调参
- 清晰的代码结构便于理解

## 📞 技术支持

如有问题或建议，请参考：
1. 相关文档文件
2. 代码注释和示例
3. 实验生成的详细报告

---

*本项目提供了从模型实现到实验分析的完整解决方案，适用于学习、研究和开发多种场景。* 