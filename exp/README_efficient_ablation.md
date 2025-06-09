# Vision Transformer MNIST 高效消融实验框架

## 概述

这是一个高效的消融实验框架，用于在PyTorch中对Vision Transformer在MNIST数据集上进行系统性的参数研究。通过采用实验设计方法，我们将原本需要243个实验的全因子设计压缩到仅31个实验，节省了87%的计算资源。

## 实验设计策略

### 1. 问题分析
原始全因子设计需要：3×3×3×3×3 = 243个实验
- Patch Size: [4, 7, 14]
- 网络深度: [3, 6, 9]  
- 注意力头数: [2, 4, 8]
- 嵌入维度: [32, 64, 128]
- 批量大小: [32, 64, 128]

### 2. 高效设计方案
我们采用以下策略减少实验数量：

1. **基线实验** (1个)：使用中等参数配置作为参考基准
2. **单因子消融** (15个)：固定其他参数，每次只变化一个因子
3. **正交设计** (9个)：使用L9(3^4)正交表探索参数交互效应
4. **极端配置** (3个)：测试最小、最大、均衡配置
5. **效率导向** (3个)：关注推理速度、参数效率、内存效率

总计：31个实验，压缩比例87%

## 文件结构

```
exp/
├── efficient_ablation_config.json     # 主实验配置文件
├── efficient_ablation_runner.py       # 实验运行器
├── run_efficient_ablation.py          # 启动脚本
├── README_efficient_ablation.md       # 说明文档
└── experiments/                        # 实验结果目录
    └── efficient_ablation_YYYYMMDD_HHMMSS/
        ├── all_results.json           # 所有实验结果
        ├── experiment_data.csv        # 结构化数据
        ├── analysis_plots.png         # 分析图表
        ├── experiment_report.md       # 详细报告
        └── [实验组]/
            └── [实验名称]/
                ├── result.json        # 单个实验结果
                └── best_model.pth     # 最佳模型
```

## 使用方法

### 1. 快速开始

```bash
# 进入实验目录
cd exp/

# 查看实验计划（空运行模式）
python run_efficient_ablation.py --dry-run

# 快速测试（8个关键实验）
python run_efficient_ablation.py --quick

# 完整实验（31个实验）
python run_efficient_ablation.py
```

### 2. 高级用法

```bash
# 使用自定义配置文件
python run_efficient_ablation.py --config my_config.json

# 组合使用
python run_efficient_ablation.py --config my_config.json --quick --dry-run
```

### 3. 直接使用Python

```python
from efficient_ablation_runner import EfficientAblationRunner

# 创建实验运行器
runner = EfficientAblationRunner('efficient_ablation_config.json')

# 运行实验
runner.run_efficient_ablation()
```

## 配置文件说明

### 实验组配置

1. **基线实验**：提供参考基准
2. **单因子消融**：分析各因子的独立效应
3. **正交设计**：探索因子间的交互效应
4. **极端配置**：测试参数边界情况
5. **效率导向**：评估不同效率指标

### 训练配置

```json
{
  "learning_rate": 3e-4,
  "weight_decay": 1e-4,
  "epochs": 30,
  "early_stopping_patience": 5,
  "min_delta": 0.001
}
```

## 实验结果分析

### 1. 自动生成内容

- **性能对比表**：所有实验的准确率、参数量、训练时间对比
- **因子效应分析**：各参数对性能的影响程度
- **可视化图表**：多维度分析图表
- **Markdown报告**：详细的实验报告

### 2. 关键指标

- **准确率**：模型在测试集上的分类准确率
- **参数效率**：每百万参数的准确率
- **计算效率**：每GFLOP的准确率
- **训练时间**：完整训练过程的时间消耗

### 3. 推荐配置

根据不同需求自动推荐：
- 最高准确率配置
- 最佳参数效率配置
- 最快训练配置

## 实验设计优势

### 1. 效率提升
- **计算资源节省87%**：从243个实验减少到31个
- **时间大幅缩短**：保持实验有效性的同时显著减少耗时
- **系统性覆盖**：通过科学的实验设计保证参数空间的全面探索

### 2. 科学性保证
- **正交设计**：最大化信息获取，最小化实验冗余
- **单因子分析**：准确识别各参数的独立效应
- **交互效应检测**：发现参数间的协同或拮抗作用

### 3. 实用性强
- **自动化流程**：一键运行，自动生成报告
- **灵活配置**：支持自定义实验参数和组合
- **结果可视化**：直观的图表和分析报告

## 实验方法论

### 1. 实验设计原理

**正交设计 (Orthogonal Design)**
- 使用L9(3^4)正交表
- 确保各因子水平的均匀分布
- 能够分离主效应和交互效应

**单因子消融 (One-Factor-At-A-Time)**
- 每次固定其他因子，只变化目标因子
- 清晰识别单个因子的影响
- 为后续分析提供基准

### 2. 统计分析方法

- **效应大小计算**：量化各因子的影响程度
- **方差分析**：检验因子效应的显著性
- **相关性分析**：发现因子间的关联关系

### 3. 结果验证

- **重复性检验**：关键实验的重复验证
- **统计显著性**：使用适当的统计检验
- **实际意义评估**：结合实际应用需求分析结果

## 扩展和定制

### 1. 添加新的因子

在配置文件中添加新的单因子消融组：

```json
{
  "new_factor_ablation": {
    "description": "新因子消融",
    "base_config": {...},
    "variable": "new_factor",
    "values": [value1, value2, value3]
  }
}
```

### 2. 修改实验配置

- 调整训练参数（学习率、epoch数等）
- 修改模型架构参数范围
- 增加新的评估指标

### 3. 自定义分析

在`efficient_ablation_runner.py`中扩展分析方法：

```python
def custom_analysis(self):
    # 自定义分析逻辑
    pass
```

## 注意事项

1. **资源需求**：确保有足够的GPU内存和存储空间
2. **时间规划**：完整实验可能需要数小时，建议先运行快速测试
3. **参数合理性**：确保配置的参数组合在实际中是可行的
4. **结果解释**：结合ViT的理论知识解释实验结果

## 技术支持

如遇到问题，请检查：
1. 依赖库是否正确安装（requirements.txt）
2. 数据路径是否正确配置
3. GPU内存是否充足
4. 配置文件格式是否正确

## 参考文献

1. Montgomery, D. C. (2017). Design and analysis of experiments.
2. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
3. Box, G. E., Hunter, J. S., & Hunter, W. G. (2005). Statistics for experimenters. 