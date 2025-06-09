# 实验文件夹 (exp)

这个文件夹包含了所有Vision Transformer MNIST消融实验相关的文件。

## 📁 文件列表

### 🚀 核心实验文件
- **`experiment_runner_fixed.py`**: 主要的消融实验框架
- **`run_experiments.py`**: 交互式实验启动器
- **`quick_experiment.py`**: 快速batch size对比实验

### ⚙️ 配置文件
- **`experiment_config_example.json`**: 实验配置文件示例

### 📖 文档文件
- **`README_experiment.md`**: 详细的实验框架使用文档
- **`README.md`**: 本文件

### 🧪 测试文件
- **`test_experiment.py`**: 测试验证脚本

## 🚀 快速开始

### 1. 测试框架
```bash
cd exp
python test_experiment.py
```

### 2. 运行实验

#### 方法一：交互式启动（推荐）
```bash
cd exp
python run_experiments.py
```

#### 方法二：直接运行
```bash
cd exp
# 快速模式
python experiment_runner_fixed.py --quick

# 完整模式
python experiment_runner_fixed.py

# 使用配置文件
python experiment_runner_fixed.py --config experiment_config_example.json
```

#### 方法三：Batch Size对比实验
```bash
cd exp
python quick_experiment.py
```

## 📊 输出结果

实验结果会保存在以下位置：
- **消融实验结果**: `experiments/experiment_YYYYMMDD_HHMMSS/`
- **Batch Size实验结果**: `batch_size_experiments/experiment_YYYYMMDD_HHMMSS/`

每个实验目录包含：
- 详细的性能分析报告 (Markdown格式)
- 性能对比表 (文本格式)
- 可视化分析图表 (PNG格式)
- 实验配置和系统信息 (JSON格式)
- 各个子实验的详细结果

## 🔧 自定义配置

复制并修改 `experiment_config_example.json` 来自定义实验参数：

```json
{
  "patch_sizes": [4, 7, 14],
  "num_layers_list": [3, 6, 9],
  "num_heads_list": [2, 4, 8],
  "embed_dims": [64, 128],
  "dropout_rates": [0.1, 0.2],
  "train_config": {
    "batch_size": 256,
    "epochs": 15,
    "early_stopping_patience": 5
  }
}
```

## 📋 依赖要求

确保已安装以下依赖：
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- tqdm >= 4.62.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- psutil >= 5.8.0

## 💡 使用提示

1. **首次使用**: 先运行 `test_experiment.py` 验证环境
2. **快速验证**: 使用 `--quick` 模式进行快速测试
3. **批量实验**: 使用配置文件进行大规模实验
4. **结果分析**: 查看生成的Markdown报告获取详细分析

## 🔗 相关文件

本实验框架依赖根目录下的以下文件：
- `vit_model.py`: Vision Transformer模型定义
- `data/`: MNIST数据集存储目录

## 📞 支持

如有问题，请参考：
1. `README_experiment.md` - 详细使用文档
2. `test_experiment.py` - 环境测试
3. 实验生成的报告文件

# Experiment Runner and Visualizer

这个项目包含两个主要组件：

## 1. 实验运行器 (efficient_ablation_runner.py)

负责运行实验并保存结果，不包含可视化功能，避免了中文字体问题。

### 使用方法：

```bash
python efficient_ablation_runner.py
```

这将：
- 加载 `efficient_ablation_config.json` 配置文件
- 运行所有配置的实验
- 保存结果到 `experiments/efficient_ablation_TIMESTAMP/` 目录
- 生成基本的文本报告和数据文件

### 输出文件：
- `all_results.json` - 所有实验结果的JSON格式
- `experiment_data.csv` - 实验数据的CSV格式，便于分析
- `factor_effects.json` - 因子效应分析结果
- `experiment_report.md` - 文本格式的实验报告

## 2. 可视化工具 (experiment_visualizer.py)

独立的可视化工具，读取实验结果并生成各种图表。

### 使用方法：

#### 方法1：直接使用可视化器
```bash
python experiment_visualizer.py --experiment-dir experiments/efficient_ablation_TIMESTAMP/
```

#### 方法2：使用便捷脚本
```bash
python run_visualization.py --experiment-dir experiments/efficient_ablation_TIMESTAMP/
```

#### 可选参数：
- `--output-dir` - 指定可视化输出目录（默认为experiment-dir/visualizations）

### 生成的可视化图表：

1. **Overview Dashboard** (`overview_dashboard.png`)
   - 实验结果总览
   - 准确率分布、参数分析、效率指标

2. **Ablation Analysis** (`ablation_analysis.png`)
   - 各个超参数的消融分析
   - 显示每个参数对性能的影响

3. **Correlation Heatmap** (`correlation_heatmap.png`)
   - 配置参数和性能指标的相关性矩阵

4. **Performance Analysis** (`performance_analysis.png`)
   - Pareto前沿分析
   - 效率比较

5. **Training Curves** (`training_curves.png`)
   - 顶级实验的训练曲线

6. **Summary Report** (`summary_report.png`)
   - 高级总结和统计信息

7. **HTML报告** (`index.html`)
   - 包含所有图表的交互式HTML报告

## 工作流程

1. **运行实验**：
   ```bash
   cd exp/
   python efficient_ablation_runner.py
   ```

2. **生成可视化**：
   ```bash
   python run_visualization.py --experiment-dir experiments/efficient_ablation_TIMESTAMP/
   ```

3. **查看结果**：
   - 打开 `experiments/efficient_ablation_TIMESTAMP/visualizations/index.html`
   - 或查看各个PNG图片文件

## 特点

### 实验运行器特点：
- ✅ 无可视化依赖，避免字体警告
- ✅ 高效内存管理
- ✅ 自动保存实验结果
- ✅ 支持早停和学习率调整
- ✅ 详细的进度显示

### 可视化工具特点：
- ✅ 独立运行，不影响实验
- ✅ 丰富的图表类型
- ✅ 自动生成HTML报告
- ✅ 支持Pareto分析
- ✅ 所有图片都保存为文件

## 依赖关系

### 实验运行器依赖：
```
torch
torchvision
numpy
pandas
tqdm
```

### 可视化工具额外依赖：
```
matplotlib
seaborn
```

## 故障排除

1. **中文字体警告**：
   - 现在已经解决，实验运行器不使用matplotlib

2. **可视化图片不显示**：
   - 确保安装了matplotlib和seaborn
   - 所有图片都自动保存到文件，无需GUI

3. **内存不足**：
   - 实验运行器已优化内存使用
   - 每个实验后自动清理GPU内存

## 示例

完整的使用示例：

```bash
# 1. 运行实验
python efficient_ablation_runner.py

# 2. 生成可视化（假设实验目录为 experiments/efficient_ablation_20231201_143022）
python run_visualization.py --experiment-dir experiments/efficient_ablation_20231201_143022/

# 3. 查看结果
# 打开 experiments/efficient_ablation_20231201_143022/visualizations/index.html
``` 