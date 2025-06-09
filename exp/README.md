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