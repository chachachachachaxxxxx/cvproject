# Vision Transformer (ViT) 在MNIST手写数字识别中的应用

## 1. 引言

Vision Transformer (ViT) 是由Google Research在2020年提出的革命性模型，将Transformer架构从自然语言处理领域成功迁移到计算机视觉领域。本项目使用ViT实现MNIST手写数字识别任务，展示了Transformer在图像分类中的强大能力。

## 2. Vision Transformer 原理详解

### 2.1 核心思想

Vision Transformer的核心思想是将图像处理方式类比于自然语言处理：
- **图像块(Image Patches)**：将图像分割成固定大小的块，类似于NLP中的tokens
- **序列建模**：将图像块序列化，使用Transformer处理这个图像块序列
- **全局注意力**：通过self-attention机制捕获图像块之间的全局依赖关系

### 2.2 模型架构

ViT的整体架构包含以下几个关键组件：

```
输入图像 (28×28×1)
    ↓
图像块划分 (7×7 patches → 16个patches)
    ↓
线性投影 + 位置编码
    ↓
Transformer Encoder 层 (×N)
    ↓
分类头 (MLP)
    ↓
输出类别 (0-9)
```

#### 2.2.1 图像块嵌入 (Patch Embedding)

将输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 重新整理为一系列扁平化的2D图像块序列：

$$x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

其中：
- $H, W$ 是原始图像的高度和宽度
- $C$ 是通道数
- $P$ 是每个图像块的分辨率
- $N = HW/P^2$ 是图像块的数量

每个图像块通过可训练的线性投影映射到 $D$ 维嵌入空间：

$$z_0 = [x_{class}; x_p^1E; x_p^2E; \cdots; x_p^NE] + E_{pos}$$

其中：
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是图像块嵌入投影矩阵
- $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ 是位置嵌入
- $x_{class}$ 是可学习的分类token

#### 2.2.2 Transformer Encoder

Transformer encoder由多个相同的层组成，每层包含：

**多头自注意力机制 (Multi-Head Self-Attention, MSA)**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MSA}(z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

其中每个attention head计算为：
$$\text{head}_i = \text{Attention}(zW_i^Q, zW_i^K, zW_i^V)$$

**多层感知机 (MLP)**：

$$\text{MLP}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**残差连接和层归一化**：

$$z'_{\ell} = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$
$$z_{\ell} = \text{MLP}(\text{LN}(z'_{\ell})) + z'_{\ell}$$

#### 2.2.3 分类头

使用分类token的最终表示进行分类：

$$y = \text{LN}(z_L^0)$$

其中 $z_L^0$ 是第 $L$ 层输出的分类token表示。

### 2.3 与CNN的对比

| 特性 | CNN | Vision Transformer |
|------|-----|-------------------|
| 感受野 | 局部 → 全局（逐层扩大） | 全局（每层都是全局） |
| 归纳偏置 | 强（平移不变性、局部性） | 弱 |
| 数据需求 | 相对较少 | 大量数据 |
| 计算复杂度 | $O(HW)$ | $O(N^2)$ |
| 可解释性 | 较难 | 注意力图可视化 |

## 3. MNIST数据集适配

### 3.1 数据预处理

MNIST数据集特点：
- 图像尺寸：28×28像素
- 通道数：1（灰度图像）
- 类别数：10（0-9数字）
- 训练集：60,000张图像
- 测试集：10,000张图像

对于ViT的适配：
- **图像块大小**：使用4×4或7×7的patch size
- **序列长度**：28×28图像用7×7 patches得到16个patches
- **数据增强**：随机旋转、平移、缩放

### 3.2 模型配置

针对MNIST的ViT配置：
- 嵌入维度：64
- 注意力头数：4
- Transformer层数：6
- MLP隐藏层维度：128
- Dropout率：0.1

## 4. 实现细节

### 4.1 关键技术点

1. **位置编码**：使用可学习的位置嵌入，为每个patch位置学习唯一的位置表示
2. **分类token**：在序列开头添加特殊的[CLS] token用于分类
3. **注意力机制**：使用scaled dot-product attention
4. **层归一化**：在每个子层之前应用LayerNorm
5. **残差连接**：帮助训练深层网络

### 4.2 训练策略

1. **优化器**：AdamW优化器，学习率3e-4
2. **学习率调度**：余弦退火学习率调度
3. **正则化**：Dropout + Weight decay
4. **批量大小**：64
5. **训练轮数**：100 epochs

### 4.3 代码架构

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, 
                 num_heads, num_layers, mlp_dim, dropout=0.1):
        # 初始化各个组件
        
    def forward(self, x):
        # 1. Patch embedding
        # 2. 添加位置编码
        # 3. 通过Transformer layers
        # 4. 分类预测
```

## 5. 实验结果分析

### 5.1 性能指标

预期结果：
- **准确率**：>98%
- **收敛速度**：相比CNN较慢，但最终性能更优
- **参数量**：相对较少（约100K参数）

### 5.2 注意力可视化

Vision Transformer的一个重要优势是可解释性。通过可视化attention maps，我们可以观察到：
- 模型关注的图像区域
- 不同注意力头的专注模式
- 随训练进行的注意力模式演化

### 5.3 与传统方法对比

| 方法 | 准确率 | 参数量 | 训练时间 |
|------|--------|--------|----------|
| LeNet | ~99% | ~60K | 快 |
| CNN | ~99.2% | ~100K | 中等 |
| **ViT** | **~98.5%** | **~100K** | **较慢** |

## 6. 结论与展望

### 6.1 优势
1. **全局建模能力**：自注意力机制能够捕获长距离依赖
2. **可扩展性**：模型规模可以灵活调整
3. **可解释性**：attention maps提供了模型决策的可视化解释
4. **迁移能力**：预训练的ViT在各种视觉任务上表现优秀

### 6.2 局限性
1. **数据需求大**：需要大量数据才能达到最佳性能
2. **计算复杂度高**：二次复杂度的注意力计算
3. **缺乏归纳偏置**：不如CNN对图像的先验知识利用充分

### 6.3 未来发展方向
1. **高效注意力机制**：Linear attention, Sparse attention
2. **混合架构**：CNN + Transformer
3. **自监督预训练**：MAE, DINO等方法
4. **移动端优化**：模型压缩和量化

## 参考文献

1. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.

2. Vaswani, A., et al. (2017). Attention is all you need. *Advances in neural information processing systems*.

3. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*. 