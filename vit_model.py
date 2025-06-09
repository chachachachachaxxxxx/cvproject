import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    将输入图像分割成patches并转换为嵌入向量
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # 线性投影层，将patch转换为嵌入向量
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.shape[0]
        
        # 将图像分割成patches
        # (batch_size, channels, height, width) -> (batch_size, n_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)
        
        # 线性投影
        x = self.projection(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 应用注意力权重
        out = torch.matmul(attention_probs, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # 最终投影
        out = self.projection(out)
        
        return out, attention_probs


class MLP(nn.Module):
    """
    多层感知机，用于Transformer block中的前馈网络
    """
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        
    def forward(self, x):
        # 多头自注意力 + 残差连接
        attn_out, attention_probs = self.attention(self.norm1(x))
        x = x + attn_out
        
        # MLP + 残差连接
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x, attention_probs


class VisionTransformer(nn.Module):
    """
    Vision Transformer模型
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10, 
                 embed_dim=64, num_heads=4, num_layers=6, mlp_dim=128, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token (可学习的分类token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding (可学习的位置编码)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        # 初始化position embedding
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化其他层的权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Patch embedding
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)
        
        # 2. 添加class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # 3. 添加位置编码
        x = x + self.position_embedding
        x = self.dropout(x)
        
        # 4. 通过Transformer layers
        attention_maps = []
        for transformer_block in self.transformer_blocks:
            x, attention_probs = transformer_block(x)
            attention_maps.append(attention_probs)
        
        # 5. Layer normalization
        x = self.norm(x)
        
        # 6. 分类：使用class token的表示
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)
        
        # 7. 分类头
        logits = self.head(cls_token_final)
        
        return logits, attention_maps


def create_vit_model(model_config=None):
    """
    创建ViT模型的工厂函数
    """
    if model_config is None:
        model_config = {
            'img_size': 28,
            'patch_size': 7,
            'in_channels': 1,
            'num_classes': 10,
            'embed_dim': 64,
            'num_heads': 4,
            'num_layers': 6,
            'mlp_dim': 128,
            'dropout': 0.1
        }
    
    return VisionTransformer(**model_config)


if __name__ == "__main__":
    # 测试模型
    model = create_vit_model()
    
    # 创建测试输入
    x = torch.randn(2, 1, 28, 28)  # (batch_size, channels, height, width)
    
    # 前向传播
    logits, attention_maps = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"注意力图数量: {len(attention_maps)}")
    if attention_maps:
        print(f"每个注意力图形状: {attention_maps[0].shape}") 