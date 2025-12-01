

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAInspiredAdapter(nn.Module):
    """
    LoRAInspiredAdapter的全参数版本。
    移除了低秩分解，所有操作都在完整的 embedding_dim 中进行。
    """
    def __init__(self, embedding_dim=768, expansion_factor=0.25, spatial_kernel=3, dropout=0.1, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.hidden_dim = int(embedding_dim * expansion_factor)
        

        self.fc1 = nn.Linear(embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, embedding_dim)
        
        # --- 2. 空间交互模块 (卷积路径) ---
        # 现在所有卷积的通道数都是 hidden_dim
        self.spatial_interact = nn.Sequential(
            # 使用深度可分离卷积的思想
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=spatial_kernel, 
                      padding=spatial_kernel//2, groups=self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )
        
        # --- 3. 通道注意力模块 ---
        # 注意力在 hidden_dim 上计算
        self.channel_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim, bias=False),
            nn.Sigmoid()
        )
        
        # --- 4. 全尺寸自注意力模块 ---
        # 在 hidden_dim 上进行自注意力计算，成本非常高
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(self.hidden_dim)

        # --- 5. 其他辅助层 ---
        self.norm_input = nn.LayerNorm(embedding_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 可学习缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.1) # 或者直接初始化为0
        
        self._init_weights()
    
    def _init_weights(self):
        # 对全参数版本，通常使用标准初始化
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.weight) # 类似于Adapter的初始化，使得初始时输出为0
        
    def forward(self, x, add_residual=True):
        B, H, W, C = x.shape
        residual = x
        
        # a. 初始归一化和第一次线性变换
        x_norm = self.norm_input(x)
        x_flat = x_norm.view(B, H * W, C)
        x_proj = self.act(self.fc1(x_flat)) # Shape: [B, H*W, hidden_dim]
        
        # b. 通道注意力
        channel_attn = self.channel_attention(x_proj.mean(dim=1))
        x_proj = x_proj * channel_attn.unsqueeze(1)
        
        # --- c. 并行处理：空间卷积 vs. 全局自注意力 ---
        
        # c.1. 空间卷积路径
        x_2d = x_proj.reshape(B, H, W, self.hidden_dim).permute(0, 3, 1, 2).contiguous()
        x_spatial = self.spatial_interact(x_2d)
        x_spatial = x_spatial.permute(0, 2, 3, 1).reshape(B, H * W, self.hidden_dim)
        
        # c.2. 全局自注意力路径
        x_attn_norm = self.norm_attn(x_proj)
        x_global_attn, _ = self.self_attention(x_attn_norm, x_attn_norm, x_attn_norm)
        
        # d. 融合并行路径的结果
        x_fused = x_spatial + x_global_attn
        
        # e. 在融合后添加残差连接
        x_fused = x_fused + x_proj
        x_fused = self.dropout(x_fused)
        
        # f. 第二次线性变换 (上投影)
        x_up = self.fc2(x_fused)
        x_up = x_up.reshape(B, H, W, C) * self.scale
        
        # g. 最终的残差连接
        if add_residual:
            return residual + x_up
        return x_up