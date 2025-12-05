

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAInspiredAdapter(nn.Module):

    def __init__(self, embedding_dim=768, expansion_factor=0.25, spatial_kernel=3, dropout=0.1, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.hidden_dim = int(embedding_dim * expansion_factor)
        

        self.fc1 = nn.Linear(embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, embedding_dim)

        self.spatial_interact = nn.Sequential(

            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=spatial_kernel, 
                      padding=spatial_kernel//2, groups=self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )
        

        self.channel_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim, bias=False),
            nn.Sigmoid()
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(self.hidden_dim)


        self.norm_input = nn.LayerNorm(embedding_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        

        self.scale = nn.Parameter(torch.ones(1) * 0.1) 
        
        self._init_weights()
    
    def _init_weights(self):

        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.weight) 
        
    def forward(self, x, add_residual=True):
        B, H, W, C = x.shape
        residual = x
        

        x_norm = self.norm_input(x)
        x_flat = x_norm.view(B, H * W, C)
        x_proj = self.act(self.fc1(x_flat))
 
        channel_attn = self.channel_attention(x_proj.mean(dim=1))
        x_proj = x_proj * channel_attn.unsqueeze(1)
        

        

        x_2d = x_proj.reshape(B, H, W, self.hidden_dim).permute(0, 3, 1, 2).contiguous()
        x_spatial = self.spatial_interact(x_2d)
        x_spatial = x_spatial.permute(0, 2, 3, 1).reshape(B, H * W, self.hidden_dim)
        

        x_attn_norm = self.norm_attn(x_proj)
        x_global_attn, _ = self.self_attention(x_attn_norm, x_attn_norm, x_attn_norm)

        x_fused = x_spatial + x_global_attn
        

        x_fused = x_fused + x_proj
        x_fused = self.dropout(x_fused)
        

        x_up = self.fc2(x_fused)
        x_up = x_up.reshape(B, H, W, C) * self.scale
        

        if add_residual:
            return residual + x_up
        return x_up