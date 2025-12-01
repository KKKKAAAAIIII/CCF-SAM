import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .common import LayerNorm2d
class FeatureSeparationModule(nn.Module):
    """
    将mask decoder的输出分离为前景和背景特征，并编码为tokens
    使用共享的特征提取器和独立的输出头
    """
    def __init__(
        self, 
        num_classes: int = 1,
        feature_dim: int = 256,
        token_dim: int = 128,
        min_confidence: float = 0.1,
        use_mask_attention: bool = False  
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.min_confidence = min_confidence
        # Mask到特征的转换
        self.mask_to_feature = nn.Sequential(
            nn.Conv2d(num_classes, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim)
        )
        self.feature_lifter = nn.Sequential(
            # 使用3个不同kernel的卷积来捕捉多尺度信息
            nn.Conv2d(1, feature_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feature_dim // 4, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(feature_dim), # 添加归一化
        )
        # 共享的特征提取器
        self.shared_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 独立的输出头
        self.target_head = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.token_dim),
            nn.LayerNorm(self.token_dim)
        )
        
        self.background_head = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.token_dim),
            nn.LayerNorm(self.token_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        masks: torch.Tensor,
        image_embeddings: torch.Tensor,
        intermediate_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = masks.shape
        
        # 1. 生成软分离权重
        soft_weights = torch.sigmoid(masks)
        soft_weights = torch.clamp(soft_weights, min=self.min_confidence, 
                                   max=1-self.min_confidence)
        lifted_features = self.feature_lifter(masks[:, 0:1, :, :])
        # 2. 将mask转换为特征
        # mask_features = self.mask_to_feature(masks)
        
        # # 3. 如果提供了中间特征，进行融合
        # if intermediate_features is not None:
        #     intermediate_features = F.interpolate(
        #         intermediate_features,
        #         size=(H, W),
        #         mode='bilinear',
        #         align_corners=False
        #     )
        #     mask_features = mask_features + intermediate_features
        # if mask_features.shape[-2:] != image_embeddings.shape[-2:]:
        #     mask_features = F.interpolate(
        #         mask_features,
        #         size=image_embeddings.shape[-2:],
        #         mode='bilinear',
        #         align_corners=False
        #     )
        # fused_features = image_embeddings + mask_features
        # if soft_weights.shape[-2:] != fused_features.shape[-2:]:
        #     soft_weights = F.interpolate(
        #         soft_weights,
        #         size=fused_features.shape[-2:], # 目标尺寸
        #         mode='bilinear',
        #         align_corners=False
        # )
        # 4. 特征分离

        if self.num_classes == 1:
            weight = soft_weights[:, 0:1, :, :]
        else:
            weight = soft_weights.max(dim=1, keepdim=True)[0]
        
        # target_features = fused_features * weight
        # background_features = fused_features * (1 - weight)
        # if weight.shape[-2:] != image_embeddings.shape[-2:]:
        #     weight = F.interpolate(
        #         weight, size=image_embeddings.shape[-2:],
        #         mode='bilinear', align_corners=False
        #     )
        
        # 2. 直接用权重分离高质量的 image_embeddings
        target_features = lifted_features * weight
        background_features = lifted_features * (1 - weight)
        # 5. 提取特征
        target_feat_extracted = self.shared_feature_extractor(target_features)
        background_feat_extracted = self.shared_feature_extractor(background_features)
        
        # 6. 生成tokens
        target_token = self.target_head(target_feat_extracted)
        background_token = self.background_head(background_feat_extracted)
        
        return {
            'target_token': target_token,
            'background_token': background_token,
            'target_features': target_features,
            'background_features': background_features
            
        }


class SimpleContrastiveLoss(nn.Module):
    """
    适用于医学图像前景/背景的监督对比损失。
    
    假设:
    1. 一个批次内的所有 `target_tokens` 都来自同一语义类别（例如，同一种病灶）。
    2. 因此，对于每个 `target_token`，批次内所有其他的 `target_tokens` 都是正样本。
    3. 所有 `bg_tokens` 都是负样本。
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, target_tokens: torch.Tensor, bg_tokens: torch.Tensor):
        """
        Args:
            target_tokens (torch.Tensor): 前景token, [B, D]。
            bg_tokens (torch.Tensor): 背景token, [B, D]。
        """
        batch_size = target_tokens.shape[0]
        device = target_tokens.device

        # 步骤1: 归一化
        target_tokens = F.normalize(target_tokens, p=2, dim=-1)
        bg_tokens = F.normalize(bg_tokens, p=2, dim=-1)
        
        # 步骤2: 将所有token拼接起来作为对比池
        # 池的结构: [targets | backgrounds]
        # all_tokens 的形状是 [2B, D]
        all_tokens = torch.cat([target_tokens, bg_tokens], dim=0)
        
        # 步骤3: 计算两两之间的相似度
        # `target_tokens` 作为 query, `all_tokens` 作为 keys
        # logits 的形状是 [B, 2B]
        logits = torch.matmul(target_tokens, all_tokens.T) / self.temperature
        
        # 步骤4: 创建一个正样本的掩码 (mask)
        # 对于第 i 个 target_token (作为 query)，哪些是它的正样本？
        # 答案是：所有其他的 target_tokens (在 all_tokens 的索引 0 到 B-1)
        
        # 创建一个 [B, 2B] 的掩码，初始全为0
        positive_mask = torch.zeros_like(logits, dtype=torch.bool)
        
        # 将 target-vs-target 部分设置为 True
        positive_mask[:, :batch_size] = True
        
        # 排除掉自己和自己对比的情况 (对角线)
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask[:, :batch_size].masked_fill_(mask_self, False)
        
        # `positive_mask` 现在准确地指出了每个 query 对应的所有正样本的位置

        # 步骤5: 计算损失
        # Supervised Contrastive Loss 的标准计算方式
        
        # 为了数值稳定性，从每一行减去最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 计算分母：log(sum(exp(sim)))
        # 分母包括所有的样本（正样本和负样本）
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算分子：sum over positives (log(exp(sim))) = sum over positives (sim)
        # 我们只关心正样本位置上的 log_prob
        # 使用掩码，将非正样本位置的 log_prob 设为0，然后求和
        # positive_mask.sum(1) 是每个样本的正样本数量 (B-1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        
        # 损失是负的 log-likelihood，所以取负号
        loss = -mean_log_prob_pos.mean()
        
        return loss


class TokenAccumulator(nn.Module):
    """跨图像累积token的模块"""
    def __init__(self, token_dim: int, momentum: float = 0.9):
        super().__init__()
        self.token_dim = token_dim
        self.momentum = momentum
        
        self.register_buffer('accumulated_target', torch.zeros(token_dim))
        self.register_buffer('accumulated_background', torch.zeros(token_dim))
        self.register_buffer('initialized', torch.tensor(False))
        
    def forward(
        self, 
        current_target_token: torch.Tensor, 
        current_bg_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        target_mean = current_target_token.mean(dim=0)
        bg_mean = current_bg_token.mean(dim=0)
        
        # --- 核心修改 ---
        # 1. 计算出“即将成为”新状态的token (仍然带有梯度)
        if not self.initialized:
            new_acc_target = target_mean
            new_acc_bg = bg_mean
        else:
            new_acc_target = self.momentum * self.accumulated_target + (1 - self.momentum) * target_mean
            new_acc_bg = self.momentum * self.accumulated_background + (1 - self.momentum) * bg_mean
            
        # 2. 更新内部状态时，使用 detach() 切断梯度，防止跨迭代传播
        with torch.no_grad():
            if not self.initialized:
                self.accumulated_target.copy_(target_mean.detach())
                self.accumulated_background.copy_(bg_mean.detach())
                self.initialized.fill_(True)
            else:
                self.accumulated_target.mul_(self.momentum).add_(target_mean.detach(), alpha=1 - self.momentum)
                self.accumulated_background.mul_(self.momentum).add_(bg_mean.detach(), alpha=1 - self.momentum)
        
        # 3. 返回那个“带有梯度”的新状态，并扩展到批次大小
        enhanced_target = new_acc_target.unsqueeze(0).expand_as(current_target_token)
        enhanced_bg = new_acc_bg.unsqueeze(0).expand_as(current_bg_token)
        
        return enhanced_target, enhanced_bg
    

class CrossAttentionModule(nn.Module):
    """使用累积的tokens来增强image embeddings (采用Pre-Norm结构)"""
    def __init__(self, embed_dim: int = 256, token_dim: int = 128, num_heads: int = 8, ffn_dim_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        
        # Token投影到embedding空间
        self.token_proj = nn.Linear(token_dim, embed_dim)
        
        # Pre-Norm LayerNorms
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_token = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=True,
            dropout=dropout # 在注意力权重上加dropout
        )
        
        # 更完整的前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_dim_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_dim_multiplier, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        image_embeddings: torch.Tensor,  # [B, C, H, W]
        target_token: torch.Tensor,      # [B, token_dim] or [token_dim]
        bg_token: torch.Tensor           # [B, token_dim] or [token_dim]
    ) -> torch.Tensor:
        B, C, H, W = image_embeddings.shape
        
        # 处理token维度
        if target_token.dim() == 1:
            target_token = target_token.unsqueeze(0).expand(B, -1)
            bg_token = bg_token.unsqueeze(0).expand(B, -1)
        
        # 投影tokens到embedding空间
        target_emb = self.token_proj(target_token)
        bg_emb = self.token_proj(bg_token)
        token_kv = torch.stack([target_emb, bg_emb], dim=1)  # [B, 2, embed_dim]
        
        # Reshape image embeddings
        img_emb_flat = image_embeddings.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # --- Pre-Norm Cross-Attention ---
        # 1. 残差连接 + Attention
        attended_emb, _ = self.cross_attn(
            query=self.norm_img(img_emb_flat),   # 对Query进行Norm
            key=self.norm_token(token_kv),     # 对Key进行Norm
            value=self.norm_token(token_kv)    # 对Value进行Norm
        )
        # 第一个残差连接
        img_emb_flat = img_emb_flat + attended_emb
        
        # --- Pre-Norm FFN ---
        # 2. 残差连接 + FFN
        ffn_output = self.ffn(self.norm_ffn(img_emb_flat))
        # 第二个残差连接
        updated_emb = img_emb_flat + ffn_output
        
        # Reshape back to image format
        updated_emb = updated_emb.permute(0, 2, 1).reshape(B, C, H, W)
        
        return updated_emb