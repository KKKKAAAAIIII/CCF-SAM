import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .common import LayerNorm2d
class FeatureSeparationModule(nn.Module):

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

            nn.Conv2d(1, feature_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feature_dim // 4, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(feature_dim), 
        )

        self.shared_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

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
        

        soft_weights = torch.sigmoid(masks)
        soft_weights = torch.clamp(soft_weights, min=self.min_confidence, 
                                   max=1-self.min_confidence)
        lifted_features = self.feature_lifter(masks[:, 0:1, :, :])
       
        if self.num_classes == 1:
            weight = soft_weights[:, 0:1, :, :]
        else:
            weight = soft_weights.max(dim=1, keepdim=True)[0]
        
        
        target_features = lifted_features * weight
        background_features = lifted_features * (1 - weight)
        
        target_feat_extracted = self.shared_feature_extractor(target_features)
        background_feat_extracted = self.shared_feature_extractor(background_features)
        

        target_token = self.target_head(target_feat_extracted)
        background_token = self.background_head(background_feat_extracted)
        
        return {
            'target_token': target_token,
            'background_token': background_token,
            'target_features': target_features,
            'background_features': background_features
            
        }


class SimpleContrastiveLoss(nn.Module):
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, target_tokens: torch.Tensor, bg_tokens: torch.Tensor):
        
        batch_size = target_tokens.shape[0]
        device = target_tokens.device

     
        target_tokens = F.normalize(target_tokens, p=2, dim=-1)
        bg_tokens = F.normalize(bg_tokens, p=2, dim=-1)
  
        all_tokens = torch.cat([target_tokens, bg_tokens], dim=0)
        
        
        logits = torch.matmul(target_tokens, all_tokens.T) / self.temperature
        
       
        positive_mask = torch.zeros_like(logits, dtype=torch.bool)
        
       
        positive_mask[:, :batch_size] = True

        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask[:, :batch_size].masked_fill_(mask_self, False)
        

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
      
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        
      
        loss = -mean_log_prob_pos.mean()
        
        return loss


class TokenAccumulator(nn.Module):

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

        if not self.initialized:
            new_acc_target = target_mean
            new_acc_bg = bg_mean
        else:
            new_acc_target = self.momentum * self.accumulated_target + (1 - self.momentum) * target_mean
            new_acc_bg = self.momentum * self.accumulated_background + (1 - self.momentum) * bg_mean
            

        with torch.no_grad():
            if not self.initialized:
                self.accumulated_target.copy_(target_mean.detach())
                self.accumulated_background.copy_(bg_mean.detach())
                self.initialized.fill_(True)
            else:
                self.accumulated_target.mul_(self.momentum).add_(target_mean.detach(), alpha=1 - self.momentum)
                self.accumulated_background.mul_(self.momentum).add_(bg_mean.detach(), alpha=1 - self.momentum)

        enhanced_target = new_acc_target.unsqueeze(0).expand_as(current_target_token)
        enhanced_bg = new_acc_bg.unsqueeze(0).expand_as(current_bg_token)
        
        return enhanced_target, enhanced_bg
    

class CrossAttentionModule(nn.Module):

    def __init__(self, embed_dim: int = 256, token_dim: int = 128, num_heads: int = 8, ffn_dim_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        

        self.token_proj = nn.Linear(token_dim, embed_dim)
        

        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_token = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=True,
            dropout=dropout 
        )
        

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
        

        if target_token.dim() == 1:
            target_token = target_token.unsqueeze(0).expand(B, -1)
            bg_token = bg_token.unsqueeze(0).expand(B, -1)
        

        target_emb = self.token_proj(target_token)
        bg_emb = self.token_proj(bg_token)
        token_kv = torch.stack([target_emb, bg_emb], dim=1)  # [B, 2, embed_dim]

        img_emb_flat = image_embeddings.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        attended_emb, _ = self.cross_attn(
            query=self.norm_img(img_emb_flat),  
            key=self.norm_token(token_kv),    
            value=self.norm_token(token_kv)   
        )

        img_emb_flat = img_emb_flat + attended_emb
        


        ffn_output = self.ffn(self.norm_ffn(img_emb_flat))

        updated_emb = img_emb_flat + ffn_output
        
        # Reshape back to image format
        updated_emb = updated_emb.permute(0, 2, 1).reshape(B, C, H, W)
        
        return updated_emb