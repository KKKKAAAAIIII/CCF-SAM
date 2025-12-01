# SAM/build_sam.py

import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

from SAM.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from SAM.modeling.image_encoder import ImageEncoderViT
from SAM.modeling.CVPSAM import SAMWithTokenEnhancement


def _build_sam_with_token_enhancement(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        checkpoint=None,
        # Token enhancement specific parameters
        use_lora=False,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_modules=('qkv', 'proj', 'mlp', 'neck'),
        use_token_accumulation=True,
        token_momentum=0.5,
        # Loss weights
        coarse_loss_weight=0.8,
        final_loss_weight=0.2,
        contrastive_loss_weight=0.01,
):
    """构建带有Token增强的SAM模型"""
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    # Configure each component
    image_encoder_config = {
        'img_size': image_size,
        'patch_size': vit_patch_size,
        'in_chans': 3,
        'embed_dim': encoder_embed_dim,
        'depth': encoder_depth,
        'num_heads': encoder_num_heads,
        'mlp_ratio': 4,
        'out_chans': prompt_embed_dim,
        'qkv_bias': True,
        'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6),
        'use_rel_pos': True,
        'rel_pos_zero_init': True,
        'window_size': 14,
        'global_attn_indexes': encoder_global_attn_indexes,
        # LoRA parameters
        'use_lora': use_lora,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_target_modules': lora_target_modules,
    }
    
    prompt_encoder_config = {
        'embed_dim': prompt_embed_dim,
        'image_embedding_size': (image_embedding_size, image_embedding_size),
        'input_image_size': (image_size, image_size),
        'mask_in_chans': 16,
    }
    
    mask_decoder_config = {
        'transformer_dim': prompt_embed_dim,
        'num_multimask_outputs': 1,
        'iou_head_depth': 3,
        'iou_head_hidden_dim': 256,
        'transformer': TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,)
    }
    
    feature_separator_config = {
        'num_classes': num_classes,
        'feature_dim': prompt_embed_dim,
        'token_dim': 128,
        'use_mask_attention': False,
    }
    
    cross_attention_config = {
        'embed_dim': prompt_embed_dim,
        'token_dim': 128,
        'num_heads': 8,
        'ffn_dim_multiplier': 4,
        'dropout': 0.1,
    }
    
    # Create the enhanced SAM model
    sam = SAMWithTokenEnhancement(
        image_encoder_config=image_encoder_config,
        mask_decoder_config=mask_decoder_config,
        prompt_encoder_config=prompt_encoder_config,
        feature_separator_config=feature_separator_config,
        cross_attention_config=cross_attention_config,
        use_token_accumulation=use_token_accumulation,
        token_momentum=token_momentum,
        coarse_loss_weight=coarse_loss_weight,
        final_loss_weight=final_loss_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    sam.train()
    # Load checkpoint if provided
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        try:
            sam.load_state_dict(state_dict)

        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint from {checkpoint} (partial weights)")
    
    return sam


def load_from(sam, state_dict, image_size, vit_patch_size):
    """Helper function to handle checkpoint loading with size mismatches"""
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[
                          2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear',
                                           align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict





# Enhanced SAM build functions with token enhancement
def build_sam_vit_b_enhanced(
    image_size, 
    num_classes, 
    pixel_mean=[123.675, 116.28, 103.53], 
    pixel_std=[58.395, 57.12, 57.375],
    checkpoint=None,
    **kwargs
):
    """Build SAM ViT-B with token enhancement"""
    return _build_sam_with_token_enhancement(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        **kwargs
    )


def build_sam_vit_l_enhanced(
    image_size, 
    num_classes, 
    pixel_mean=[123.675, 116.28, 103.53], 
    pixel_std=[58.395, 57.12, 57.375],
    checkpoint=None,
    **kwargs
):
    """Build SAM ViT-L with token enhancement"""
    return _build_sam_with_token_enhancement(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        **kwargs
    )


def build_sam_vit_h_enhanced(
    image_size, 
    num_classes, 
    pixel_mean=[123.675, 116.28, 103.53], 
    pixel_std=[58.395, 57.12, 57.375],
    checkpoint=None,
    **kwargs
):
    """Build SAM ViT-H with token enhancement"""
    return _build_sam_with_token_enhancement(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        **kwargs
    )


# Update registry
build_sam = build_sam_vit_h_enhanced  # Default to enhanced ViT-H

sam_model_registry = {
    "default": build_sam,
    "vit_b_enhanced": build_sam_vit_b_enhanced,
    "vit_l_enhanced": build_sam_vit_l_enhanced,
    "vit_h_enhanced": build_sam_vit_h_enhanced,
}



class ModelSAMEnhanced(nn.Module):
    """增强版的SAM模型封装，包含Token Enhancement"""
    def __init__(
        self, 
        image_size=512, 
        num_classes=1,
        model_type="vit_b",
        checkpoint=None,
        use_lora=False,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
        use_token_accumulation=True,
        token_momentum=0.5,
        coarse_loss_weight=0.8,
        final_loss_weight=0.2,
        contrastive_loss_weight=0.01,
    ):
        super().__init__()
        
        # Default checkpoints
        default_checkpoints = {
            "vit_b": "./SAM/sam_vit_b_01ec64.pth",
            "vit_l": "./SAM/sam_vit_l_0b3195.pth",
            "vit_h": "./SAM/sam_vit_h_4b8939.pth",
        }
        
        if checkpoint is None and model_type in default_checkpoints:
            checkpoint = default_checkpoints[model_type]
        
        # Build enhanced model
        enhanced_model_type = f"{model_type}_enhanced"
        if enhanced_model_type not in sam_model_registry:
            raise ValueError(f"Model type {enhanced_model_type} not found in registry")
        
        self.sam = sam_model_registry[enhanced_model_type](
            image_size=image_size,
            num_classes=num_classes,
            checkpoint=checkpoint,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
            # LoRA parameters
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # Token accumulation
            use_token_accumulation=use_token_accumulation,
            token_momentum=token_momentum,
            # Loss weights
            coarse_loss_weight=coarse_loss_weight,
            final_loss_weight=final_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
        )
        
        self.image_size = image_size
        self.num_classes = num_classes
    
    def forward(self, x, use_accumulated_tokens=True, return_all_outputs=False):
        """
        Forward pass for enhanced SAM
        
        Args:
            x: Input images [B, 3, H, W]
            use_accumulated_tokens: Whether to use accumulated tokens
            return_all_outputs: Whether to return all intermediate outputs
            
        Returns:
            If return_all_outputs is False: returns final masks
            If return_all_outputs is True: returns dict with all outputs
        """
        outputs = self.sam(
            x, 
            use_accumulated_tokens=use_accumulated_tokens,
            return_all_outputs=return_all_outputs
        )
        
        if return_all_outputs:
            return outputs
        else:
            # Return only the final masks for backward compatibility
            return outputs['coarse_masks']
    
    def get_trainable_params(self):
        """Get trainable parameters for optimizer"""
        return self.sam.get_trainable_params()
    
    def reset_token_accumulator(self):
        """Reset the token accumulator (useful when switching datasets)"""
        if hasattr(self.sam, 'token_accumulator'):
            self.sam.token_accumulator.reset()




