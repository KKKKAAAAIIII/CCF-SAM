
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from functools import partial
from typing import Any, Dict, List, Tuple
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .feature_extraction import (
    FeatureSeparationModule, 
    SimpleContrastiveLoss,
    TokenAccumulator,
    CrossAttentionModule
)


class SAMWithTokenEnhancement(nn.Module):
    """
    扩展的SAM模型，包含token增强机制
    实现了完整的训练pipeline：
    1. Image → Image Encoder → Image Embeddings
    2. Image Embeddings → Mask Decoder → Coarse Mask
    3. Coarse Mask → Feature Separation → Target/BG Tokens
    4. Tokens + Image Embeddings → Cross Attention → Updated Embeddings
    5. Updated Embeddings → Mask Decoder → Final Mask
    6. Token Accumulation across images
    """
    
    def __init__(
        self,
        image_encoder_config: Optional[Dict] = None,
        mask_decoder_config: Optional[Dict] = None,
        prompt_encoder_config: Optional[Dict] = None,
        feature_separator_config: Optional[Dict] = None,
        cross_attention_config: Optional[Dict] = None,
        # Token accumulation
        use_token_accumulation: bool = True,
        token_momentum: float = 0.5,
        # Loss weights
        coarse_loss_weight: float = 0.8,
        final_loss_weight: float = 0.2,
        contrastive_loss_weight: float = 0.01,
        # LoRA config for image encoder
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Tuple[str, ...] = ('qkv', 'proj', 'mlp', 'neck'),
    ):
        super().__init__()
        
        # Default configs
        # self.image_encoder_config = image_encoder_config
        
        
        # self.mask_decoder_config = mask_decoder_config
        
        # self.prompt_encoder_config = prompt_encoder_config

        # self.feature_separator_config = feature_separator_config

        # self.cross_attention_config = cross_attention_config

        # Initialize components
        self.image_encoder = ImageEncoderViT(**image_encoder_config)
        self.prompt_encoder = PromptEncoder(**prompt_encoder_config)
        
        # We need the transformer for mask decoder
        # from .transformer import TwoWayTransformer
        # transformer = TwoWayTransformer(
        #     depth=2,
        #     embedding_dim=mask_decoder_config['transformer_dim'],
        #     mlp_dim=2048,
        #     num_heads=8,
        # )
        # mask_decoder_config['transformer'] = transformer
        self.mask_decoder = MaskDecoder(**mask_decoder_config)
        
        # New components
        self.feature_separator = FeatureSeparationModule(**feature_separator_config)
        self.cross_attention = CrossAttentionModule(**cross_attention_config)
        
        # Token accumulation
        self.use_token_accumulation = use_token_accumulation
        if use_token_accumulation:
            # 旧的: self.token_accumulator = TokenAccumulator(...)
            # 新的:
            self.token_accumulator = TokenAccumulator(
                token_dim=feature_separator_config['token_dim'],
                momentum=token_momentum
            )
        
        # Loss functions
        self.contrastive_loss_fn = SimpleContrastiveLoss(temperature=0.1)
        
        # Loss weights
        self.coarse_loss_weight = coarse_loss_weight
        self.final_loss_weight = final_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        # Register buffers for normalization
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        
    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_input, use_accumulated_tokens, return_all_outputs):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input,  use_accumulated_tokens, return_all_outputs)
        else:
            outputs = self.forward_train(batched_input,  use_accumulated_tokens, return_all_outputs)
        return outputs
    
    def forward_train(
        self,
        images: torch.Tensor,
        use_accumulated_tokens: bool = True,
        return_all_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播流程
        
        Args:
            images: Input images [B, 3, H, W]
            use_accumulated_tokens: Whether to use accumulated tokens from previous images
            return_all_outputs: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing masks, tokens, losses, etc.
        """
        # 1. Preprocess and encode images
        images = self.preprocess(images)
        image_embeddings = self.image_encoder(images)  # [B, C, H, W]
        
        # 2. Generate coarse masks (first pass through decoder)
        # No prompts for automatic segmentation
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, 
            boxes=None, 
            masks=None
        )
        
        coarse_mask, coarse_iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
        coarse_masks = self.postprocess_masks(
            coarse_mask,
            input_size=(256, 256),
            original_size=(256, 256)
        )
        # 3. Feature separation and token generation
        # Use the first mask for binary segmentation
        separation_output = self.feature_separator(
            coarse_mask[:, 0:1, :, :],
            image_embeddings=image_embeddings.detach()
        )
        
        target_token = separation_output['target_token']
        bg_token = separation_output['background_token']
        contrastive_loss = self.contrastive_loss_fn(target_token, bg_token)
        # 4. Token enhancement with accumulation
        if use_accumulated_tokens and self.use_token_accumulation:
            # 新的调用方式：直接将当前token传入，获取更新后的增强token
            enhanced_target, enhanced_bg = self.token_accumulator(
                target_token,
                bg_token
            )
        else:
            # 如果不使用累积，则直接使用当前token
            enhanced_target = target_token
            enhanced_bg = bg_token
        
        # contrastive_loss = self.contrastive_loss_fn(enhanced_target, enhanced_bg)
        # 5. Cross attention to enhance image embeddings
        updated_embeddings = self.cross_attention(
            image_embeddings,
            enhanced_target,
            enhanced_bg
        )
        
        # 6. Generate final masks with enhanced embeddings
        final_mask, final_iou_predictions = self.mask_decoder(
            image_embeddings=updated_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )

        final_masks = self.postprocess_masks(
            final_mask,
            input_size=(256, 256),
            original_size=(256, 256)
        )
        
        # Prepare outputs
        outputs = {
            'final_masks': final_masks,
            'coarse_masks': coarse_masks,
            'contrastive_loss': contrastive_loss,
            'coarse_iou_predictions': coarse_iou_predictions,
            'final_iou_predictions': final_iou_predictions,
        }
        
        if return_all_outputs:
            outputs.update({
                'target_token': target_token,
                'background_token': bg_token,
                'enhanced_target_token': enhanced_target,
                'enhanced_background_token': enhanced_bg,
        
                'image_embeddings': image_embeddings,
                'updated_embeddings': updated_embeddings,
                'coarse_iou_predictions': coarse_iou_predictions,
            'final_iou_predictions': final_iou_predictions
            })
        
        return outputs
    @torch.no_grad()
    def forward_test(
        self,
        images: List[Dict[str, Any]],
        use_accumulated_tokens: bool = True,
        return_all_outputs: bool = True,
    ) ->List[Dict[str, torch.Tensor]]:
        """
        完整的前向传播流程
        
        Args:
            images: Input images [B, 3, H, W]
            use_accumulated_tokens: Whether to use accumulated tokens from previous images
            return_all_outputs: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing masks, tokens, losses, etc.
        """
        # 1. Preprocess and encode images
        images = self.preprocess(images)
        image_embeddings = self.image_encoder(images)  # [B, C, H, W]
        outputs = []
        for image_record, curr_embedding in zip(images, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            coarse_mask, coarse_iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            coarse_masks = self.postprocess_masks(
                coarse_mask,
                input_size=(256, 256),
                original_size=(256, 256)
            )
            separation_output = self.feature_separator(
            coarse_mask[:, 0:1, :, :],
            image_embeddings=image_embeddings.detach()
        )
        
            target_token = separation_output['target_token']
            bg_token = separation_output['background_token']
            
        # 4. Token enhancement with accumulation
            if use_accumulated_tokens and self.use_token_accumulation:
            # 新的调用方式：直接将当前token传入，获取更新后的增强token
                enhanced_target, enhanced_bg = self.token_accumulator(
                target_token,
                bg_token
            )
            else:
            # 如果不使用累积，则直接使用当前token
                enhanced_target = target_token
                enhanced_bg = bg_token
        
        # contrastive_loss = self.contrastive_loss_fn(enhanced_target, enhanced_bg)
        # 5. Cross attention to enhance image embeddings
            updated_embeddings = self.cross_attention(
            image_embeddings,
            enhanced_target,
            enhanced_bg
        )
        
        # 6. Generate final masks with enhanced embeddings
            final_mask, final_iou_predictions = self.mask_decoder(
            image_embeddings=updated_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )

            final_masks = self.postprocess_masks(
            final_mask,
            input_size=(256, 256),
            original_size=(256, 256)
        )
            final_masks = final_masks > 0.0
            outputs.append(
                {
                    "masks": final_masks,
                    "iou_predictions": final_iou_predictions,
                    "low_res_logits": final_iou_predictions,
                }
            )
        return outputs
        # 2. Generate coarse masks (first pass through decoder)
        # No prompts for automatic segmentation
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=None, 
        #     boxes=None, 
        #     masks=None
        # )
        
        # coarse_mask, coarse_iou_predictions = self.mask_decoder(
        #     image_embeddings=image_embeddings,
        #     image_pe=self.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=True
        # )
        # coarse_masks = self.postprocess_masks(
        #     coarse_mask,
        #     input_size=(256, 256),
        #     original_size=(256, 256)
        # )
        # # 3. Feature separation and token generation
        # # Use the first mask for binary segmentation
        # separation_output = self.feature_separator(
        #     coarse_mask[:, 0:1, :, :],
        #     image_embeddings=image_embeddings.detach()
        # )
        
        # target_token = separation_output['target_token']
        # bg_token = separation_output['background_token']
        # contrastive_loss = self.contrastive_loss_fn(target_token, bg_token)
        # # 4. Token enhancement with accumulation
        # if use_accumulated_tokens and self.use_token_accumulation:
        #     # 新的调用方式：直接将当前token传入，获取更新后的增强token
        #     enhanced_target, enhanced_bg = self.token_accumulator(
        #         target_token,
        #         bg_token
        #     )
        # else:
        #     # 如果不使用累积，则直接使用当前token
        #     enhanced_target = target_token
        #     enhanced_bg = bg_token
        
        # # contrastive_loss = self.contrastive_loss_fn(enhanced_target, enhanced_bg)
        # # 5. Cross attention to enhance image embeddings
        # updated_embeddings = self.cross_attention(
        #     image_embeddings,
        #     enhanced_target,
        #     enhanced_bg
        # )
        
        # # 6. Generate final masks with enhanced embeddings
        # final_mask, final_iou_predictions = self.mask_decoder(
        #     image_embeddings=updated_embeddings,
        #     image_pe=self.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=True
        # )

        # final_masks = self.postprocess_masks(
        #     final_mask,
        #     input_size=(256, 256),
        #     original_size=(256, 256)
        # )

        # # Prepare outputs
        # outputs = {
        #     'final_masks': final_masks,
        #     'coarse_masks': coarse_masks,
        #     'contrastive_loss': contrastive_loss,
        #     'coarse_iou_predictions': coarse_iou_predictions,
        #     'final_iou_predictions': final_iou_predictions,
        # }
        
        # if return_all_outputs:
        #     outputs.update({
        #         'target_token': target_token,
        #         'background_token': bg_token,
        #         'enhanced_target_token': enhanced_target,
        #         'enhanced_background_token': enhanced_bg,
        
        #         'image_embeddings': image_embeddings,
        #         'updated_embeddings': updated_embeddings,
        #         'coarse_iou_predictions': coarse_iou_predictions,
        #     'final_iou_predictions': final_iou_predictions
        #     })
        
        # return outputs
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    
    def get_trainable_params(self):
        """Get all trainable parameters for optimizer"""
        trainable_params = []
        
        # LoRA parameters from image encoder
        for name, param in self.image_encoder.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # All parameters from new modules
        trainable_params.extend(self.feature_separator.parameters())
        trainable_params.extend(self.cross_attention.parameters())
        
        # Optionally, you can also fine-tune mask decoder
        trainable_params.extend(self.mask_decoder.parameters())
        trainable_params.extend(self.prompt_encoder.parameters())
        
        return trainable_params

