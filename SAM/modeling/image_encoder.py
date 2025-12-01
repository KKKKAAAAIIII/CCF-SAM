
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d
from .adapter_add import LoRAInspiredAdapter
import numpy as np
import math

np.random.seed(42)


# LoRA implementation
class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 16, 
        alpha: float = 16.0,
        dropout: float = 0.2
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [*, in_features]
        # LoRA forward: x @ A @ B * scaling
        result = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return result


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.2
    ):
        super().__init__()
        # Frozen pre-trained weights
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combine frozen weights with LoRA adaptation
        return self.linear(x) + self.lora(x)


class Conv2dWithLoRA(nn.Module):
    """
    Conv2d layer with LoRA adaptation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        self.conv.weight.requires_grad = False
        if bias and self.conv.bias is not None:
            self.conv.bias.requires_grad = False
            
        # LoRA adaptation
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # For conv2d, we use 1x1 convolutions for LoRA to maintain spatial dimensions
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        
        self.scaling = alpha / rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard convolution
        out = self.conv(x)
        
        # LoRA path
        # Apply same padding and stride logic for spatial consistency
        if self.kernel_size > 1:
            # For non-1x1 conv, we need to match the spatial dimensions
            lora_out = self.lora_dropout(x)
            lora_out = F.avg_pool2d(lora_out, self.kernel_size, self.stride, self.padding)
            lora_out = self.lora_A(lora_out)
            lora_out = self.lora_B(lora_out) * self.scaling
        else:
            # For 1x1 conv, direct application
            lora_out = self.lora_dropout(x)
            lora_out = self.lora_A(lora_out)
            lora_out = self.lora_B(lora_out) * self.scaling
        
        return out + lora_out


class MLPBlockWithLoRA(nn.Module):
    """
    MLP block with LoRA adaptation
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_lora = use_lora
        
        if use_lora:
            self.lin1 = LinearWithLoRA(
                embedding_dim, mlp_dim, 
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
            self.lin2 = LinearWithLoRA(
                mlp_dim, embedding_dim,
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
        else:
            self.lin1 = nn.Linear(embedding_dim, mlp_dim)
            self.lin2 = nn.Linear(mlp_dim, embedding_dim)
            
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        # LoRA parameters
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Tuple[str, ...] = ('qkv', 'proj', 'mlp', 'neck'),
    ) -> None:
        """
        Args:
            ... (existing args)
            use_lora (bool): Whether to use LoRA adaptation
            lora_rank (int): LoRA rank
            lora_alpha (float): LoRA alpha scaling parameter
            lora_dropout (float): Dropout rate for LoRA
            lora_target_modules (tuple): Which modules to apply LoRA to
                Options: 'qkv', 'proj', 'mlp', 'neck'
        """
        super().__init__()
        self.img_size = img_size
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                # LoRA parameters
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
            )
            self.blocks.append(block)

        # Neck with optional LoRA
        if use_lora and 'neck' in lora_target_modules:
            self.neck = nn.Sequential(
                Conv2dWithLoRA(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                ),
                LayerNorm2d(out_chans),
                Conv2dWithLoRA(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                ),
                LayerNorm2d(out_chans),
            )
        else:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )
        self.adapter_sa = nn.Linear(embed_dim,embed_dim//32)
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False
           
        # Unfreeze LoRA parameters
        if use_lora:
            for name, param in self.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        for name, param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        num=0
        for blk in self.blocks:
            num += 1
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        # LoRA parameters
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Tuple[str, ...] = ('qkv', 'proj', 'mlp', 'neck'),
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            # LoRA parameters
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        self.adapter = LoRAInspiredAdapter(embedding_dim=dim)
        self.norm2 = norm_layer(dim)
        
        # MLP with optional LoRA
        if use_lora and 'mlp' in lora_target_modules:
            self.mlp = MLPBlockWithLoRA(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                act=act_layer,
                use_lora=False,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            self.mlp = MLPBlockWithLoRA(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                act=act_layer,
                use_lora=False,
            )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
       
        adapt_x = self.adapter(x, add_residual=True)
        x = adapt_x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        # LoRA parameters
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Tuple[str, ...] = ('qkv', 'proj', 'mlp', 'neck'),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_lora = use_lora
        self.lora_target_modules = lora_target_modules

        # QKV projection with optional LoRA
        if use_lora and 'qkv' in lora_target_modules:
            self.qkv = LinearWithLoRA(
                dim, dim * 3, bias=qkv_bias,
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Output projection with optional LoRA
        if use_lora and 'proj' in lora_target_modules:
            self.proj = LinearWithLoRA(
                dim, dim, bias=True,
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
        else:
            self.proj = nn.Linear(dim, dim)
        # self.adapter_deep = Adapter_IPF(dim,dim//32)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor,deep_adapt=None) -> torch.Tensor:
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        # x = self.adapter_deep(x,deep_adapt)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


