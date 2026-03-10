import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from typing import Union
from typing import Optional
from typing import Callable
from timm.models.vision_transformer import Mlp
from timm.models.vision_transformer import Attention



def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return out


class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim: int):
    super().__init__()

    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

    return embeddings


class SEBlock3D(nn.Module):
    def __init__(self, channels: int,
                 reduction: int = 8):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)

        return x * w


class SEBlock2D(nn.Module):
    def __init__(self, channels: int,
                 reduction: int = 8):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)

        return x * w


class AttentionGate3D(nn.Module):
    def __init__(self, channels_l: int,
                 channels_g: int,
                 inter_channels: int=None):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(channels_l // 2, 1)
        
        self.weights_g = nn.Sequential(
            nn.Conv3d(channels_g, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )

        self.weights_x = nn.Sequential(
            nn.Conv3d(channels_l, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, g: torch.Tensor,
                l: torch.Tensor) -> torch.Tensor:
        """
        g: decoded features
        l: encoder skiped features
        """

        attention = self.psi(self.weights_g(g) + self.weights_x(l))

        out = l * attention

        return out


class AttentionGate2D(nn.Module):
    def __init__(self, channels_l: int,
                 channels_g: int,
                 inter_channels: int=None):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(channels_l // 2, 1)
        
        self.weights_g = nn.Sequential(
            nn.Conv2d(channels_g, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.weights_x = nn.Sequential(
            nn.Conv2d(channels_l, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, g: torch.Tensor,
                l: torch.Tensor) -> torch.Tensor:
        """
        g: decoded features
        l: encoder skiped features
        """

        attention = self.psi(self.weights_g(g) + self.weights_x(l))

        out = l * attention

        return out


class PatchEmbed3D(nn.Module):
    """ 3D Image (volume) to Patch Embedding """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]]=224,
        patch_size: int=16,
        in_chans: int=1,
        embed_dim: int=768,
        norm_layer: Optional[Callable]=None,
        flatten: bool=True,
        bias: bool=True,
        strict_img_size: bool=True,
        dynamic_img_pad: bool=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)

        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)

        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]

        return img_size, grid_size, num_patches

    def _assert(self, cond, msg):
        if not cond:
            raise AssertionError(msg)

    def dynamic_feat_size(self, img_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get grid (feature) size for given image size accounting for dynamic padding."""
        if self.dynamic_img_pad:
            return (
                math.ceil(img_size[0] / self.patch_size[0]),
                math.ceil(img_size[1] / self.patch_size[1]),
                math.ceil(img_size[2] / self.patch_size[2]),
            )
        else:
            return (
                img_size[0] // self.patch_size[0],
                img_size[1] // self.patch_size[1],
                img_size[2] // self.patch_size[2],
            )

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                self._assert(D == self.img_size[0], f'Input depth ({D}) doesnt match model ({self.img_size[0]}).')
                self._assert(H == self.img_size[1], f'Input height ({H}) doesnt match model ({self.img_size[1]}).')
                self._assert(W == self.img_size[2], f'Input width ({W}) doesnt match model ({self.img_size[2]}).')
            elif not self.dynamic_img_pad:
                self._assert(D % self.patch_size[0] == 0,
                        f'Input depth ({D}) should be divisible by patch size ({self.patch_size[0]}).')
                self._assert(H % self.patch_size[1] == 0,
                        f'Input height ({H}) should be divisible by patch size ({self.patch_size[1]}).')
                self._assert(W % self.patch_size[2] == 0,
                        f'Input width ({W}) should be divisible by patch size ({self.patch_size[2]}).')

        if self.dynamic_img_pad:
            pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
            pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
            pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
            # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x = self.proj(x)  # [B, embed_dim, D', H', W']

        self.grid_size = x.shape[2:]  # (D_p, H_p, W_p)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        x = self.norm(x)
        return x


class PatchEmbed2D(nn.Module):
    """ 2D Image (volume) to Patch Embedding """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]]=224,
        patch_size: int=16,
        in_chans: int=1,
        embed_dim: int=768,
        norm_layer: Optional[Callable]=None,
        flatten: bool=True,
        bias: bool=True,
        strict_img_size: bool=True,
        dynamic_img_pad: bool=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)

        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]

        return img_size, grid_size, num_patches

    def _assert(self, cond, msg):
        if not cond:
            raise AssertionError(msg)

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size accounting for dynamic padding."""
        if self.dynamic_img_pad:
            return (
                math.ceil(img_size[0] / self.patch_size[0]),
                math.ceil(img_size[1] / self.patch_size[1]),
            )
        else:
            return (
                img_size[0] // self.patch_size[0],
                img_size[1] // self.patch_size[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                self._assert(H == self.img_size[0], f'Input height ({H}) doesnt match model ({self.img_size[0]}).')
                self._assert(W == self.img_size[1], f'Input width ({W}) doesnt match model ({self.img_size[1]}).')
            elif not self.dynamic_img_pad:
                self._assert(H % self.patch_size[0] == 0,
                        f'Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).')
                self._assert(W % self.patch_size[1] == 0,
                        f'Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).')

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)  # [B, embed_dim, D', H', W']

        self.grid_size = x.shape[2:]  # (H_p, W_p)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        x = self.norm(x)
        return x


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 time_emb_dim: int=None,
                 up: bool=False,
                 down: bool=True,
                 use_se: bool=False):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        if time_emb_dim is not None:
            self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

        if up:
            # up-sampling (decoder part)
            self.conv1 = nn.Conv3d(2 * in_channels, out_channels, 3, padding=1)
            self.transform = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                           nn.Upsample(scale_factor=2, mode='trilinear'))
        else:
            # down-sampling (encoder part)
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            if down:
                self.transform = nn.Conv3d(out_channels, out_channels, 4, 2, 1)
            else:
                self.transform = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_channels)
        self.bnorm2 = nn.BatchNorm3d(out_channels)

        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock3D(out_channels)

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        # First Conv
        h = self.bnorm1(F.silu(self.conv1(x)))    
        
        if self.time_emb_dim is not None:
            # Time embedding
            time_emb = self.time_mlp(t)
            # Extend last 3 dimensions
            time_emb = time_emb[(..., ) + (None, ) * 3]
            
            # Add time channel
            h = h + time_emb

        # Second Conv
        h = self.bnorm2(F.silu(self.conv2(h)))

        if self.use_se:
            h = self.se_block(h)
    
        # Down or Upsample
        out = F.silu(self.transform(h))
    
        return out


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 time_emb_dim: int=None,
                 up: bool=False,
                 down: bool=True,
                 use_se: bool=False):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        if time_emb_dim is not None:
            self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

        if up:
            # up-sampling (decoder part)
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            self.transform = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                           nn.Upsample(scale_factor=2, mode='bilinear'))
        else:
            # down-sampling (encoder part)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            if down:
                self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            else:
                self.transform = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock2D(out_channels)

    def forward(self, x:torch.Tensor, t:torch.Tensor):
        # First Conv
        h = self.bnorm1(F.silu(self.conv1(x)))    
        
        if self.time_emb_dim is not None:
            # Time embedding
            time_emb = self.time_mlp(t)
            # Extend last 2 dimensions
            time_emb = time_emb[(..., ) + (None, ) * 2]
            
            # Add time channel
            h = h + time_emb

        # Second Conv
        h = self.bnorm2(F.silu(self.conv2(h)))

        if self.use_se:
            h = self.se_block(h)
    
        # Down or Upsample
        out = F.silu(self.transform(h))
    
        return out


class DiTBlock(nn.Module):
    """
    Codes are largely taken from 
    https://github.com/facebookresearch/DiT/
    """
    def __init__(self, hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float=4.,
                 **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu,
                       drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(self, x: torch.Tensor,
                ctx: torch.Tensor)-> torch.Tensor:
        components = self.adaLN_modulation(ctx).chunk(6, dim=1)

        shift_msa, scale_msa, gate_msa = components[:3]
        shift_mlp, scale_mlp, gate_mlp = components[3:]

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int,
                 patch_size: int,
                 out_channels: int):
        super().__init__()

        # total_size = grid_size[0] * grid_size[1] * grid_size[2]
        total_size = patch_size * patch_size * patch_size

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, total_size * out_channels)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(ctx).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)

        x = self.linear(x)

        return x


class DiTFinalLayer2D(nn.Module):
    def __init__(self, hidden_size: int,
                 patch_size: int,
                 out_channels: int):
        super().__init__()

        # total_size = grid_size[0] * grid_size[1]
        total_size = patch_size * patch_size

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, total_size * out_channels)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(ctx).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)

        x = self.linear(x)

        return x
