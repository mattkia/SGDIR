"""Implementations of 2D and 3D UNet and UNet+DiT backbones
used in SGDIR and SGDIRDiT
"""

import torch

import torch.nn as nn


from typing import Union

from models.modules import DiTBlock
from models.modules import ConvBlock3D
from models.modules import ConvBlock2D
from models.modules import PatchEmbed3D
from models.modules import PatchEmbed2D
from models.modules import DiTFinalLayer
from models.modules import DiTFinalLayer2D
from models.modules import AttentionGate3D
from models.modules import AttentionGate2D
from models.modules import SinusoidalPositionEmbeddings

from utils import get_3d_sincos_pos_embed
from utils import get_2d_sincos_pos_embed



class UNet2D(nn.Module):
    """The implementation of 2D UNet backbone used in 2D SGDIR
    """
    def __init__(
        self,
        in_channels: int=2,
        out_channels: int=2, 
        down_channels: tuple=(32, 32, 32), 
        up_channels: tuple=(32, 32, 32), 
        time_emb_dim: int=32,
        decoder_only: bool=True,
        use_attention_gates: bool=True,
        use_se_attention: bool=False
    ) -> None:
        """
        :param in_channels:
            The combined channels of fixed and moving images

        :param out_channels:
            The dimension of output deformation (must be 2)

        :param down_channels:
            The channels used for the encoder

        :param up_channels: 
            The channels used for the decoder (currently the up_channels must
            be down_channels in reverse order)

        :param time_emb_dim:
            The dimension of time embedding

        :param decoder_only:
            If true, only the decoder becomes time-embedded (recommended)

        :param use_attention_gates:
            If true, skip connections are equipped with attention gates

        :param use_se_attention:
            If true, squeeze-and-excitation modules will be used
        """
        super().__init__()

        self.use_attn_gate = use_attention_gates
    
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList()

        encoder_time_dim = None if decoder_only else time_emb_dim
        for i in range(len(down_channels) - 1):
            block = ConvBlock2D(down_channels[i],
                                down_channels[i + 1],
                                encoder_time_dim,
                                False,
                                True,
                                use_se_attention)
            self.downs.append(block)

        # Upsample
        self.ups = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            block = ConvBlock2D(up_channels[i],
                                up_channels[i + 1],
                                time_emb_dim,
                                True,
                                False,
                                use_se_attention)
            self.ups.append(block)

        # Add attention gates
        if self.use_attn_gate:
            self.attn_gates = nn.ModuleList()

            for i in range(len(self.ups)):
                block = AttentionGate2D(channels_l=down_channels[-i - 1],
                                        channels_g=up_channels[i],
                                        inter_channels=max(up_channels[i] // 2, 1))
                self.attn_gates.append(block)

        # Final projection
        self.output = nn.Conv2d(up_channels[-1], out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: 
            The concatenation of fixed and moving images on channel dim
            Expected shape is [B, 2C, H, W]
        :param timestep: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        :returns:
            Displacement field at time timestep
            Expected shape is [B, 2, H, W]
        """
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Apply attention gate if set true
            if self.use_attn_gate:
                residual_x = self.attn_gates[i](x, residual_x)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


class UNet3D(nn.Module):
    """The implementation of 3D UNet backbone used in 3D SGDIR
    """
    def __init__(
        self,
        in_channels: int=2,
        out_channels: int=3, 
        down_channels: tuple=(32, 32, 32), 
        up_channels: tuple=(32, 32, 32), 
        time_emb_dim: int=32,
        decoder_only: bool=True,
        use_attention_gates: bool=True,
        use_se_attention: bool=False
    ) -> None:
        """
        :param in_channels:
            The combined channels of fixed and moving images

        :param out_channels:
            The dimension of output deformation (must be 3)

        :param down_channels:
            The channels used for the encoder

        :param up_channels: 
            The channels used for the decoder (currently the up_channels must
            be down_channels in reverse order)

        :param time_emb_dim:
            The dimension of time embedding

        :param decoder_only:
            If true, only the decoder becomes time-embedded (recommended)

        :param use_attention_gates:
            If true, skip connections are equipped with attention gates

        :param use_se_attention:
            If true, squeeze-and-excitation modules will be used
        """
        super().__init__()

        self.use_attn_gate = use_attention_gates
    
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv3d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList()

        encoder_time_dim = None if decoder_only else time_emb_dim
        for i in range(len(down_channels) - 1):
            block = ConvBlock3D(down_channels[i],
                                down_channels[i + 1],
                                encoder_time_dim,
                                False,
                                True,
                                use_se_attention)
            self.downs.append(block)

        # Upsample
        self.ups = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            block = ConvBlock3D(up_channels[i],
                                up_channels[i + 1],
                                time_emb_dim,
                                True,
                                False,
                                use_se_attention)
            self.ups.append(block)

        # Add attention gates
        if self.use_attn_gate:
            self.attn_gates = nn.ModuleList()

            for i in range(len(self.ups)):
                block = AttentionGate3D(channels_l=down_channels[-i - 1],
                                        channels_g=up_channels[i],
                                        inter_channels=max(up_channels[i] // 2, 1))
                self.attn_gates.append(block)

        # Final projection
        self.output = nn.Conv3d(up_channels[-1], out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: 
            The concatenation of fixed and moving images on channel dim
            Expected shape is [B, 2C, D, H, W]

        :param timestep: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]

        :returns:
            Displacement field at time timestep
            Expected shape is [B, 3, D, H, W]
        """
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Apply attention gate if set true
            if self.use_attn_gate:
                residual_x = self.attn_gates[i](x, residual_x)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


class DiT2D(nn.Module):
    """Implementation of 2D DiT used in LatentDiT2D

    Disclaimer: This code is inspired by
        https://github.com/facebookresearch/DiT
    """
    def __init__(
        self,
        input_size: Union[int, tuple[int, int]]=32,
        patch_size: int=2,
        in_channels: int=4,
        hidden_size: int=1152,
        depth: int=28,
        num_heads: int=16,
        mlp_ratio: float=4.
    ):
        """
        :param input_size:
            The spatial dimensions of the input
        
        :param patch_size:
            The patch size for patch embedding module

        :param in_channels:
            The number of input channels

        :param depth:
            The number of DiT layers

        :param num_heads:
            The number of heads in multi-head attention modules

        :param mlp_ratio:
            The mlp hidden layer ratio of the DiT modules
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed2D(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList()

        for _ in range(depth):
            block = DiTBlock(hidden_size, num_heads, mlp_ratio)
            self.blocks.append(block)
        
        self.final_layer = DiTFinalLayer2D(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_unit(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
        
        self.apply(_basic_unit)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # zero out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out otuput layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x:
            [N, T, pH * pW * C]

        :returns:
            imgs [N, C, H, W]
        """
        c = self.out_channels
        pH, pW = self.x_embedder.patch_size
        H_p, W_p = self.x_embedder.grid_size

        # sanity check
        assert H_p * W_p == x.shape[1], f'Expected {H_p*W_p} patches, got {x.shape[1]}'
        assert x.shape[2] == pH * pW * c, f'Last dim should be {pH*pW*c}, got {x.shape[2]}'

        # reshape into grid of patches
        x = x.reshape(x.shape[0], H_p, W_p, pH, pW, c)  # (N, H_p, W_p, pH, pW, C)

        # bring channels up front
        x = x.permute(0, 5, 1, 3, 2, 4)  # (N, C, H_p, pH, W_p, pW)

        # merge patch grids
        imgs = x.reshape(x.shape[0], c, H_p * pH, W_p * pW)

        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: 
            The input to the DiT layer
            Expected shape is [B, C, H, W]

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.t_embedder(t)                   # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x


class DiT3D(nn.Module):
    """Implementation of 3D DiT used in LatentDiT3D

    Disclaimer: This code is inspired by
        https://github.com/facebookresearch/DiT
    """
    def __init__(
        self,
        input_size: Union[int, tuple[int, int, int]]=32,
        patch_size: int=2,
        in_channels: int=4,
        hidden_size: int=1152,
        depth: int=28,
        num_heads: int=16,
        mlp_ratio: float=4.
    ):
        """
        :param input_size:
            The spatial dimensions of the input
        
        :param patch_size:
            The patch size for patch embedding module

        :param in_channels:
            The number of input channels

        :param depth:
            The number of DiT layers

        :param num_heads:
            The number of heads in multi-head attention modules

        :param mlp_ratio:
            The mlp hidden layer ratio of the DiT modules
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList()

        for _ in range(depth):
            block = DiTBlock(hidden_size, num_heads, mlp_ratio)
            self.blocks.append(block)
        
        self.final_layer = DiTFinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_unit(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
        
        self.apply(_basic_unit)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # zero out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out otuput layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x:
            [N, T, pD * pH * pW * C]

        :returns:
            imgs [N, C, D, H, W]
        """
        c = self.out_channels
        pD, pH, pW = self.x_embedder.patch_size
        D_p, H_p, W_p = self.x_embedder.grid_size

        # sanity check
        assert D_p * H_p * W_p == x.shape[1], f'Expected {D_p*H_p*W_p} patches, got {x.shape[1]}'
        assert x.shape[2] == pD * pH * pW * c, f'Last dim should be {pD*pH*pW*c}, got {x.shape[2]}'

        # reshape into grid of patches
        x = x.reshape(x.shape[0], D_p, H_p, W_p, pD, pH, pW, c)  # (N, D_p, H_p, W_p, pD, pH, pW, C)

        # bring channels up front
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (N, C, D_p, pD, H_p, pH, W_p, pW)

        # merge patch grids
        imgs = x.reshape(x.shape[0], c, D_p * pD, H_p * pH, W_p * pW)

        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: 
            The input to the DiT layer
            Expected shape is [B, C, D, H, W]

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = D * H * W / patch_size ** 3
        c = self.t_embedder(t)                   # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 3 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, D, H, W)

        return x


class LatentDiT2D(nn.Module):
    """Implementation of UNet encoder decoder + DiT latents
    used in 2D SGDIRDiT
    """
    def __init__(
        self,
        image_size: tuple[int],
        in_channels: int=2,
        out_channels: int=2,
        down_channels: tuple=(32, 32, 32),
        use_attention_gates: bool=True,
        use_se_attention: bool=True,
        decoder_time_embed_dim: int=None,
        patch_size: int=1,
        embed_dim: int=384,
        depth: int=4,
        num_heads: int=16,
        mlp_ratio: float=4.
    ):
        """
        :param image_size:
            The spatial dimensions of the image
            Used to compute the input size of the DiT

        :param in_channels:
            The combined channels of fixed and moving images

        :param out_channels:
            The dimension of output deformation (must be 2)
        
        :param down_channels:
            The channels used for the UNet encoder

        :param up_channels: 
            The channels used for the decoder (currently the up_channels must
            be down_channels in reverse order)
        
        :param use_attention_gates:
            If true, skip connections are equipped with attention gates

        :param use_se_attention:
            If true, squeeze-and-excitation modules will be used

        :param decoder_time_embed_dim:
            If set, UNet decoder will be equipped with time-embedding too
        
        :param patch_size:
            The patch size for patch embedding module

        :param embed_dim:
            The number of input channels to the DiT

        :param depth:
            The number of DiT layers

        :param num_heads:
            The number of heads in multi-head attention modules

        :param mlp_ratio:
            The mlp hidden layer ratio of the DiT modules
        """
        super().__init__()

        up_channels = down_channels[::-1]

        self.use_attn_gates = use_attention_gates
        self.decoder_time_embed_dim = decoder_time_embed_dim

        # constructing the decoder time embedding
        if decoder_time_embed_dim is not None:
            self.time_embedder = nn.Sequential(
                SinusoidalPositionEmbeddings(decoder_time_embed_dim),
                nn.Linear(decoder_time_embed_dim, decoder_time_embed_dim),
                nn.SiLU()
            )

        # constructing the encoder
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3,padding=1)

        self.downs = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            block = ConvBlock2D(down_channels[i],
                                down_channels[i + 1],
                                None,
                                False,
                                True,
                                use_se_attention)
            self.downs.append(block)

        # construct the DiT
        divider = 2 ** (len(down_channels) - 1)
        input_size = (image_size[0] // divider,
                      image_size[1] // divider)
        
        self.dit = DiT2D(input_size=input_size,
                         patch_size=patch_size,
                         in_channels=down_channels[-1],
                         hidden_size=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio)
        
        # construct the decoder
        self.ups = nn.ModuleList()
        
        for i in range(len(up_channels) - 1):
            block = ConvBlock2D(up_channels[i],
                                up_channels[i + 1],
                                decoder_time_embed_dim,
                                True,
                                False,
                                use_se_attention)
            self.ups.append(block)
        
        # adding the attention gates
        if use_attention_gates:
            self.attn_gates = nn.ModuleList()

            for i in range(len(self.ups)):
                block = AttentionGate2D(channels_l=down_channels[-i - 1],
                                        channels_g=up_channels[i],
                                        inter_channels=max(up_channels[i] // 2, 1))
                self.attn_gates.append(block)
        
        # final projection layer
        self.output = nn.Conv2d(up_channels[-1],
                                out_channels,
                                kernel_size=3,
                                padding=1)
    
    def encode(
        self,
        x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Implements the encoder of the UNet

        :param x:
            The concatenation of fixed and moving images on channel dim
            Expected shape is [B, 2C, H, W]

        :returns:
            The list of outputs of each encoder layer
        """
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, None)
            residual_inputs.append(x)

        return residual_inputs
    
    def bottleneck(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Implements the DiT bottleneck

        :param x:
            The output of last layer of encoder

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        """
        out = self.dit(x, t)

        return out
    
    def decode(
        self,
        x: torch.Tensor,
        residuals: list[torch.Tensor],
        t: torch.Tensor=None
    ) -> torch.Tensor:
        """Implements the decoder of the UNet

        :param x:
            The output of the last DiT module

        :param residuals:
            The list of encoder outputs used for skip connection

        :param t:
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        """
        decoder_time = None
        if self.decoder_time_embed_dim is not None:
            decoder_time = self.time_embedder(t)

        for i, up in enumerate(self.ups):
            residual_x = residuals.pop()

            # Apply attention gate if set true
            if self.use_attn_gates:
                residual_x = self.attn_gates[i](x, residual_x)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, decoder_time)
        
        out = self.output(x)

        return out

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Implements the end to end forward call of 2D SGDIRDiT

        :param x: 
            The concatenation of fixed and moving images on channel dim
            Expected shape is [B, 2C, H, W]

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]

        :returns:
            Displacement field at time t
            Expected shape is [B, 2, H, W]
        """
        encodings = self.encode(x)

        dit_out = self.bottleneck(encodings[-1], t)

        decoded = self.decode(dit_out, encodings, t)

        return decoded


class LatentDiT3D(nn.Module):
    """Implementation of UNet encoder decoder + DiT latents
    used in 3D SGDIRDiT
    """
    def __init__(
        self,
        image_size: tuple[int],
        in_channels: int=2,
        out_channels: int=3,
        down_channels: tuple=(32, 32, 32),
        use_attention_gates: bool=True,
        use_se_attention: bool=True,
        decoder_time_embed_dim: int=None,
        patch_size: int=1,
        embed_dim: int=384,
        depth: int=4,
        num_heads: int=16,
        mlp_ratio: float=4.
    ):
        """
        :param image_size:
            The spatial dimensions of the image
            Used to compute the input size of the DiT

        :param in_channels:
            The combined channels of fixed and moving images

        :param out_channels:
            The dimension of output deformation (must be 3)
        
        :param down_channels:
            The channels used for the UNet encoder

        :param up_channels: 
            The channels used for the decoder (currently the up_channels must
            be down_channels in reverse order)
        
        :param use_attention_gates:
            If true, skip connections are equipped with attention gates

        :param use_se_attention:
            If true, squeeze-and-excitation modules will be used

        :param decoder_time_embed_dim:
            If set, UNet decoder will be equipped with time-embedding too
        
        :param patch_size:
            The patch size for patch embedding module

        :param embed_dim:
            The number of input channels to the DiT

        :param depth:
            The number of DiT layers

        :param num_heads:
            The number of heads in multi-head attention modules

        :param mlp_ratio:
            The mlp hidden layer ratio of the DiT modules
        """
        super().__init__()

        up_channels = down_channels[::-1]

        self.use_attn_gates = use_attention_gates
        self.decoder_time_embed_dim = decoder_time_embed_dim

        # constructing the decoder time embedding
        if decoder_time_embed_dim is not None:
            self.time_embedder = nn.Sequential(
                SinusoidalPositionEmbeddings(decoder_time_embed_dim),
                nn.Linear(decoder_time_embed_dim, decoder_time_embed_dim),
                nn.SiLU()
            )

        # constructing the encoder
        self.conv0 = nn.Conv3d(in_channels, down_channels[0], 3,padding=1)

        self.downs = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            block = ConvBlock3D(down_channels[i],
                                down_channels[i + 1],
                                None,
                                False,
                                True,
                                use_se_attention)
            self.downs.append(block)

        # construct the DiT
        divider = 2 ** (len(down_channels) - 1)
        input_size = (image_size[0] // divider,
                      image_size[1] // divider,
                      image_size[2] // divider)
        
        self.dit = DiT3D(input_size=input_size,
                         patch_size=patch_size,
                         in_channels=down_channels[-1],
                         hidden_size=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio)
        
        # construct the decoder
        self.ups = nn.ModuleList()
        
        for i in range(len(up_channels) - 1):
            block = ConvBlock3D(up_channels[i],
                                up_channels[i + 1],
                                decoder_time_embed_dim,
                                True,
                                False,
                                use_se_attention)
            self.ups.append(block)
        
        # adding the attention gates
        if use_attention_gates:
            self.attn_gates = nn.ModuleList()

            for i in range(len(self.ups)):
                block = AttentionGate3D(channels_l=down_channels[-i - 1],
                                        channels_g=up_channels[i],
                                        inter_channels=max(up_channels[i] // 2, 1))
                self.attn_gates.append(block)
        
        # final projection layer
        self.output = nn.Conv3d(up_channels[-1],
                                out_channels,
                                kernel_size=3,
                                padding=1)
    
    def encode(
        self,
        x: torch.Tensor
    ) -> list[torch.Tensor]:
        """Implements the encoder of the UNet

        :param x:
            The concatenation of fixed and moving images on channel dim
            Expected shape is [B, 2C, D, H, W]

        :returns:
            The list of outputs of each encoder layer
        """
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, None)
            residual_inputs.append(x)

        return residual_inputs
    
    def bottleneck(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Implements the DiT bottleneck

        :param x:
            The output of last layer of encoder

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        """
        out = self.dit(x, t)

        return out
    
    def decode(
        self,
        x: torch.Tensor,
        residuals: list[torch.Tensor],
        t: torch.Tensor=None
    ) -> torch.Tensor:
        """Implements the decoder of the UNet

        :param x:
            The output of the last DiT module

        :param residuals:
            The list of encoder outputs used for skip connection

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]
        """
        decoder_time = None
        if self.decoder_time_embed_dim is not None:
            decoder_time = self.time_embedder(t)

        for i, up in enumerate(self.ups):
            residual_x = residuals.pop()

            # Apply attention gate if set true
            if self.use_attn_gates:
                residual_x = self.attn_gates[i](x, residual_x)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, decoder_time)
        
        out = self.output(x)

        return out

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Implements the end to end forward call of 3D SGDIRDiT

        :param x: 
            The concatenation of fixed and moving images on channel dim
            Expected shape is [B, 2C, D, H, W]

        :param t: 
            The sampled timestep in [0, 1] interval
            Expected shape is [B]

        :returns:
            Displacement field at time t
            Expected shape is [B, 3, D, H, W]
        """
        encodings = self.encode(x)

        dit_out = self.bottleneck(encodings[-1], t)

        decoded = self.decode(dit_out, encodings, t)

        return decoded
