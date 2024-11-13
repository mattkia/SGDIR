import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from typing import List

from utils import NCCLoss


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


class Block3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int=None, up: bool=False):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        if time_emb_dim is not None:
            self.time_mlp =  nn.Linear(time_emb_dim, out_channels)

        if up:
            # up-sampling (decoder part)
            self.conv1 = nn.Conv3d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                       nn.Upsample(scale_factor=2, mode='trilinear'))
        else:
            # down-sampling (encoder part)
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv3d(out_channels, out_channels, 4, 2, 1)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_channels)
        self.bnorm2 = nn.BatchNorm3d(out_channels)

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
    
        # Down or Upsample
        out = F.silu(self.transform(h))
    
        return out


class UNet3d(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, in_channels: int=1, out_channels: int=1, 
                 down_channels: List|Tuple=(32, 32, 32), 
                 up_channels: List|Tuple=(32, 32, 32), 
                 time_emb_dim: int=32,
                 decoder_only: bool=True) -> None:
        super().__init__()
    
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Initial projection
        self.conv0 = nn.Conv3d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        if decoder_only:
            self.downs = nn.ModuleList([Block3d(down_channels[i], down_channels[i+1], None) for i in range(len(down_channels)-1)])
        else:
            self.downs = nn.ModuleList([Block3d(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])

        # Upsample
        self.ups = nn.ModuleList([Block3d(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])

        # Final projection
        self.output = nn.Conv3d(up_channels[-1], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


class FlowNet3D(nn.Module):
    """
    Implementation of both Time-Independet and Time-Dependent Phi network based on interpolative cmposition
    """
    def __init__(self, 
                 down_channels: List|Tuple=(32, 32, 32),
                 up_channels: List|Tuple=(32, 32, 32), 
                 time_emb_dim: int=64, 
                 loss_type: str='ncc',
                 decoder_only: bool=True) -> None:
        super().__init__()

        self.loss_type = loss_type
        self.ncc_loss = NCCLoss(win=11)
        self.net = UNet3d(in_channels=2, 
                          out_channels=3, 
                          down_channels=down_channels, 
                          up_channels=up_channels, 
                          time_emb_dim=time_emb_dim,
                          decoder_only=decoder_only)
  
    def forward(self, I: torch.Tensor, J: torch.Tensor, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            t (torch.Tensor): sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, D, H, W, 3]
        """
        flow = t * self.velocity(I, J, t)

        phi_t = self.make_grid(flow, xyz)

        return phi_t
  
    def velocity(self, I: torch.Tensor, J: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            t (torch.Tensor): sampled time with size [1]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([I, J], dim=1)

        velocity = self.net(u_in, t)

        return velocity

    def loss_flow(self, I: torch.Tensor, J: torch.Tensor, xyz: torch.Tensor, res: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        t = torch.rand(1, device=I.device)

        flow_J = t * self.velocity(I, J, t)
        Jw = self.warp(J, flow_J, xyz)

        flow_I = (t - 1.) * self.velocity(I, J, t - 1.)
        Iw = self.warp(I, flow_I, xyz)

        if res != 1:
            Iw = F.interpolate(Iw, scale_factor=res, mode='trilinear')
            Jw = F.interpolate(Jw, scale_factor=res, mode='trilinear')

        if self.loss_type == 'mse':
            image_loss = res * F.mse_loss(Jw, Iw)
        else:
            image_loss = res * self.ncc_loss(Jw, Iw)

        flow_I_J = self.compose(flow_I, flow_J, xyz)
        grid_I_J = self.make_grid(flow_I_J, xyz)
        flow_J_I = self.compose(flow_J, flow_I, xyz)
        grid_J_I = self.make_grid(flow_J_I, xyz)
        flow = (2. * t - 1.) * self.velocity(I, J, 2. * t - 1.)
        grid = self.make_grid(flow, xyz)

        flow_loss = 0.5 * (torch.mean((grid - grid_I_J) ** 2) + torch.mean((grid - grid_J_I) ** 2))

        return image_loss, flow_loss
  
    def make_grid(self, flow: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(self, image: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        grid = grid + flow.permute(0, 2, 3, 4, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image, grid, padding_mode='reflection', align_corners=True)

        return warped

    def compose(self, flow1: torch.Tensor, flow2: torch.Tensor, grid: torch.Tensor):
        grid = grid + flow2.permute(0, 2, 3, 4, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1, grid, padding_mode='reflection', align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(self, grid: torch.Tensor) -> torch.Tensor:
        _, d, h, w, _ = grid.size()
        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((d - 1) / 2)) / (d - 1) * 2
        
        return grid
