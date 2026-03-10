import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from metrics import NCCLoss
from metrics import NGFLoss


class SGDIR(nn.Module):
    """
    Implementation of both Time-Independet and Time-Dependent Phi network based on interpolative cmposition
    """
    def __init__(self, 
                 backbone: nn.Module, 
                 loss_type: str='ncc') -> None:
        super().__init__()

        self.loss_type = loss_type
        self.ncc_loss = NCCLoss(win=11)
        if loss_type == 'ngf':
            self.ngf_loss = NGFLoss()

        self.net = backbone
  
    def forward(self, fixed: torch.Tensor,
                moving: torch.Tensor,
                xyz: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            t (torch.Tensor): sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, D, H, W, 3]
        """
        flow = t * self.velocity(fixed, moving, t)

        phi_t = self.make_grid(flow, xyz)

        return phi_t
  
    def velocity(self, fixed: torch.Tensor,
                 moving: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            t (torch.Tensor): sampled time with size [1]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        velocity = self.net(u_in, t)

        return velocity

    def loss_flow(self, fixed: torch.Tensor,
                  moving: torch.Tensor,
                  xyz: torch.Tensor,
                  res: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        t = torch.rand(1, device=fixed.device)

        flow_J = t * self.velocity(fixed, moving, t)
        Jw = self.warp(moving, flow_J, xyz)

        flow_I = (t - 1.) * self.velocity(fixed, moving, t - 1.)
        Iw = self.warp(fixed, flow_I, xyz)

        if res != 1:
            Iw = F.interpolate(Iw, scale_factor=res, mode='trilinear')
            Jw = F.interpolate(Jw, scale_factor=res, mode='trilinear')

        if self.loss_type == 'mse':
            image_loss = res * F.mse_loss(Jw, Iw)
        elif self.loss_type == 'ngf':
            image_loss = res * self.ngf_loss(Jw, Iw)
        else:
            image_loss = res * self.ncc_loss(Jw, Iw)

        flow_I_J = self.compose(flow_I, flow_J, xyz)
        grid_I_J = self.make_grid(flow_I_J, xyz)
        flow_J_I = self.compose(flow_J, flow_I, xyz)
        grid_J_I = self.make_grid(flow_J_I, xyz)
        flow = (2. * t - 1.) * self.velocity(fixed, moving, 2. * t - 1.)
        grid = self.make_grid(flow, xyz)

        flow_loss = 0.5 * (torch.mean((grid - grid_I_J) ** 2) + torch.mean((grid - grid_J_I) ** 2))

        return image_loss, flow_loss
  
    def make_grid(self, flow: torch.Tensor,
                  grid: torch.Tensor) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(self, image: torch.Tensor,
             flow: torch.Tensor,
             grid: torch.Tensor) -> torch.Tensor:
        grid = grid + flow.permute(0, 2, 3, 4, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image, grid, padding_mode='reflection', align_corners=True)

        return warped

    def compose(self, flow1: torch.Tensor,
                flow2: torch.Tensor,
                grid: torch.Tensor):
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


class SGDIR2D(nn.Module):
    """
    Implementation of both Time-Independet and Time-Dependent Phi network based on interpolative cmposition
    """
    def __init__(self, 
                 backbone: nn.Module, 
                 loss_type: str='ncc') -> None:
        super().__init__()

        self.loss_type = loss_type
        self.ncc_loss = NCCLoss(win=11)

        self.net = backbone
  
    def forward(self, fixed: torch.Tensor,
                moving: torch.Tensor,
                xy: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            xyz (torch.Tensor): identity grid with size [B, H, W, 3]
            t (torch.Tensor): sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, H, W, 2]
        """
        flow = t * self.velocity(fixed, moving, t)

        phi_t = self.make_grid(flow, xy)

        return phi_t
  
    def velocity(self, fixed: torch.Tensor,
                 moving: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            t (torch.Tensor): sampled time with size [1]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 2, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        velocity = self.net(u_in, t)

        return velocity

    def loss_flow(self, fixed: torch.Tensor,
                  moving: torch.Tensor,
                  xyz: torch.Tensor,
                  res: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            xyz (torch.Tensor): identity grid with size [B, H, W, 2]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        t = torch.rand(1, device=fixed.device)

        flow_J = t * self.velocity(fixed, moving, t)
        Jw = self.warp(moving, flow_J, xyz)

        flow_I = (t - 1.) * self.velocity(fixed, moving, t - 1.)
        Iw = self.warp(fixed, flow_I, xyz)

        if res != 1:
            Iw = F.interpolate(Iw, scale_factor=res, mode='bilinear', antialias=True)
            Jw = F.interpolate(Jw, scale_factor=res, mode='bilinear', antialias=True)

        if self.loss_type == 'mse':
            image_loss = res * F.mse_loss(Jw, Iw)
        else:
            image_loss = res * self.ncc_loss(Jw, Iw)

        flow_I_J = self.compose(flow_I, flow_J, xyz)
        grid_I_J = self.make_grid(flow_I_J, xyz)
        flow_J_I = self.compose(flow_J, flow_I, xyz)
        grid_J_I = self.make_grid(flow_J_I, xyz)
        flow = (2. * t - 1.) * self.velocity(fixed, moving, 2. * t - 1.)
        grid = self.make_grid(flow, xyz)

        flow_loss = 0.5 * (torch.mean((grid - grid_I_J) ** 2) + torch.mean((grid - grid_J_I) ** 2))

        return image_loss, flow_loss
  
    def make_grid(self, flow: torch.Tensor,
                  grid: torch.Tensor) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(self, image: torch.Tensor,
             flow: torch.Tensor,
             grid: torch.Tensor) -> torch.Tensor:
        grid = grid + flow.permute(0, 2, 3, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image, grid, padding_mode='reflection', align_corners=True)

        return warped

    def compose(self, flow1: torch.Tensor,
                flow2: torch.Tensor,
                grid: torch.Tensor):
        grid = grid + flow2.permute(0, 2, 3, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1, grid, padding_mode='reflection', align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(self, grid: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = grid.size()

        grid[:, :, :, 0] = (grid[:, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, 1] = (grid[:, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        
        return grid


class SGDIRDiT(nn.Module):
    """
    Implementation of both Time-Independet and Time-Dependent Phi network based on interpolative cmposition
    """
    def __init__(self, 
                 backbone: nn.Module, 
                 loss_type: str='ncc') -> None:
        super().__init__()

        self.loss_type = loss_type
        self.ncc_loss = NCCLoss(win=11)
        if loss_type == 'ngf':
            self.ngf_loss = NGFLoss()

        self.net = backbone
  
    def forward(self, fixed: torch.Tensor,
                moving: torch.Tensor,
                xyz: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            t (torch.Tensor): sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, D, H, W, 3]
        """
        flow = t * self.velocity(fixed, moving, t)

        phi_t = self.make_grid(flow, xyz)

        return phi_t
  
    def velocity(self, fixed: torch.Tensor,
                 moving: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            t (torch.Tensor): sampled time with size [1]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        velocity = self.net(u_in, t)

        return velocity

    def loss_flow(self, fixed: torch.Tensor,
                  moving: torch.Tensor,
                  xyz: torch.Tensor,
                  res: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, D, H, W]
            J (torch.Tensor): moving image with size [B, 1, D, H, W]
            xyz (torch.Tensor): identity grid with size [B, D, H, W, 3]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        t = torch.rand(1, device=fixed.device)

        v1, v2, v3 = self.compute_velocities(fixed, moving, t)

        flow_J = t * v1
        Jw = self.warp(moving, flow_J, xyz)

        flow_I = (t - 1.) * v2
        Iw = self.warp(fixed, flow_I, xyz)

        if res != 1:
            Iw = F.interpolate(Iw, scale_factor=res, mode='trilinear')
            Jw = F.interpolate(Jw, scale_factor=res, mode='trilinear')

        if self.loss_type == 'mse':
            image_loss = res * F.mse_loss(Jw, Iw)
        elif self.loss_type == 'ngf':
            image_loss = res * (self.ngf_loss(Jw, Iw) + F.mse_loss(Jw, Iw))
        else:
            image_loss = res * self.ncc_loss(Jw, Iw)

        flow_I_J = self.compose(flow_I, flow_J, xyz)
        grid_I_J = self.make_grid(flow_I_J, xyz)
        flow_J_I = self.compose(flow_J, flow_I, xyz)
        grid_J_I = self.make_grid(flow_J_I, xyz)
        flow = (2. * t - 1.) * v3
        grid = self.make_grid(flow, xyz)

        flow_loss = 0.5 * (torch.mean((grid - grid_I_J) ** 2) + torch.mean((grid - grid_J_I) ** 2))

        return image_loss, flow_loss
  
    def make_grid(self, flow: torch.Tensor,
                  grid: torch.Tensor) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(self, image: torch.Tensor,
             flow: torch.Tensor,
             grid: torch.Tensor) -> torch.Tensor:
        grid = grid + flow.permute(0, 2, 3, 4, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image, grid, padding_mode='reflection', align_corners=True)

        return warped

    def compose(self, flow1: torch.Tensor,
                flow2: torch.Tensor,
                grid: torch.Tensor):
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

    def compute_velocities(self, fixed: torch.Tensor,
                           moving: torch.Tensor,
                           t: torch.Tensor) -> Tuple[torch.Tensor]:
        encodings = self.net.encode(torch.cat([fixed, moving], dim=1))

        latent_t = self.net.bottleneck(encodings[-1], t)
        latent_t_1 = self.net.bottleneck(encodings[-1], t - 1)
        latent_2t_1 = self.net.bottleneck(encodings[-1], 2 * t - 1)

        decoder_t = None
        decoder_t_1 = None
        decoder_2t_1 = None
        if self.net.decoder_time_embed_dim is not None:
            decoder_t = t
            decoder_t_1 = t - 1
            decoder_2t_1 = 2 * t - 1

        decoded1 = self.net.decode(latent_t, encodings.copy(), decoder_t)
        decoded2 = self.net.decode(latent_t_1, encodings.copy(), decoder_t_1)
        decoded3 = self.net.decode(latent_2t_1, encodings, decoder_2t_1)

        return decoded1, decoded2, decoded3


class SGDIRDiT2D(nn.Module):
    """
    Implementation of both Time-Independet and Time-Dependent Phi network based on interpolative cmposition
    """
    def __init__(self, 
                 backbone: nn.Module, 
                 loss_type: str='ncc') -> None:
        super().__init__()

        self.loss_type = loss_type
        self.ncc_loss = NCCLoss(win=11)
        if loss_type == 'ngf':
            self.ngf_loss = NGFLoss()

        self.net = backbone
  
    def forward(self, fixed: torch.Tensor,
                moving: torch.Tensor,
                xy: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            xy (torch.Tensor): identity grid with size [B, H, W, 2]
            t (torch.Tensor): sampled time with size [B]
        Returns:
            torch.Tensor: the deformation grid at time t with size [B, H, W, 2]
        """
        flow = t * self.velocity(fixed, moving, t)

        phi_t = self.make_grid(flow, xy)

        return phi_t
  
    def velocity(self, fixed: torch.Tensor,
                 moving: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            t (torch.Tensor): sampled time with size [1]
        Returns:
            torch.Tensor: the vector field at time t with size [B, 2, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        velocity = self.net(u_in, t)

        return velocity

    def loss_flow(self, fixed: torch.Tensor,
                  moving: torch.Tensor,
                  xy: torch.Tensor,
                  res: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            I (torch.Tensor): fixed image with size [B, 1, H, W]
            J (torch.Tensor): moving image with size [B, 1, H, W]
            xy (torch.Tensor): identity grid with size [B, H, W, 2]
            res (float): the resolution at which the ncc loss is computed
        Returns:
            torch.Tensor, torch.Tensor: ncc loss, semigroup loss
        """
        t = torch.rand(1, device=fixed.device)

        v1, v2, v3 = self.compute_velocities(fixed, moving, t)

        flow_J = t * v1
        Jw = self.warp(moving, flow_J, xy)

        flow_I = (t - 1.) * v2
        Iw = self.warp(fixed, flow_I, xy)

        if res != 1:
            Iw = F.interpolate(Iw, scale_factor=res, mode='bilinear')
            Jw = F.interpolate(Jw, scale_factor=res, mode='bilinear')

        if self.loss_type == 'mse':
            image_loss = res * F.mse_loss(Jw, Iw)
        elif self.loss_type == 'ngf':
            image_loss = res * (self.ngf_loss(Jw, Iw) + F.mse_loss(Jw, Iw))
        else:
            image_loss = res * self.ncc_loss(Jw, Iw)

        flow_I_J = self.compose(flow_I, flow_J, xy)
        grid_I_J = self.make_grid(flow_I_J, xy)
        flow_J_I = self.compose(flow_J, flow_I, xy)
        grid_J_I = self.make_grid(flow_J_I, xy)
        flow = (2. * t - 1.) * v3
        grid = self.make_grid(flow, xy)

        flow_loss = 0.5 * (torch.mean((grid - grid_I_J) ** 2) + torch.mean((grid - grid_J_I) ** 2))

        return image_loss, flow_loss
  
    def make_grid(self, flow: torch.Tensor,
                  grid: torch.Tensor) -> torch.Tensor:
        phi = grid + flow.permute(0, 2, 3, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(self, image: torch.Tensor,
             flow: torch.Tensor,
             grid: torch.Tensor) -> torch.Tensor:
        grid = grid + flow.permute(0, 2, 3, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image, grid, padding_mode='reflection', align_corners=True)

        return warped

    def compose(self, flow1: torch.Tensor,
                flow2: torch.Tensor,
                grid: torch.Tensor):
        grid = grid + flow2.permute(0, 2, 3, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1, grid, padding_mode='reflection', align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(self, grid: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = grid.size()

        grid[:, :, :, 0] = (grid[:, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, 1] = (grid[:, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        
        return grid

    def compute_velocities(self, fixed: torch.Tensor,
                           moving: torch.Tensor,
                           t: torch.Tensor) -> Tuple[torch.Tensor]:
        encodings = self.net.encode(torch.cat([fixed, moving], dim=1))

        latent_t = self.net.bottleneck(encodings[-1], t)
        latent_t_1 = self.net.bottleneck(encodings[-1], t - 1)
        latent_2t_1 = self.net.bottleneck(encodings[-1], 2 * t - 1)

        decoder_t = None
        decoder_t_1 = None
        decoder_2t_1 = None
        if self.net.decoder_time_embed_dim is not None:
            decoder_t = t
            decoder_t_1 = t - 1
            decoder_2t_1 = 2 * t - 1

        decoded1 = self.net.decode(latent_t, encodings.copy(), decoder_t)
        decoded2 = self.net.decode(latent_t_1, encodings.copy(), decoder_t_1)
        decoded3 = self.net.decode(latent_2t_1, encodings, decoder_2t_1)

        return decoded1, decoded2, decoded3
