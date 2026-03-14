"""Constructions of 2D and 3D SGDIR and SGDIRDiT given different backbones
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from metrics import NCCLoss
from metrics import NGFLoss


class SGDIR(nn.Module):
    """Implementation of main 3D SGDIR with UNet-based backbone
    """
    def __init__(
        self,
        backbone: nn.Module,
        loss_type: str='ncc'
    ) -> None:
        """
        :param backbone: 
            An instance of UNet3D

        :param loss_type:
            The loss function to be used
            Available loss functions: 'ncc', 'nfg', 'mse'
        """
        super().__init__()

        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = NCCLoss(win=11)
        elif loss_type == 'ngf':
            self.loss_fn = NGFLoss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError('The given loss function is not defined!')

        self.net = backbone
  
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed: 
            Fixed image with size [B, 1, D, H, W]

        :param moving: 
            Moving image with size [B, 1, D, H, W]

        :param id_grid: 
            Identity grid with size [B, D, H, W, 3]

        :param t:
            Sampled time with size [B]

        :returns:
            The deformation grid at time t with size [B, D, H, W, 3]
        """
        flow = t * self.flow_core(fixed, moving, t)

        phi_t = self.make_grid(flow, id_grid)

        return phi_t
  
    def flow_core(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed:
            Fixed image with size [B, 1, D, H, W]

        :param moving:
            Moving image with size [B, 1, D, H, W]

        :param t:
            Sampled time with size [1]

        :returns:
            The vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        flow = self.net(u_in, t)

        return flow

    def loss_flow(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param fixed:
            Fixed image with size [B, 1, D, H, W]

        :param moving:
            Moving image with size [B, 1, D, H, W]

        :param id_grid:
            Identity grid with size [B, D, H, W, 3]

        :param res:
            The resolution at which the similarity loss is computed

        :returns:
            (Similarity loss, Semigroup loss)
        """
        t = torch.rand(1, device=fixed.device)

        flow_moving = t * self.flow_core(fixed, moving, t)
        moving_warped = self.warp(moving, flow_moving, id_grid)

        flow_fixed = (t - 1.) * self.flow_core(fixed, moving, t - 1.)
        fixed_warped = self.warp(fixed, flow_fixed, id_grid)

        if res != 1:
            fixed_warped = F.interpolate(fixed_warped,
                                         scale_factor=res,
                                         mode='trilinear')
            moving_warped = F.interpolate(moving_warped,
                                          scale_factor=res,
                                          mode='trilinear')

        image_loss = res * self.loss_fn(moving_warped, fixed_warped)

        flow_fixed_moving = self.compose(flow_fixed, flow_moving, id_grid)
        grid_fixed_moving = self.make_grid(flow_fixed_moving, id_grid)

        flow_moving_fixed = self.compose(flow_moving, flow_fixed, id_grid)
        grid_moving_fixed = self.make_grid(flow_moving_fixed, id_grid)

        flow = (2. * t - 1.) * self.flow_core(fixed, moving, 2. * t - 1.)
        grid = self.make_grid(flow, id_grid)

        flow_loss = 0.5 * (torch.mean((grid - grid_fixed_moving) ** 2) +
                           torch.mean((grid - grid_moving_fixed) ** 2))

        return image_loss, flow_loss
  
    def make_grid(
        self,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Applies a displacement field to an unnormalized grid

        :param flow:
            A displacement field with shape [B, 3, D, H, W]
        
        :param grid:
            An unnormalized grid with shape [B, D, H, W, 3]

        :returns:
            The [-1, 1] normalized deformation grid obtained from the flow
            Shape will be [B, D, H, W, 3]
        """
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Warping operation based on a displacement field
        and an unnormalized grid

        :param image:
            The input image with shape [B, 1, D, H, W]
        
        :param flow:
            The displacement field with shape [B, 3, D, H, W]

        :param grid:
            The unnormalized grid with shape [B, D, H, W, 3]

        :returns:
            The warped image with shape [B, 1, D, H, W]
        """
        grid = grid + flow.permute(0, 2, 3, 4, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image,
                               grid,
                               padding_mode='reflection',
                               align_corners=True)

        return warped

    def compose(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Implements the flow compostion: flow_1 o flow_2

        :param flow_1:
            The first flow with shape [B, 3, D, H, W]

        :param flow_2:
            The second flow with shape [B, 3, D, H, W]

        :param grid:
            The grid on which flow_2 acts with shape [B, D, H, W, 3]

        :returns:
            The composed flow_1 o flow_2 with shape [B, 3, D, H, W]
        """
        grid = grid + flow2.permute(0, 2, 3, 4, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1,
                                      grid,
                                      padding_mode='reflection',
                                      align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes an unnormalized grid into [-1, 1] range

        :param grid:
            The unnormalized grid with shape [B, D, H, W, 3]

        :returns:
            The [-1, 1] normalized grid with shape [B, D, H, W, 3]
        """
        _, d, h, w, _ = grid.size()

        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((d - 1) / 2)) / (d - 1) * 2
        
        return grid


class SGDIR2D(nn.Module):
    """Implementation of main 2D SGDIR with UNet-based backbone
    """
    def __init__(
        self,
        backbone: nn.Module,
        loss_type: str='ncc'
    ) -> None:
        """
        :param backbone: 
            An instance of UNet2D

        :param loss_type:
            The loss function to be used
            Available loss functions: 'ncc', 'mse'
        """
        super().__init__()

        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = NCCLoss(win=11)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError('The given loss function is not defined!')

        self.net = backbone
  
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed: 
            Fixed image with size [B, 1, H, W]

        :param moving: 
            Moving image with size [B, 1, H, W]

        :param id_grid: 
            Identity grid with size [B, H, W, 2]

        :param t:
            Sampled time with size [B]

        :returns:
            The deformation grid at time t with size [B, H, W, 2]
        """
        flow = t * self.flow_core(fixed, moving, t)

        phi_t = self.make_grid(flow, id_grid)

        return phi_t
  
    def flow_core(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed:
            Fixed image with size [B, 1, H, W]

        :param moving:
            Moving image with size [B, 1, H, W]

        :param t:
            Sampled time with size [1]

        :returns:
            The vector field at time t with size [B, 2, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        flow = self.net(u_in, t)

        return flow

    def loss_flow(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param fixed:
            Fixed image with size [B, 1, D, H, W]

        :param moving:
            Moving image with size [B, 1, D, H, W]

        :param id_grid:
            Identity grid with size [B, D, H, W, 3]

        :param res:
            The resolution at which the similarity loss is computed

        :returns:
            (Similarity loss, Semigroup loss)
        """
        t = torch.rand(1, device=fixed.device)

        flow_moving = t * self.flow_core(fixed, moving, t)
        moving_warped = self.warp(moving, flow_moving, id_grid)

        flow_fixed = (t - 1.) * self.flow_core(fixed, moving, t - 1.)
        fixed_warped = self.warp(fixed, flow_fixed, id_grid)

        if res != 1:
            fixed_warped = F.interpolate(fixed_warped,
                                         scale_factor=res,
                                         mode='bilinear',
                                         antialias=True)
            moving_warped = F.interpolate(moving_warped,
                                          scale_factor=res,
                                          mode='bilinear',
                                          antialias=True)

        image_loss = res * self.loss_fn(moving_warped, fixed_warped)

        flow_fixed_moving = self.compose(flow_fixed, flow_moving, id_grid)
        grid_fixed_moving = self.make_grid(flow_fixed_moving, id_grid)

        flow_moving_fixed = self.compose(flow_moving, flow_fixed, id_grid)
        grid_moving_fixed = self.make_grid(flow_moving_fixed, id_grid)

        flow = (2. * t - 1.) * self.flow_core(fixed, moving, 2. * t - 1.)
        grid = self.make_grid(flow, id_grid)

        flow_loss = 0.5 * (torch.mean((grid - grid_fixed_moving) ** 2) +
                           torch.mean((grid - grid_moving_fixed) ** 2))

        return image_loss, flow_loss
  
    def make_grid(
        self,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Applies a displacement field to an unnormalized grid

        :param flow:
            A displacement field with shape [B, 2, H, W]
        
        :param grid:
            An unnormalized grid with shape [B, H, W, 2]

        :returns:
            The [-1, 1] normalized deformation grid obtained from the flow
            Shape will be [B, H, W, 2]
        """
        phi = grid + flow.permute(0, 2, 3, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Warping operation based on a displacement field
        and an unnormalized grid

        :param image:
            The input image with shape [B, 1, H, W]
        
        :param flow:
            The displacement field with shape [B, 2, H, W]

        :param grid:
            The unnormalized grid with shape [B, H, W, 2]

        :returns:
            The warped image with shape [B, 1, H, W]
        """
        grid = grid + flow.permute(0, 2, 3, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image,
                               grid,
                               padding_mode='reflection',
                               align_corners=True)

        return warped

    def compose(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Implements the flow compostion: flow_1 o flow_2

        :param flow_1:
            The first flow with shape [B, 2, H, W]

        :param flow_2:
            The second flow with shape [B, 2, H, W]

        :param grid:
            The grid on which flow_2 acts with shape [B, H, W, 2]

        :returns:
            The composed flow_1 o flow_2 with shape [B, 2, H, W]
        """
        grid = grid + flow2.permute(0, 2, 3, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1,
                                      grid,
                                      padding_mode='reflection',
                                      align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes an unnormalized grid into [-1, 1] range

        :param grid:
            The unnormalized grid with shape [B, H, W, 2]

        :returns:
            The [-1, 1] normalized grid with shape [B, H, W, 2]
        """
        _, h, w, _ = grid.size()

        grid[:, :, :, 0] = (grid[:, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, 1] = (grid[:, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        
        return grid


class SGDIRDiT(nn.Module):
    """Implementation of main 3D SGDIRDiT with LatentDiT3D backbone
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        loss_type: str='ncc'
    ) -> None:
        """
        :param backbone: 
            An instance of LatentDiT3D

        :param loss_type:
            The loss function to be used
            Available loss functions: 'ncc', 'ngf', 'mse'
        """
        super().__init__()

        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = NCCLoss(win=11)
        elif loss_type == 'ngf':
            self.loss_fn = NGFLoss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError('The given loss function is not defined!')

        self.net = backbone
  
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed: 
            Fixed image with size [B, 1, D, H, W]

        :param moving: 
            Moving image with size [B, 1, D, H, W]

        :param id_grid: 
            Identity grid with size [B, D, H, W, 3]

        :param t:
            Sampled time with size [B]

        :returns:
            The deformation grid at time t with size [B, D, H, W, 3]
        """
        flow = t * self.flow_core(fixed, moving, t)

        phi_t = self.make_grid(flow, id_grid)

        return phi_t
  
    def flow_core(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed:
            Fixed image with size [B, 1, D, H, W]

        :param moving:
            Moving image with size [B, 1, D, H, W]

        :param t:
            Sampled time with size [1]

        :returns:
            The vector field at time t with size [B, 3, D, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        flow = self.net(u_in, t)

        return flow

    def loss_flow(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param fixed:
            Fixed image with size [B, 1, D, H, W]

        :param moving:
            Moving image with size [B, 1, D, H, W]

        :param id_grid:
            Identity grid with size [B, D, H, W, 3]

        :param res:
            The resolution at which the similarity loss is computed

        :returns:
            (Similarity loss, Semigroup loss)
        """
        t = torch.rand(1, device=fixed.device)

        v1, v2, v3 = self.compute_flows(fixed, moving, t)

        flow_moving = t * v1
        moving_warped = self.warp(moving, flow_moving, id_grid)

        flow_fixed = (t - 1.) * v2
        fixed_warped = self.warp(fixed, flow_fixed, id_grid)

        if res != 1:
            fixed_warped = F.interpolate(fixed_warped,
                                         scale_factor=res,
                                         mode='trilinear')
            moving_warped = F.interpolate(moving_warped,
                                          scale_factor=res,
                                          mode='trilinear')

        image_loss = res * self.loss_fn(moving_warped, fixed_warped)

        flow_fixed_moving = self.compose(flow_fixed, flow_moving, id_grid)
        grid_fixed_moving = self.make_grid(flow_fixed_moving, id_grid)

        flow_moving_fixed = self.compose(flow_moving, flow_fixed, id_grid)
        grid_moving_fixed = self.make_grid(flow_moving_fixed, id_grid)

        flow = (2. * t - 1.) * v3
        grid = self.make_grid(flow, id_grid)

        flow_loss = 0.5 * (torch.mean((grid - grid_fixed_moving) ** 2) +
                           torch.mean((grid - grid_moving_fixed) ** 2))

        return image_loss, flow_loss
  
    def make_grid(
        self,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Applies a displacement field to an unnormalized grid

        :param flow:
            A displacement field with shape [B, 3, D, H, W]
        
        :param grid:
            An unnormalized grid with shape [B, D, H, W, 3]

        :returns:
            The [-1, 1] normalized deformation grid obtained from the flow
            Shape will be [B, D, H, W, 3]
        """
        phi = grid + flow.permute(0, 2, 3, 4, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Warping operation based on a displacement field
        and an unnormalized grid

        :param image:
            The input image with shape [B, 1, D, H, W]
        
        :param flow:
            The displacement field with shape [B, 3, D, H, W]

        :param grid:
            The unnormalized grid with shape [B, D, H, W, 3]

        :returns:
            The warped image with shape [B, 1, D, H, W]
        """
        grid = grid + flow.permute(0, 2, 3, 4, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image,
                               grid,
                               padding_mode='reflection',
                               align_corners=True)

        return warped

    def compose(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Implements the flow compostion: flow_1 o flow_2

        :param flow_1:
            The first flow with shape [B, 3, D, H, W]

        :param flow_2:
            The second flow with shape [B, 3, D, H, W]

        :param grid:
            The grid on which flow_2 acts with shape [B, D, H, W, 3]

        :returns:
            The composed flow_1 o flow_2 with shape [B, 3, D, H, W]
        """
        grid = grid + flow2.permute(0, 2, 3, 4, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1,
                                      grid,
                                      padding_mode='reflection',
                                      align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes an unnormalized grid into [-1, 1] range

        :param grid:
            The unnormalized grid with shape [B, D, H, W, 3]

        :returns:
            The [-1, 1] normalized grid with shape [B, D, H, W, 3]
        """
        _, d, h, w, _ = grid.size()

        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((d - 1) / 2)) / (d - 1) * 2
        
        return grid

    def compute_flows(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Computes the flows at times t, t-1 and 2t-1

        :param fixed: 
            Fixed image with size [B, 1, D, H, W]

        :param moving: 
            Moving image with size [B, 1, D, H, W]

        :param t:
            Sampled time with size [B]

        :returns:
            Flows f(t), f(t-1), f(2t-1) each with shape [B, 3, D, H, W]
        """
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
    """Implementation of main 2D SGDIRDiT with LatentDiT2D backbone
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        loss_type: str='ncc'
    ) -> None:
        """
        :param backbone: 
            An instance of LatentDiT2D

        :param loss_type:
            The loss function to be used
            Available loss functions: 'ncc', 'mse'
        """
        super().__init__()

        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = NCCLoss(win=11)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError('The given loss function is not defined!')

        self.net = backbone
  
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed: 
            Fixed image with size [B, 1, H, W]

        :param moving: 
            Moving image with size [B, 1, H, W]

        :param id_grid: 
            Identity grid with size [B, H, W, 2]

        :param t:
            Sampled time with size [B]

        :returns:
            The deformation grid at time t with size [B, H, W, 2]
        """
        flow = t * self.flow_core(fixed, moving, t)

        phi_t = self.make_grid(flow, id_grid)

        return phi_t
  
    def flow_core(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed:
            Fixed image with size [B, 1, H, W]

        :param moving:
            Moving image with size [B, 1, H, W]

        :param t:
            Sampled time with size [1]

        :returns:
            The vector field at time t with size [B, 2, H, W]
        """
        u_in = torch.cat([fixed, moving], dim=1)

        flow = self.net(u_in, t)

        return flow

    def loss_flow(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        id_grid: torch.Tensor,
        res: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param fixed:
            Fixed image with size [B, 1, H, W]

        :param moving:
            Moving image with size [B, 1, H, W]

        :param id_grid:
            Identity grid with size [B, H, W, 2]

        :param res:
            The resolution at which the similarity loss is computed

        :returns:
            (Similarity loss, Semigroup loss)
        """
        t = torch.rand(1, device=fixed.device)

        v1, v2, v3 = self.compute_flows(fixed, moving, t)

        flow_moving = t * v1
        moving_warped = self.warp(moving, flow_moving, id_grid)

        flow_fixed = (t - 1.) * v2
        fixed_warped = self.warp(fixed, flow_fixed, id_grid)

        if res != 1:
            fixed_warped = F.interpolate(fixed_warped,
                                         scale_factor=res,
                                         mode='bilinear')
            moving_warped = F.interpolate(moving_warped,
                                          scale_factor=res,
                                          mode='bilinear')

        image_loss = res * self.loss_fn(moving_warped, fixed_warped)

        flow_fixed_moving = self.compose(flow_fixed, flow_moving, id_grid)
        grid_fixed_moving = self.make_grid(flow_fixed_moving, id_grid)

        flow_moving_fixed = self.compose(flow_moving, flow_fixed, id_grid)
        grid_moving_fixed = self.make_grid(flow_moving_fixed, id_grid)

        flow = (2. * t - 1.) * v3
        grid = self.make_grid(flow, id_grid)

        flow_loss = 0.5 * (torch.mean((grid - grid_fixed_moving) ** 2) +
                           torch.mean((grid - grid_moving_fixed) ** 2))

        return image_loss, flow_loss
  
    def make_grid(
        self,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Applies a displacement field to an unnormalized grid

        :param flow:
            A displacement field with shape [B, 2, H, W]
        
        :param grid:
            An unnormalized grid with shape [B, H, W, 2]

        :returns:
            The [-1, 1] normalized deformation grid obtained from the flow
            Shape will be [B, H, W, 2]
        """
        phi = grid + flow.permute(0, 2, 3, 1)

        phi = self.grid_normalizer(phi)

        return phi
  
    def warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Warping operation based on a displacement field
        and an unnormalized grid

        :param image:
            The input image with shape [B, 1, H, W]
        
        :param flow:
            The displacement field with shape [B, 2, H, W]

        :param grid:
            The unnormalized grid with shape [B, H, W, 2]

        :returns:
            The warped image with shape [B, 1, H, W]
        """
        grid = grid + flow.permute(0, 2, 3, 1)
        grid = self.grid_normalizer(grid)

        warped = F.grid_sample(image,
                               grid,
                               padding_mode='reflection',
                               align_corners=True)

        return warped

    def compose(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Implements the flow compostion: flow_1 o flow_2

        :param flow_1:
            The first flow with shape [B, 2, H, W]

        :param flow_2:
            The second flow with shape [B, 2, H, W]

        :param grid:
            The grid on which flow_2 acts with shape [B, H, W, 2]

        :returns:
            The composed flow_1 o flow_2 with shape [B, 2, H, W]
        """
        grid = grid + flow2.permute(0, 2, 3, 1)

        grid = self.grid_normalizer(grid)

        composed_flow = F.grid_sample(flow1,
                                      grid,
                                      padding_mode='reflection',
                                      align_corners=True) + flow2

        return composed_flow
  
    def grid_normalizer(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """Normalizes an unnormalized grid into [-1, 1] range

        :param grid:
            The unnormalized grid with shape [B, H, W, 2]

        :returns:
            The [-1, 1] normalized grid with shape [B, H, W, 2]
        """
        _, h, w, _ = grid.size()

        grid[:, :, :, 0] = (grid[:, :, :, 0] - ((w - 1) / 2)) / (w - 1) * 2
        grid[:, :, :, 1] = (grid[:, :, :, 1] - ((h - 1) / 2)) / (h - 1) * 2
        
        return grid

    def compute_flows(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Computes the flows at times t, t-1 and 2t-1

        :param fixed: 
            Fixed image with size [B, 1, H, W]

        :param moving: 
            Moving image with size [B, 1, H, W]

        :param t:
            Sampled time with size [B]

        :returns:
            Flows f(t), f(t-1), f(2t-1) each with shape [B, 2, H, W]
        """
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
