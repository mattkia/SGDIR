"""Implementations of loss functions and evaluation metrics
"""

import torch
import scipy.ndimage

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import cKDTree
from torchvision.transforms.functional import rgb_to_grayscale

from utils import dsc_score
from utils import jacobian_determinant


class Dice(nn.Module):
    """Implements Dice score for evaluation
    """
    def __init__(
        self,
        structured: bool=True
    ):
        """
        :param structured:
            Whether to use structured dice or volumetric dice.
        """
        super().__init__()

        self.structured = structured

    def forward(
        self,
        seg_map1: torch.Tensor, 
        seg_map2: torch.Tensor
    ) -> torch.Tensor:
        """
        :param seg_map_1:
            First segmentation mask;
            Accepted shapes: [B, C, D, H, W] or [B C, H, W]

        :param seg_map_2:
            Second segmentation mask;
            Accepted shapes: [B, C, D, H, W] or [B C, H, W]

        :returns:
            The dice score between the two segmentation maps
        """
        
        dice = dsc_score(seg_map1=seg_map1,
                         seg_map2=seg_map2,
                         structured=self.structured)

        return dice


class DicePerStructure(nn.Module):
    """Computes and tracks the Dice score for each anatomical
    region
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        seg_map_1: torch.Tensor, 
        seg_map_2: torch.Tensor
    ) -> torch.Tensor:
        """
        :param seg_map_1:
            First segmentation mask;
            Accepted shapes: [B, C, D, H, W] or [B C, H, W]

        :param seg_map_2:
            Second segmentation mask;
            Accepted shapes: [B, C, D, H, W] or [B C, H, W]

        :returns:
            A tensor keeping the average Dice score for each
            anatomical region
        """
        
        device = seg_map_1.device

        labels = torch.unique(torch.cat((seg_map_1, seg_map_2)))
        labels = labels[torch.where(labels != 0)]

        structured_dices = torch.zeros(len(labels), device=device)

        for idx, lab in enumerate(labels):
            top = 2 * torch.sum(torch.logical_and(seg_map_1 == lab, seg_map_2 == lab))
            bottom = torch.sum(seg_map_1 == lab) + torch.sum(seg_map_2 == lab)
            bottom = torch.maximum(bottom, torch.tensor(torch.finfo(float).eps, device=device))
            structured_dices[idx] += top / bottom

        return structured_dices


class SurfaceDice(nn.Module):
    """Computes the Surface Dice score. Useful for segmentation
    maps of cortical regions.
    """
    def __init__(
        self,
        tolerance_mm: float=1.0
    ):
        """
        :param tolerance_mm:
            Distance threshold (in mm) for surfaces to be considered overlapping.
        """
        super().__init__()
        self.tolerance = tolerance_mm

    def forward(
        self,
        seg_map1: torch.Tensor,
        seg_map2: torch.Tensor,
        spacing: tuple=(1.0, 1.0, 1.0)
    ) -> float:
        """
        :param seg_map1:
            First segmentation map.
            Accepted shape: [B, 1, D, H, W]

        :param seg_map2:
            Second segmentation map.
            Accepted shape: [B, 1, D, H, W]

        :param spacing:
            Voxel spacing in (Z, Y, X) order

        :returns:
            Average Surface Dice Coefficient across labels and batch.
        """
        seg1_np = seg_map1[0, 0].cpu().numpy()
        seg2_np = seg_map2[0, 0].cpu().numpy()

        labels = np.unique(np.concatenate([np.unique(seg1_np), np.unique(seg2_np)]))
        labels = labels[labels != 0]  # exclude background

        sdc_list = []

        for label in labels:
            mask1 = (seg1_np == label)
            mask2 = (seg2_np == label)

            # Extract surfaces (XOR between mask and eroded mask)
            surface1 = np.array(np.where(mask1 ^ scipy.ndimage.binary_erosion(mask1))).T
            surface2 = np.array(np.where(mask2 ^ scipy.ndimage.binary_erosion(mask2))).T

            if len(surface1) == 0 or len(surface2) == 0:
                continue

            # Apply voxel spacing
            surface1 = surface1 * np.array(spacing)
            surface2 = surface2 * np.array(spacing)

            # KD-trees for efficient nearest-neighbor distances
            tree1 = cKDTree(surface1)
            tree2 = cKDTree(surface2)

            # Distance from surface1 to nearest point in surface2
            dist_1_to_2 = tree2.query(surface1, k=1)[0]
            # Distance from surface2 to nearest point in surface1
            dist_2_to_1 = tree1.query(surface2, k=1)[0]

            # Points within tolerance are considered overlapping
            overlap_1 = np.sum(dist_1_to_2 <= self.tolerance)
            overlap_2 = np.sum(dist_2_to_1 <= self.tolerance)

            denom = len(surface1) + len(surface2)
            if denom == 0:
                continue

            sdc_label = (overlap_1 + overlap_2) / denom
            sdc_list.append(sdc_label)

        return float(np.mean(sdc_list)) if sdc_list else float('nan')


class TRE(nn.Module):
    """Implements Target to Registration Error (TRE)
    between two sets of keypoints.

    Main Assumptions:
    1- axis ordering of the deformation grid and the keypoints
    2- voxel spacing
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        fixed_keypoints: torch.Tensor,
        moving_keypoints: torch.Tensor,
        deformation_grid: torch.Tensor,
        id_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed_keypoints:
            [1, N, 3] - fixed keypoints (ground truth),
            in voxel coordinates [x, y, z]

        :param moving_keypoints:
            [1, N, 3] - moving keypoints (to be warped),
            in voxel coordinates [x, y, z]

        :param deformation_grid:
            [1, D, H, W, 3] - normalized grid for sampling

        :param id_grid:
            [1, D, H, W, 3] - identity grid, same normalization

        :returns:
            TRE (mean Euclidean error), moved keypoints, displacement field at keypoints
        """
        _, d, h, w, _ = deformation_grid.shape

        # De-normalize the deformation grid
        deformation_grid[..., 0] = (deformation_grid[..., 0] * (w - 1) / 2) + ((w - 1) / 2)
        deformation_grid[..., 1] = (deformation_grid[..., 1] * (h - 1) / 2) + ((h - 1) / 2)
        deformation_grid[..., 2] = (deformation_grid[..., 2] * (d - 1) / 2) + ((d - 1) / 2)

        flow = deformation_grid - id_grid  # shape [1, D, H, W, 3]

        # Convert kp2 to normalized grid coordinates [-1, 1] for sampling
        kp2_norm = moving_keypoints.clone()
        kp2_norm[..., 0] = (kp2_norm[..., 0] / (w - 1)) * 2 - 1
        kp2_norm[..., 1] = (kp2_norm[..., 1] / (h - 1)) * 2 - 1
        kp2_norm[..., 2] = (kp2_norm[..., 2] / (d - 1)) * 2 - 1

        # grid_sample expects [B, C, D, H, W] input and [B, N, 1, 1, 3] grid
        flow = flow.permute(0, 4, 1, 2, 3)
        kp2_grid = kp2_norm.view(1, -1, 1, 1, 3)

        # Interpolate displacements at kp2 locations
        # final shape: [1, N, 3]
        displacement_field = F.grid_sample(flow, kp2_grid, align_corners=True).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # apply displacement to kp2
        moved_coords = moving_keypoints - displacement_field  

        # Compute TRE = average Euclidean distance
        tre = ((fixed_keypoints - moved_coords) ** 2).sum(dim=-1).sqrt().mean()

        return tre, moved_coords, displacement_field


class SSIM(nn.Module):
    """Implements the Structural Similarity Index Measure (SSIM)
    for 3D images. The code is inspired by
    
    https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/
    """
    def __init__(
        self,
        dim: int=3,
        window_size: int=11
    ) -> None:
        """
        :param window_size:
            The window size on which local correlations are computed.
        """
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.channels = 1
        self.kernel = self.__create_kernel(window_size, self.channels)
        self.conv = F.conv3d if dim == 3 else F.conv2d
  
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        :param img1:
            The first image [B, C, D, H, W] or [B, D, H, W].
        
        :param img2:
            The second image [B, C, D, H, W] or [B, D, H, W].
        
        :returns:
            The SSIM between img1 and img2.
        """
        n_channels = img1.size(1)
        
        if n_channels == self.channels and self.kernel.data.type == img1.data.type:
            window = self.kernel
        else:
            window = self.__create_kernel(self.window_size, n_channels)
            
        window = window.to(img1.device)
        window = window.type_as(img1)
        
        ssim = self.__ssim(img1, img2, window, self.window_size, n_channels)
        
        return ssim
       
    def __create_kernel(
            self,
            window_size: int,
            channels: int
        ) -> torch.Tensor:
        """
        :param window_size:
            The window size of the Gaussian kernels.

        :pram channels:
            Number of image channels.
        
        :returns:
            A 2D/3D Gaussian kernel
        """
        kernel1d = self.__gaussian(window_size, 1.5).unsqueeze(1)
        kernel2d = torch.mm(kernel1d, kernel1d.t()).unsqueeze(0).unsqueeze(0)

        if self.dim == 2:
            kernel = kernel2d.expand(channels, 1, window_size, window_size).contiguous()
            return kernel
        
        kernel3d = torch.mm(kernel1d, kernel2d.reshape(1, -1))
        kernel3d = kernel3d.reshape(window_size, window_size, window_size).unsqueeze(0).unsqueeze(0)
        
        kernel = kernel3d.expand(channels, 1, window_size, window_size, window_size).contiguous()
        
        return kernel    
    
    def __gaussian(
            self,
            window_size: int,
            sigma: float
        ) -> torch.Tensor:
        """
        :param window_size:
            The window size of the Gaussian kernel.
        
        :param sigma:
            The sigma parameter of the Gaussian distribution.

        :returns:
            A 1D Gaussian kernel.
        """
        torch_sigma = torch.tensor(sigma ** 2)
        torch_win_size = torch.tensor(window_size // 2)

        kernel_core = lambda x: -(x - torch_win_size) ** 2 / (2 * torch_sigma)

        kernel = torch.Tensor([torch.exp(kernel_core(x)) for x in range(window_size)])
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def __ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        kernel: torch.Tensor, 
        kernel_size: int,
        channels: int
    ) -> torch.Tensor:
        """
        :param img1:
            The first image [B, C, D, H, W] or [B, C, H, W].
        
        :param img2:
            The second image [B, C, D, H, W] or [B, C, H, W].
        
        :param kernel:
            The Gaussian kernel to be used.

        :param kernel_size:
            The Gaussian kernel size.
        
        :param channels:
            The number of image channels.
        
        :returns:
            The SSIM between img1 and img2.
        """
        mu1 = self.conv(img1, kernel, padding=kernel_size // 2, groups=channels)
        mu2 = self.conv(img2, kernel, padding=kernel_size // 2, groups=channels)
        
        mu1_squared = mu1 ** 2
        mu2_squared = mu2 ** 2
        
        mu1mu2 = mu1 * mu2
        
        sigma1 = self.conv(img1 * img1, kernel, padding=kernel_size // 2, groups=channels) - mu1_squared
        sigma2 = self.conv(img2 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu2_squared
        sigma12 = self.conv(img1 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu1mu2
        
        const1 = 0.01 ** 2
        const2 = 0.03 ** 2
        
        numerator = (2 * mu1mu2 + const1) * (2 * sigma12 + const2)
        denominator = (mu1_squared + mu2_squared + const1) * (sigma1 + sigma2 + const2)
        ssim_map = numerator / denominator

        return ssim_map.mean()


class SDLogJ(nn.Module):
    """Computes Standard Deviation of Logarithm of Jacobinan
    Determinant. The Jacobian determinant is computed by centeral
    defference.
    """
    def __init__(
        self,
        eps: float=1e-9
    ):
        """
        :param eps:
            The threshold to cutoff the determinants.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        deformation_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        :param deformation_grid:
            A 2D/3D [-1, 1] normalized deformation grid.
        
        :returns:
            SDLogJ of the deformation grid.
        """
        if len(deformation_grid.size()) == 4:
            return self.compute_sdlogj_2d(deformation_grid)
        elif len(deformation_grid.size()) == 5:
            return self.compute_sdlogj(deformation_grid)

    def compute_sdlogj(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """
        :param grid:
            The 3D deformation grid [B, D, H, W, 3]

        :returns:
            The SDLogJ of the 3D grid.
        """
        dd = (grid[:, 2:, 1:-1, 1:-1, :] - grid[:, :-2, 1:-1, 1:-1, :]) / 2.0
        dh = (grid[:, 1:-1, 2:, 1:-1, :] - grid[:, 1:-1, :-2, 1:-1, :]) / 2.0
        dw = (grid[:, 1:-1, 1:-1, 2:, :] - grid[:, 1:-1, 1:-1, :-2, :]) / 2.0

        J11 = dd[..., 0]; J12 = dh[..., 0]; J13 = dw[..., 0]
        J21 = dd[..., 1]; J22 = dh[..., 1]; J23 = dw[..., 1]
        J31 = dd[..., 2]; J32 = dh[..., 2]; J33 = dw[..., 2]

        det = J11 * (J22 * J33 - J23 * J32) - \
              J12 * (J21 * J33 - J23 * J31) + \
              J13 * (J21 * J32 - J22 * J31)

        return torch.std(torch.log(det.clamp(min=1e-9)))
    
    def compute_sdlogj_2d(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """
        :param grid:
            The 3D deformation grid [B, H, W, 2]

        :returns:
            The SDLogJ of the 2D grid.
        """
        dx = (grid[:, 1:-1, 2:, :] - grid[:, 1:-1, :-2, :]) / 2.0
        dy = (grid[:, 2:, 1:-1, :] - grid[:, :-2, 1:-1, :]) / 2.0

        J11 = dx[..., 0]; J12 = dy[..., 0]
        J21 = dx[..., 1]; J22 = dy[..., 1]

        det = J11 * J22 - J12 * J21

        return torch.std(torch.log(det.clamp(min=1e-9)))


class NCCLoss(nn.Module):
    """Implements Local Normalized Cross Correlation.
    Codes are mainly inspired by

    https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks
    """
    def __init__(
        self,
        win: int=7,
        eps: float=1e-5
    ) -> None:
        """
        :param win:
            The local window size.
        
        :param eps:
            The denominator correction to avoid division by zero.
        """
        super().__init__()
        
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor
    ) -> torch.Tensor:
        """
        :param fixed:
            The fixed image.
            Accepted shapes: [B, C, D, H, W] or [B, C, H, W]

        :param moving:
            The moving image.
            Accepted shapes: [B, C, D, H, W] or [B, C, H, W]
        """
        device = fixed.device

        # converto to grayscale if the images are colored
        if fixed.size(1) > 1:
            fixed = rgb_to_grayscale(fixed)
        if moving.size(1) > 1:
            moving = rgb_to_grayscale(moving)
        
        ndims = len(fixed.shape) - 2
        
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        
        if ndims == 3:
            weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=device, requires_grad=False)
        else:
            weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=device, requires_grad=False)
            
        conv_fn = F.conv3d if ndims == 3 else F.conv2d

        # compute CC squares
        fixed2 = fixed * fixed
        moving2 = moving * moving
        fixed_moving = fixed * moving

        # compute filters
        # compute local sums via convolution
        fixed_sum = conv_fn(fixed, weight, padding=int(win_size/2))
        moving_sum = conv_fn(moving, weight, padding=int(win_size/2))
        fixed2_sum = conv_fn(fixed2, weight, padding=int(win_size/2))
        moving2_sum = conv_fn(moving2, weight, padding=int(win_size/2))
        fixed_moving_sum = conv_fn(fixed_moving, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = fixed_sum/win_size
        u_J = moving_sum/win_size

        cross = fixed_moving_sum - u_J * fixed_sum - u_I * moving_sum + u_I * u_J * win_size
        I_var = fixed2_sum - 2 * u_I * fixed_sum + u_I * u_I * win_size
        J_var = moving2_sum - 2 * u_J * moving_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        loss = -1.0 * torch.mean(cc)
        
        return loss


class JacobianDeterminant(nn.Module):
    """Computes the percentage of voxels/pixels
    with non-positive jacobian determinant.
    """
    def __init__(self):
        super().__init__()

    def forward(self, deformation_grid: torch.Tensor) -> torch.Tensor:
        """
        :param deformation_grid: 
            2D or 3D [-1, 1] normalized deformation grid
            Accepted shapes: [B, D, H, W, 3], [B, H, W, 2]

        :returns:
            The percentage of negative determinants
        """
        neg_dets_percentage = jacobian_determinant(deformation_grid)
        
        return neg_dets_percentage


class HD95(nn.Module):
    """Compute the 95th percentile Hausdorff Distance (HD95)
    for two segmentation maps.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        seg_map1: torch.Tensor,
        seg_map2: torch.Tensor,
        spacing: tuple=(1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        :param seg_map1:
            First segmentation map.
            Accepted shapes: [1, 1, D, H, W] or [1, 1, H, W]

        :param seg_map2:
            First segmentation map.
            Accepted shapes: [1, 1, D, H, W] or [1, 1, H, W]

        :param spacing:
            Voxel spacing in (Z, Y, X) pr (Y, X) order

        :returns:
            The HD95 metric
        """
        
        # Convert to numpy
        seg1_np = seg_map1[0, 0].cpu().numpy()
        seg2_np = seg_map2[0, 0].cpu().numpy()

        labels = np.unique(np.concatenate([np.unique(seg1_np), np.unique(seg2_np)]))
        labels = labels[labels != 0]  # Remove background
        
        hd95_list = []

        for label in labels:
            mask1 = (seg1_np == label)
            mask2 = (seg2_np == label)

            surface1 = np.array(np.where(scipy.ndimage.binary_erosion(mask1) ^ mask1)).T
            surface2 = np.array(np.where(scipy.ndimage.binary_erosion(mask2) ^ mask2)).T

            if len(surface1) == 0 or len(surface2) == 0:
                continue

            # Apply voxel spacing
            surface1 = surface1 * np.array(spacing)
            surface2 = surface2 * np.array(spacing)

            # Use KDTree for nearest neighbor search (efficient distance computation)
            tree1 = cKDTree(surface1)
            tree2 = cKDTree(surface2)

            # Compute nearest neighbor distances
            dist_1_to_2 = tree1.query(surface2, k=1)[0]  # Closest distance from surface2 to surface1
            dist_2_to_1 = tree2.query(surface1, k=1)[0]  # Closest distance from surface1 to surface2

            # Compute 95th percentile distance
            hd95_label = max(np.percentile(dist_1_to_2, 95), np.percentile(dist_2_to_1, 95))
            hd95_list.append(hd95_label)

        return np.mean(hd95_list) if hd95_list else float('nan')


class NGFLoss(nn.Module):
    """Computes the 3D Normalized Gradient Field Loss
    """
    def __init__(
        self,
        eps: float=1e-8,
        reduction: str='mean'
    ):
        """
        :param eps:
            Stability correction.
        
        :param reduction:
            The method to compute final value.
        """
        super().__init__()
        assert reduction in ('mean', 'sum')

        self.eps = float(eps)
        self.reduction = reduction

        kernel_x = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel_y = torch.zeros_like(kernel_x)
        kernel_z = torch.zeros_like(kernel_x)

        # for x: differences along last axis (width)
        kernel_x[0, 0, 1, 1, 0] = -0.5
        kernel_x[0, 0, 1, 1, 2] = 0.5
        
        # for y: differences along height
        kernel_y[0, 0, 1, 0, 1] = -0.5
        kernel_y[0, 0, 1, 2, 1] = 0.5
        
        # for z: differences along depth
        kernel_z[0, 0, 0, 1, 1] = -0.5
        kernel_z[0, 0, 2, 1, 1] = 0.5

        # register kernels as buffers so they move to device with the module
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        self.register_buffer('kernel_z', kernel_z)

    def _image_to_single_channel(
        self,
        img: torch.Tensor
    ) -> torch.Tensor:
        """
        :param img:
            The input image.
            Accepted shape: [B, C, D, H, W]
        """
        num_channels = img.shape[1]
        if num_channels == 1:
            return img
        else:
            return img.mean(dim=1, keepdim=True)

    def _gradients(
        self,
        img: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradients with conv3d central difference kernels.

        :param img:
            The input image.
            Accepted shape: [B, 1, D, H, W]

        :returns:
            Image gradients [B, 3, D, H, W] where channels are [gx, gy, gz]
        """
        gx = F.conv3d(img, self.kernel_x, padding=1)
        gy = F.conv3d(img, self.kernel_y, padding=1)
        gz = F.conv3d(img, self.kernel_z, padding=1)

        return torch.cat([gx, gy, gz], dim=1)

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: torch.Tensor=None
    ) -> torch.Tensor:
        """
        :param fixed:
            The fixed image [B, C, D, H, W]

        :param moving:
            The moving image [B, C, D, H, W]

        :param mask:
            Optional boolean/float mask where 1 indicates valid voxels
            Accepted shapes: [B, 1, D, H, W] or [B, D, H, W]

        :returns:
            NGF loss scalar
        """
        F_img = self._image_to_single_channel(fixed)
        M_img = self._image_to_single_channel(moving)

        # compute gradients
        gF = self._gradients(F_img)  # (B,3,D,H,W)
        gM = self._gradients(M_img)

        # normalize gradient vectors per voxel
        # compute squared norm of gradient vectors: sum over vector components
        gF_sq = (gF * gF).sum(dim=1, keepdim=True)  # (B,1,D,H,W)
        gM_sq = (gM * gM).sum(dim=1, keepdim=True)

        # norm with eps for stability
        nF = gF / torch.sqrt(gF_sq + (self.eps ** 2))
        nM = gM / torch.sqrt(gM_sq + (self.eps ** 2))

        # dot product per voxel
        dot = (nF * nM).sum(dim=1, keepdim=True)  # (B,1,D,H,W)

        # per-voxel NGF energy: 1 - (dot)^2 (in [0,1])
        ngf_per_voxel = 1.0 - dot * dot  # (B,1,D,H,W)

        # apply mask if provided
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)  # (B,1,D,H,W)
            if mask.shape != ngf_per_voxel.shape:
                raise ValueError("mask must have shape (B,1,D,H,W) or (B,D,H,W)")
            maskf = mask.to(dtype=ngf_per_voxel.dtype)
            ngf_per_voxel = ngf_per_voxel * maskf
            valid_voxels = maskf.sum()
            if valid_voxels == 0:
                # no valid voxels; return zero to avoid division by zero
                return torch.tensor(0.0, device=ngf_per_voxel.device, dtype=ngf_per_voxel.dtype)
        else:
            valid_voxels = torch.tensor(ngf_per_voxel.numel(), device=ngf_per_voxel.device, dtype=ngf_per_voxel.dtype)

        if self.reduction == "sum":
            loss = ngf_per_voxel.sum()
        else:  # mean
            loss = ngf_per_voxel.sum() / valid_voxels

        return loss


class ASSD(nn.Module):
    """Compute the Average Symmetric Surface Distance (ASSD)
    for two segmentation maps.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        seg_map1: torch.Tensor,
        seg_map2: torch.Tensor,
        spacing: tuple=(1.0, 1.0, 1.0)
    ):
        """
        :param seg_map1:
            First segmentation map [1, 1, D, H, W] oe [1, 1, H, W].

        :param seg_map2:
            Second segmentation map [1, 1, D, H, W] or [1, 1, H, W].

        :param spacing:
            Voxel spacing in (Z, Y, X) or (Y, X) order.

        :returns:
            The ASSD metric.
        """
        
        # Convert to numpy
        seg1_np = seg_map1[0, 0].cpu().numpy()
        seg2_np = seg_map2[0, 0].cpu().numpy()

        # Get unique labels (excluding background)
        labels = np.unique(np.concatenate([np.unique(seg1_np), np.unique(seg2_np)]))
        labels = labels[labels != 0]

        assd_list = []

        for label in labels:
            mask1 = (seg1_np == label)
            mask2 = (seg2_np == label)

            # Extract surfaces using binary erosion
            surface1 = np.array(np.where(mask1 ^ scipy.ndimage.binary_erosion(mask1))).T
            surface2 = np.array(np.where(mask2 ^ scipy.ndimage.binary_erosion(mask2))).T

            if len(surface1) == 0 or len(surface2) == 0:
                continue

            # Apply voxel spacing
            surface1 = surface1 * np.array(spacing)
            surface2 = surface2 * np.array(spacing)

            # Use KDTree for efficient nearest-neighbor distance computation
            tree1 = cKDTree(surface1)
            tree2 = cKDTree(surface2)

            dist_1_to_2 = tree1.query(surface2, k=1)[0]
            dist_2_to_1 = tree2.query(surface1, k=1)[0]

            # Compute mean symmetric surface distance for this label
            assd_label = (dist_1_to_2.mean() + dist_2_to_1.mean()) / 2.0
            assd_list.append(assd_label)

        return np.mean(assd_list) if assd_list else float('nan')
