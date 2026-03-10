import torch
import scipy.ndimage

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from typing import List
from scipy.spatial import cKDTree
from torchvision.transforms.functional import rgb_to_grayscale


class Dice(nn.Module):
    def __init__(self, structured: bool=True):
        super().__init__()

        self.structured = structured

    def forward(self, seg_map_1: torch.Tensor, 
                seg_map_2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_map_1: The segmentation map of the fixed image.
            seg_map_2: The segmentation map of the warped moving image.

        Returns
            torch.Tensor: A scaler representing the average DSC score accross all anatomical
                          regions.
        """
        
        device = seg_map_1.device
        batch_size = seg_map_1.size(0)        

        if self.structured:
            dsc_score = 0.
            
            for b in range(batch_size):
                labels = torch.unique(torch.cat((seg_map_1[b], seg_map_2[b])))
                labels = labels[torch.where(labels != 0)]

                dicem = torch.zeros(len(labels))
                for idx, lab in enumerate(labels):
                    top = 2 * torch.sum(torch.logical_and(seg_map_1 == lab, seg_map_2[b] == lab))
                    bottom = torch.sum(seg_map_1 == lab) + torch.sum(seg_map_2[b] == lab)
                    bottom = torch.maximum(bottom, torch.tensor(torch.finfo(float).eps, device=device))
                    dicem[idx] = top / bottom
                
                dsc_score += dicem.mean() / batch_size
        
        else:
            max_classes = max(torch.max(seg_map_1), torch.max(seg_map_2))
            
            denominator = 2 * torch.prod(torch.tensor(seg_map_1.shape[1:]))
            denominator = denominator.unsqueeze(0).to(seg_map_1.device)
            if batch_size > 1:
                denominator = denominator.repeat(batch_size, 1)
            
            denom_list = [seg_map_1[i][seg_map_1[i] != 0].size(0) + 
                          seg_map_2[i][seg_map_2[i] != 0].size(0) 
                          for i in range(batch_size)]
            
            denominator = torch.tensor(denom_list, device=device)
            seg_map_1[seg_map_1 == 0] = max_classes + 10
            seg_map_2[seg_map_2 == 0] = max_classes + 12
            
            numerator = 2 * (seg_map_1 == seg_map_2).flatten(start_dim=1).sum(dim=-1)
            
            dsc_score = (numerator / denominator).mean()


        return dsc_score


class DicePerStructure(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seg_map_1: torch.Tensor, 
                seg_map_2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_map_1: The segmentation map of the fixed image.
            seg_map_2: The segmentation map of the warped moving image.

        Returns
            torch.Tensor: A scaler representing the average DSC score per anatomical
                          regions.
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
    def __init__(self, tolerance_mm: float = 1.0):
        """
        Args:
            tolerance_mm (float): Distance threshold (in mm) for surfaces to be considered overlapping.
        """
        super().__init__()
        self.tolerance = tolerance_mm

    def forward(self, seg1: torch.Tensor,
                seg2: torch.Tensor,
                spacing: Tuple|List = (1.0, 1.0, 1.0)) -> float:
        """
        Compute the Surface Dice Coefficient (SDC) for two segmentation maps.

        Args:
            seg1 (torch.Tensor): First segmentation map (shape: [B, 1, D, H, W])
            seg2 (torch.Tensor): Second segmentation map (shape: [B, 1, D, H, W])
            spacing (tuple): Voxel spacing in (Z, Y, X) order

        Returns:
            float: Average Surface Dice Coefficient across labels and batch.
        """
        seg1_np = seg1[0, 0].cpu().numpy()
        seg2_np = seg2[0, 0].cpu().numpy()

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
    """
    This class computes the Target to Registration Error (TRE) between two sets of keypoints.
    Main Assumptions:
    1- axis ordering of the deformation grid and the keypoints
    2- voxel spacing
    """
    def __init__(self):
        super().__init__()

    def forward(self, fixed_keypoints: torch.Tensor,
                moving_keypoints: torch.Tensor,
                deformation_grid: torch.Tensor,
                id_grid: torch.Tensor
                ) -> torch.Tensor:
        
        """
        Args:
            fixed_keypoints: [1, N, 3] - fixed keypoints (ground truth), in voxel coordinates [x, y, z]
            moving_keypoints: [1, N, 3] - moving keypoints (to be warped), in voxel coordinates [x, y, z]
            deformation_grid: [1, D, H, W, 3] - normalized grid for sampling
            id_grid: [1, D, H, W, 3] - identity grid, same normalization

        Returns:
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


class SSIM3d(nn.Module):
    """
    This class implements the Structural Similarity Index Measure (SSIM) for 3d images.
    The codes are mainly refactored from
    
    https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/
    """
    def __init__(self, window_size: int=11,
                 size_average: bool=True) -> None:
        super().__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.kernel = self.__create_3d_kernel(window_size, self.channels)
  
    def forward(self, img1: torch.Tensor,
                img2: torch.Tensor) -> torch.Tensor:
        n_channels = img1.size(1)
        
        if n_channels == self.channels and self.kernel.data.type == img1.data.type:
            window = self.kernel
        else:
            window = self.__create_3d_kernel(self.window_size, n_channels)
            
        window = window.to(img1.device)
        window = window.type_as(img1)
        
        ssim = self.__ssim(img1, img2, window, self.window_size, n_channels, self.size_average)
        
        return ssim
       
    def __create_3d_kernel(self, window_size: int,
                           channels: int) -> torch.Tensor:
        kernel1d = self.__gaussian(window_size, 1.5).unsqueeze(1)
        kernel2d = torch.mm(kernel1d, kernel1d.t()).unsqueeze(0).unsqueeze(0)
        kernel3d = torch.mm(kernel1d, kernel2d.reshape(1, -1))
        kernel3d = kernel3d.reshape(window_size, window_size, window_size).unsqueeze(0).unsqueeze(0)
        
        kernel = kernel3d.expand(channels, 1, window_size, window_size, window_size).contiguous()
        
        return kernel    
    
    def __gaussian(self, window_size: int,
                   sigma: float) -> torch.Tensor:
        kernel = torch.Tensor([torch.exp(-(x - torch.tensor(window_size // 2))**2 / (2 * torch.tensor(sigma ** 2))) for x in range(window_size)])
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def __ssim(self, img1: torch.Tensor,
               img2: torch.Tensor,
               kernel: torch.Tensor, 
               kernel_size: int,
               channels: int,
               size_average: bool) -> torch.Tensor:
        mu1 = F.conv3d(img1, kernel, padding=kernel_size // 2, groups=channels)
        mu2 = F.conv3d(img2, kernel, padding=kernel_size // 2, groups=channels)
        
        mu1_squared = mu1 ** 2
        mu2_squared = mu2 ** 2
        
        mu1mu2 = mu1 * mu2
        
        sigma1 = F.conv3d(img1 * img1, kernel, padding=kernel_size // 2, groups=channels) - mu1_squared
        sigma2 = F.conv3d(img2 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu2_squared
        sigma12 = F.conv3d(img1 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu1mu2
        
        const1 = 0.01 ** 2
        const2 = 0.03 ** 2
        
        numerator = (2 * mu1mu2 + const1) * (2 * sigma12 + const2)
        denominator = (mu1_squared + mu2_squared + const1) * (sigma1 + sigma2 + const2)
        ssim_map = numerator / denominator
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class SSIM2d(nn.Module):
    """
    This class implements the Structural Similarity Index Measure (SSIM) for 2D images.
    Adapted from a 3D implementation.
    """
    def __init__(self, window_size: int = 11,
                 size_average: bool = True) -> None:
        super().__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.kernel = self.__create_2d_kernel(window_size, self.channels)
  
    def forward(self, img1: torch.Tensor,
                img2: torch.Tensor) -> torch.Tensor:
        n_channels = img1.size(1)
        
        if n_channels == self.channels and self.kernel.data.type() == img1.data.type():
            window = self.kernel
        else:
            window = self.__create_2d_kernel(self.window_size, n_channels)
            
        window = window.to(img1.device)
        window = window.type_as(img1)
        
        ssim = self.__ssim(img1, img2, window, self.window_size, n_channels, self.size_average)
        
        return ssim
       
    def __create_2d_kernel(self, window_size: int,
                           channels: int) -> torch.Tensor:
        kernel1d = self.__gaussian(window_size, 1.5).unsqueeze(1)
        kernel2d = torch.mm(kernel1d, kernel1d.t()).unsqueeze(0).unsqueeze(0)
        
        kernel = kernel2d.expand(channels, 1, window_size, window_size).contiguous()
        
        return kernel    
    
    def __gaussian(self, window_size: int,
                   sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([torch.exp(-(x - torch.tensor(window_size // 2)) ** 2 / (2 * torch.tensor(sigma ** 2)))
                              for x in range(window_size)])
        return gauss / gauss.sum()
    
    def __ssim(self, img1: torch.Tensor,
               img2: torch.Tensor,
               kernel: torch.Tensor, 
               kernel_size: int,
               channels: int,
               size_average: bool) -> torch.Tensor:
        mu1 = F.conv2d(img1, kernel, padding=kernel_size // 2, groups=channels)
        mu2 = F.conv2d(img2, kernel, padding=kernel_size // 2, groups=channels)
        
        mu1_squared = mu1 ** 2
        mu2_squared = mu2 ** 2
        mu1mu2 = mu1 * mu2
        
        sigma1 = F.conv2d(img1 * img1, kernel, padding=kernel_size // 2, groups=channels) - mu1_squared
        sigma2 = F.conv2d(img2 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu2_squared
        sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu1mu2
        
        const1 = 0.01 ** 2
        const2 = 0.03 ** 2
        
        numerator = (2 * mu1mu2 + const1) * (2 * sigma12 + const2)
        denominator = (mu1_squared + mu2_squared + const1) * (sigma1 + sigma2 + const2)
        ssim_map = numerator / denominator
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class SDLogJ(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, deformation_grid: torch.Tensor) -> torch.Tensor:
        if len(deformation_grid.size()) == 4:
            dy = deformation_grid[:, 1:, :-1, :] - deformation_grid[:, :-1, :-1, :]
            dx = deformation_grid[:, :-1, 1:, :] - deformation_grid[:, :-1, :-1, :]

            determinants = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        elif len(deformation_grid.size()) == 5:
            dy = deformation_grid[:, 1:, :-1, :-1, :] - deformation_grid[:, :-1, :-1, :-1, :]
            dx = deformation_grid[:, :-1, 1:, :-1, :] - deformation_grid[:, :-1, :-1, :-1, :]
            dz = deformation_grid[:, :-1, :-1, 1:, :] - deformation_grid[:, :-1, :-1, :-1, :]

            det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
            det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy [:, :, :,:, 2] * dz[:, :, :, :, 0])
            det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy [:, :, :,:, 1] * dz[:, :, :, :, 0])

            determinants = det0 - det1 + det2

        # remove boundary artifacts
        valid = determinants > self.eps
        if valid.sum() == 0:
            return torch.zeros((), device=deformation_grid.device)

        logJ = torch.log(determinants[valid])
        return logJ.std()


class NCCLoss(nn.Module):
    """
    local (over window) normalized cross correlation
    codes taken from SYMNet repo and modified
    """
    def __init__(self, win: int=7, eps: float=1e-5) -> None:
        super().__init__()
        
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, fixed: torch.Tensor,
                moving: torch.Tensor) -> torch.Tensor:
        assert len(fixed.shape) == 4 or len(fixed.shape) == 5, '[!] Expected shape of [B, C, H. W] or [B, C, D, H, W]'
        assert len(moving.shape) == 4 or len(moving.shape) == 5, '[!] Expected shape of [B, C, H. W] or [B, C, D, H, W]'
        
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
    def __init__(self):
        super().__init__()

    def forward(self, deformation_grid: torch.Tensor) -> torch.Tensor:
        if len(deformation_grid.size()) == 4:
            dy = deformation_grid[:, 1:, :-1, :] - deformation_grid[:, :-1, :-1, :]
            dx = deformation_grid[:, :-1, 1:, :] - deformation_grid[:, :-1, :-1, :]

            determinants = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        elif len(deformation_grid.size()) == 5:
            dy = deformation_grid[:, 1:, :-1, :-1, :] - deformation_grid[:, :-1, :-1, :-1, :]
            dx = deformation_grid[:, :-1, 1:, :-1, :] - deformation_grid[:, :-1, :-1, :-1, :]
            dz = deformation_grid[:, :-1, :-1, 1:, :] - deformation_grid[:, :-1, :-1, :-1, :]

            det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
            det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy [:, :, :,:, 2] * dz[:, :, :, :, 0])
            det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy [:, :, :,:, 1] * dz[:, :, :, :, 0])

            determinants = det0 - det1 + det2
        else:
            raise Exception('[!] Invalid shape for the deformation grid')
        
        num_neg_dets = len(determinants[determinants <= 0])
        total_points = torch.prod(torch.tensor(determinants.size(), device=determinants.device))
        
        neg_dets_percentage = num_neg_dets * 100 / total_points
        
        return neg_dets_percentage


class HD95(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seg1: torch.Tensor,
                seg2: torch.Tensor,
                spacing: Tuple|List=(1.0, 1.0, 1.0)):
        """
        Compute the 95th percentile Hausdorff Distance (HD95) for two segmentation maps.

        Args:
            seg1 (torch.Tensor): First segmentation map (shape: [D, H, W])
            seg2 (torch.Tensor): Second segmentation map (shape: [D, H, W])
            spacing (tuple): Voxel spacing in (Z, Y, X) order

        Returns:
            float: The HD95 metric
        """
        
        # Convert to numpy
        seg1_np = seg1[0, 0].cpu().numpy()
        seg2_np = seg2[0, 0].cpu().numpy()

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
    def __init__(self, eps: float=1e-8,
                 reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'sum')
        self.eps = float(eps)
        self.reduction = reduction

        # We will build full 3D kernels with correct axes:
        # kx should vary on last dimension (W), ky on H, kz on D.
        # Create 3x3x3 kernels with central-difference along each axis.
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
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.register_buffer("kernel_z", kernel_z)

    def _image_to_single_channel(self, img: torch.Tensor) -> torch.Tensor:
        # if multi-channel, average channels -> produce shape (B,1,D,H,W)
        if img.dim() != 5:
            raise ValueError("Expected image shape (B, C, D, H, W)")
        B, C, D, H, W = img.shape
        if C == 1:
            return img
        else:
            # mean over channels but keep channel dimension
            return img.mean(dim=1, keepdim=True)

    def _gradients(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients with conv3d central difference kernels.
        img: (B,1,D,H,W)
        returns: grad (B, 3, D, H, W) where channels are [gx, gy, gz]
        """
        # padding=1 to keep same shape (same padding)
        gx = F.conv3d(img, self.kernel_x, padding=1)
        gy = F.conv3d(img, self.kernel_y, padding=1)
        gz = F.conv3d(img, self.kernel_z, padding=1)
        return torch.cat([gx, gy, gz], dim=1)

    def forward(self,
                fixed: torch.Tensor,
                moving: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        fixed, moving: (B, C, D, H, W)
        mask: optional (B, 1, D, H, W) or (B, D, H, W) boolean/float mask where 1 indicates valid voxels
        returns: scalar loss
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
    def __init__(self):
        super().__init__()

    def forward(self, seg1: torch.Tensor,
                seg2: torch.Tensor,
                spacing: Tuple | List = (1.0, 1.0, 1.0)):
        """
        Compute the Average Symmetric Surface Distance (ASSD) for two segmentation maps.

        Args:
            seg1 (torch.Tensor): First segmentation map (shape: [D, H, W])
            seg2 (torch.Tensor): Second segmentation map (shape: [D, H, W])
            spacing (tuple): Voxel spacing in (Z, Y, X) order

        Returns:
            float: The ASSD metric
        """
        
        # Convert to numpy
        seg1_np = seg1[0, 0].cpu().numpy()
        seg2_np = seg2[0, 0].cpu().numpy()

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
