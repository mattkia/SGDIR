"""Implementations of the helper and utility functions

The codes for positional encodings are inspired by
https://github.com/facebookresearch/DiT
"""

import torch
from typing import Union

import numpy as np
import torch.nn.functional as F


def warp(image: torch.Tensor,
         deformation_grid: torch.Tensor,
         nearest: bool=False) -> torch.Tensor:
    """Implements image warping by a deformation grid.

    :param image: 
        2D or 3D image
        Acceptable shapes: [B, C, D, H, W], [B, C, H, W]
    
    :param deformation_grid:
        2D or 3D [-1, 1] normalized deformation grid
        Acceptade shapes: [B, D, H, W, 3], [B, H, W, 2]
    
    :param nearest:
        Whether to use nearest neighbor interpolation.
        Set to True for warping segmentation masks
    
    :returns:
        The warped image by the deformation grid
    """

    mode = 'nearest' if nearest else 'bilinear'

    warped = F.grid_sample(image, 
                           deformation_grid, 
                           padding_mode='reflection', 
                           align_corners=True, 
                           mode=mode)

    return warped

def jacobian_determinant(deformation_grid: torch.Tensor) -> torch.Tensor:
    """Computes the percentage of voxels/pixels having non-positive
    Jacobian determinant.
    
    :param deformation_grid: 
        2D or 3D [-1, 1] normalized deformation grid
        Accepted shapes: [B, D, H, W, 3], [B, H, W, 2]

    :returns:
        The percentage of negative determinants
    """
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
    
    num_neg_dets = len(determinants[determinants <= 0])
    total_points = torch.prod(torch.tensor(determinants.size(), device=determinants.device))
    
    neg_dets_percentage = num_neg_dets * 100 / total_points
    
    return neg_dets_percentage

def dsc_score(seg_map1: torch.Tensor,
              seg_map2: torch.Tensor,
              bg: bool=False,
              structured: bool=False) -> torch.Tensor:
    """Receives the segmentation maps of two 2d or 3d images
    and computes the dice score coefficient between the two maps.

    :param seg_map1: 
        [B, C, D, H, W] or [B, C, H, W]; first segmentation map

    :param seg_map2:
        [B, C, D, H, W] or [B, C, H, W], second segmentation map

    :param bg:
        Consideres background matches if set True. Default False

    :param structured:
        If True, the dice over each structure is calculated,
        otherwise the volumetric dice is computed. Defalut False

    :returns:
        The dice score between the two segmentation maps
    """
    if not structured:
        max_classes = max(torch.max(seg_map1), torch.max(seg_map2))
        
        batch_size = seg_map1.size(0)
        
        denominator = 2 * torch.prod(torch.tensor(seg_map1.shape[1:]))
        denominator = denominator.unsqueeze(0).to(seg_map1.device)
        if batch_size > 1:
            denominator = denominator.repeat(batch_size, 1)
        
        if not bg:
            denom_list = [seg_map1[i][seg_map1[i] != 0].size(0) + seg_map2[i][seg_map2[i] != 0].size(0) for i in range(batch_size)]
            denominator = torch.tensor(denom_list, device=seg_map1.device)
            seg_map1[seg_map1 == 0] = max_classes + 10
            seg_map2[seg_map2 == 0] = max_classes + 12
        
        numerator = 2 * (seg_map1 == seg_map2).flatten(start_dim=1).sum(dim=-1)
        
        dsc_score = (numerator / denominator).mean()
    else:
        dsc_score = 0.
        batch_size = seg_map1.size(0)
        
        for b in range(batch_size):
            labels = torch.unique(torch.cat((seg_map1[b], seg_map2[b])))
            labels = labels[torch.where(labels != 0)]

            dicem = torch.zeros(len(labels))
            for idx, lab in enumerate(labels):
                top = 2 * torch.sum(torch.logical_and(seg_map1 == lab, seg_map2[b] == lab))
                bottom = torch.sum(seg_map1 == lab) + torch.sum(seg_map2[b] == lab)
                bottom = torch.maximum(bottom, torch.tensor(torch.finfo(float).eps, device=seg_map1.device))
                dicem[idx] = top / bottom
            
            dsc_score += dicem.mean() / batch_size
    
    return dsc_score

def rolling_dice(network: torch.nn.Module,
                 fixed_img: torch.Tensor,
                 moving_img: torch.Tensor, 
                 fixed_seg: torch.Tensor,
                 moving_seg: torch.Tensor, 
                 grid: torch.Tensor,
                 save_path: str,
                 num_frames: int=100) -> None:
    """Receives an instance of the SGDIR, the fixed and moving images,
    the fixed and moving segmentations masks, and computes the dice
    score along the trajectory of warping (i.e., the dice score of warped
    images at each time step). The results are saved into a
    fwd_rolling_dice.txt file

    :param network:
        An instanc of the SGDIR.

    :param fixed_img:
        Fixed image [1, 1, D, H, W] or [1, 1, H, W].

    :param moving_img:
        Moving image [1, 1, D, H, W] or [1, 1, H, W].

    :param fixed_seg:
        Fixed image segmentation mask [1, 1, D, H, W] or [1, 1, H, W].

    :param moving_seg:
        Moving image segmentation mask [1, 1, D, H, W] or [1, 1, H, W].

    :param grid:
        The identty grid [1, D, H, W, 3] or [1, H, W, 2].

    :param save_path:
        The directory in which the results will be saved.

    :param num_frames:
        The number of time steps at which the dice score should be computed
        Default 100.
    """    
    timesteps = torch.linspace(0, 1, num_frames)

    f_dice_scores = []
    f_jacs = []
    
    for t in timesteps:
        xyz = network(fixed_img, moving_img, grid, t.view(1).to(fixed_seg.device))
        xyzr = network(fixed_img, moving_img, grid, (t-1).view(1).to(fixed_seg.device))
        
        seg_Jw = F.grid_sample(moving_seg,
                               xyz,
                               mode='nearest',
                               align_corners=True,
                               padding_mode='reflection')
        seg_Iw = F.grid_sample(fixed_seg,
                               xyzr,
                               mode='nearest',
                               align_corners=True,
                               padding_mode='reflection')
    
        f_dice = dsc_score(seg_Jw, seg_Iw, structured=False)
        f_jac = jacobian_determinant(xyz)
    
        f_dice_scores.append(f_dice.item())
        f_jacs.append(f_jac.item())
    
    np.savetxt(f'{save_path}/fwd_rolling_dice.txt', f_dice_scores)
    np.savetxt(f'{save_path}/fwd_rolling_jacs.txt', f_jacs)

def grid_denormalizer(deformation_grid: torch.Tensor) -> torch.Tensor:
    """Denormalizes a [-1, 1] normalized grid back to its original size

    :param deformation_grid:
        A 2D or 3D [-1, 1] normalized grid.
        Accepted shapes: [B, D, H, W, 3] or [B, H, W, 2].
    
    :returns:
        A denormalized 2D or 3D grid with the same shape as input.
    """
    shape = deformation_grid.shape[1:-1][::-1]
    
    denormalized_grid = deformation_grid.clone()

    for i in range(len(shape)):
        denormalized_grid[..., i] = (denormalized_grid[..., i] * (shape[i] - 1)) / 2. + ((shape[i] - 1) / 2)
    
    return denormalized_grid

def get_3d_sincos_pos_embed(embed_dim: int, grid_size: Union[int, tuple[int, int, int]]) -> np.ndarray:
    """Generates 3D sinusoidal positional embeddings.

    :param embed_dim:
        The embedding dimension.

    :param grid_size:
        The grid size, either an int or tuple (D, H, W).
        
    :returns:
        The positional embeddings with shape [D*H*W, embed_dim].
    """
    if isinstance(grid_size, int):
        D = H = W = grid_size
    else:
        D, H, W = grid_size

    # create grid
    grid_d = np.arange(D, dtype=np.float32)
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, grid_d, indexing="ij")  # order: W, H, D
    grid = np.stack(grid, axis=0)  # [3, W, H, D]

    grid = grid.reshape([3, 1, D, H, W])  # [3, 1, D, H, W]

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed

def get_2d_sincos_pos_embed(embed_dim: int,
                            grid_size: Union[int, tuple[int, int]]) -> np.ndarray:
    """Generates 2D sinusoidal positional embeddings.

    :param embed_dim:
        The embedding dimension.

    :param grid_size:
        The grid size, either an int or tuple (H, W).
        
    :returns:
        The positional embeddings with shape [H*W, embed_dim].
    """
    if isinstance(grid_size, int):
        H = W = grid_size
    else:
        H, W = grid_size

    # create grid
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, indexing="ij")  # order: W, H, D
    grid = np.stack(grid, axis=0)  # [2, W, H]

    grid = grid.reshape([2, 1, H, W])  # [2, 1, H, W]

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim: int,
                                      grid: np.ndarray) -> np.ndarray:
    """Generates 3D sinusoidal positional embeddings from a given grid.

    :param embed_dim:
        The embedding dimension.

    :param grid:
        The position grid with shape [3, 1, D, H, W].
        
    :returns:
        The positional embeddings with shape [D*H*W, embed_dim].
    """
    assert embed_dim % 3 == 0, "Embed dimension must be divisible by 3 for 3D sin-cos embedding"

    # split dim equally across D, H, W
    dim_each = embed_dim // 3

    emb_d = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])  # (D*H*W, D/3)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim: int,
                                      grid: np.ndarray) -> np.ndarray:
    """Generates 2D sinusoidal positional embeddings from a given grid.

    :param embed_dim:
        The embedding dimension.

    :param grid:
        The position grid with shape [2, 1, H, W].
        
    :returns:
        The positional embeddings with shape [H*W, embed_dim].
    """
    assert embed_dim % 2 == 0, "Embed dimension must be divisible by 2 for 3D sin-cos embedding"

    # split dim equally across H, W
    dim_each = embed_dim // 2

    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                      pos: np.ndarray) -> np.ndarray:
    """Generates 1D sinusoidal positional embeddings from positions.

    :param embed_dim:
        The output dimension for each position.

    :param pos:
        The position grid of any shape.
        
    :returns:
        The positional embeddings with shape [pos_size, embed_dim].
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)  # frequencies

    pos = pos.reshape(-1)  # flatten
    out = np.einsum('m,d->md', pos, omega)  # outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
