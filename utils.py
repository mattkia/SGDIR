import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def warp(image: torch.Tensor,
         deformation_grid: torch.Tensor,
         nearest: bool=False) -> torch.Tensor:

    mode = 'nearest' if nearest else 'bilinear'

    warped = F.grid_sample(image, 
                           deformation_grid, 
                           padding_mode='reflection', 
                           align_corners=True, 
                           mode=mode)

    return warped

def jacobian_determinant(deformation_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of the Jacobian numerically, given the deformed
    output grid and returns the percentage of negative values
    
    Args:
        deformation_grid (torch.Tensor): [B, D, H, W, 3]

    Returns:
        torch.Tensor: the percentage of negative determinants
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

def save_determinant_intensity(deformed_grid: torch.Tensor, save_path: str, prefix: str=None) -> None:
    """
    This function receives the deformation grid output by FlowNet3D and plots the
    determinants of its Jacobian for better analysis.
    
    Args:
        deformed_grid (torch.Tensor): the deformation grid output by FlowNet3D with shape [1, D, H, W, 3]
        save_path (str): the directory where the plot is saved
        prefix (str): a prefix to the file name that is to be saved
    """
    dy = deformed_grid[:, 1:, :-1, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dx = deformed_grid[:, :-1, 1:, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dz = deformed_grid[:, :-1, :-1, 1:, :] - deformed_grid[:, :-1, :-1, :-1, :]

    det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy [:, :, :,:, 2] * dz[:, :, :, :, 0])
    det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy [:, :, :,:, 1] * dz[:, :, :, :, 0])

    determinants = det0 - det1 + det2
    
    determinants[determinants >= 0] = 0
    determinants = torch.abs(determinants)
    
    mid_size = determinants.size(2) // 2
    for i in range(determinants.size(0)):
        filename = f'det_{i}.png'
        if prefix is not None:
            filename = 'batch_' + prefix + '_' + filename
        plt.imsave(f'{save_path}/{filename}', determinants[i, :, mid_size, :].cpu().numpy())
        plt.close()

def dsc_score(seg_map1: torch.Tensor, seg_map2: torch.Tensor, bg: bool=False, structured: bool=False) -> torch.Tensor:
    """
    This function receives the segmentation maps of two 2d or 3d images and computes the dice score coefficient
    between the two images.

    Args:
        seg_map1 (torch.Tensor): [B, C, w, h, d] or [B, C, h, w], first segmentation map
        seg_map2 (torch.Tensor): [B, C, w, h, d] or [B, C, h, w], first segmentation map
        bg (bool) - Defaults to False: consideres background matches if set True
        structured (bool) - Defaults to False: if True, the dice over each structure is calculated, otherwise the
                                               volumetric dice is computed
    Returns:
        torch.Tensor: the dice score between the two images
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

def rolling_dice(network: torch.nn.Module, fixed_img: torch.Tensor, moving_img: torch.Tensor, 
                 fixed_seg: torch.Tensor, moving_seg: torch.Tensor, 
                 grid: torch.Tensor, save_path: str, num_frames: int=100) -> None:
    """
    This functions receives an instance of the FlowNet3D, the fixed and moving images, the
    fixed and moving segmentations masks, and computes the dice score along the trajectory of warping
    (i.e., the dice score of warped images at each time step). The results are saved into a 
    fwd_rolling_dice.txt file
    
    Args:
        network (torch.nn.Module): an instanc of the FlowNet3D
        fixed_img (torch.Tensor): fixed image [1, 1, D, H, W]
        moving_img (torch.Tensor): moving image [1, 1, D, H, W]
        fixed_seg (torch.Tensor): fixed image segmentation mask [1, 1, D, H, W]
        moving_seg (torch.Tensor): moving image segmentation mask [1, 1, D, H, W]
        grid (torch.Tensor): the initial grid [1, D, H, W, 3]
        save_path (str): the directory in which the results will be saved
        num_frames (int) - Defaluts to 100: the number of time steps at which the dice score should be computed
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

def evaluate_semi_group_property(network: nn.Module, fixed_img: torch.Tensor, moving_img: torch.Tensor, grid: torch.Tensor, save_path: str):
    """
    This function receives the an instance of FlowNet3D along with the fixed and moving images and evaluates the percentage
    negative Jacobian determinants and how well the semigroup property holds. 
    The final results are saved into jacs.tx and errors.txt files which contain the information on Jacobian determinants
    and the semigroup property errors, respectively.
    
    Args:
        network (torch.nn.Module): an instanc of the FlowNet3D
        fixed_img (torch.Tensor): fixed image [1, 1, D, H, W]
        moving_img (torch.Tensor): moving image [1, 1, D, H, W]
        grid (torch.Tensor): the initial grid [1, D, H, W, 3]
        save_path (str): the directory in which the results will be saved
    """
    t = torch.rand(100, device=fixed_img.device) * 2. - 1.
    s = torch.rand(100, device=fixed_img.device) * 2. - 1. - t
    prop_errors = np.zeros((100, 100), dtype=np.float32)
    jacs = np.zeros((100,), dtype=np.float32)
    with torch.no_grad():
        for i, t1 in enumerate(t):
            t1 = t1.reshape(1)
            flow_t = t1 * network.velocity(fixed_img, moving_img, t1)
            grid_t = network.make_grid(flow_t, grid)
            jacs[i] = jacobian_determinant(grid_t)
            for j, t2 in enumerate(s):
                t2 = t2.reshape(1)
                flow_s = t2 * network.velocity(fixed_img, moving_img, t2)
                composed_flow = network.compose(flow_t, flow_s, grid)
                grid = network.make_grid(composed_flow, grid)
                new_flow = (t1 + t2) * network.velocity(fixed_img, moving_img, t1 + t2)
                new_grid = network.make_grid(new_flow, grid)
                prop_errors[i, j] = torch.mean((grid - new_grid) ** 2).cpu().numpy()
    
    np.savetxt(f'{save_path}/errors.txt', prop_errors)
    np.savetxt(f'{save_path}/jacs.txt', jacs)

def grid_denormalizer(deformation_grid: torch.Tensor) -> torch.Tensor:
    shape = deformation_grid.shape[1:-1][::-1]
    
    denormalized_grid = deformation_grid.clone()

    for i in range(len(shape)):
        denormalized_grid[..., i] = (denormalized_grid[..., i] * (shape[i] - 1)) / 2. + ((shape[i] - 1) / 2)
    
    return denormalized_grid

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or tuple (D, H, W)
    return:
    pos_embed: [D*H*W, embed_dim]
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

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or tuple (H, W)
    return:
    pos_embed: [H*W, embed_dim]
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

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: [3, 1, D, H, W]
    return: [D*H*W, embed_dim]
    """
    assert embed_dim % 3 == 0, "Embed dimension must be divisible by 3 for 3D sin-cos embedding"

    # split dim equally across D, H, W
    dim_each = embed_dim // 3

    emb_d = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (D*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])  # (D*H*W, D/3)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: [2, 1, H, W]
    return: [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0, "Embed dimension must be divisible by 2 for 3D sin-cos embedding"

    # split dim equally across H, W
    dim_each = embed_dim // 2

    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: position grid (any shape)
    return: [pos_size, embed_dim]
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
