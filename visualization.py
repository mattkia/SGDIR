"""Implementations of visualization helper functions
"""

import os
import torch
import shutil

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from skimage import io
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

from utils import warp
from utils import grid_denormalizer


def save_grid(
    deformation_grid: torch.Tensor,
    path: str,
    ax: plt.Axes=None,
    color: str='#404040',
    alpha: float=1.,
    down_factor: int=1
) -> None:
    """Saves the 2D/3D normalized deformation grid.

    :param path:
        complete/path/to/file.png
    
    :param ax:
        An instance of matplotlib ax.
        Used for overlaying the grid on exisiting figures.

    :param color:
        The hex color code for grid color.

    :param alpha:
        The opacity of the grid.

    :param down_factor:
        Down scaling factor for better visualization.
    """
    if len(deformation_grid.size()) == 4:
        xy = deformation_grid[0].detach().cpu().numpy()
    else:
        mid_size = deformation_grid.size(2) // 2
        xy = deformation_grid[0, :, mid_size, :, [0, 2]].detach().cpu().numpy()

    xy = xy[::down_factor, ::down_factor, :]
    if ax is None:
        fig, ax = plt.subplots()
        segs1 = xy
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, linewidth=0.8, color=color, alpha=alpha))
        ax.add_collection(LineCollection(segs2, linewidth=0.8, color=color, alpha=alpha))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.autoscale()
        
        fig.savefig(path)
        plt.close()
    else:
        segs1 = xy
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, linewidth=0.8, color=color, alpha=alpha))
        ax.add_collection(LineCollection(segs2, linewidth=0.8, color=color, alpha=alpha))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.autoscale()

def save_image(
    image: torch.Tensor,
    path: str,
    ax: plt.Axes=None,
    cmap: str='gray'
) -> None:
    """Saves 2D/3D images. For 3D images its saves the
    middle slice along the y-axis.

    :param image:
        The 2D/3D image.
        Acceptable shapes: [1, 1, D, H, W] or [1, 1, H, W]
    
    :param path:
        complete/path/to/file.png

    :param ax:
        An instance of matplotlib ax.
        Used for overlaying the image on exisiting figures.
    
    :param cmap:
        The color map used to display the image.
    """
    if len(image.size()) == 4:
        image = image[0, 0].detach().cpu().numpy()
    else:
        mid_size = image.size(3) // 2
        image = image[0, 0, :, mid_size, :].detach().cpu().numpy()
    
    if ax is None:
        fig, ax = plt.subplots()

        ax.imshow(image, cmap=cmap)
        fig.savefig(path)
        plt.close()
    else:
        ax.imshow(image, cmap=cmap)
    
def save_mask(
    image: torch.Tensor,
    path: str,
    ax: plt.Axes=None,
    alpha: float=1.,
    num_classes: int=35
) -> None:
    """Saves the 2D/3D segmentation masks. For 3D masks
    it saves the middle slice along the y-axis.

    :param image:
        The 2D/3D segmentation mask.
        Acceptable shapes: [1, 1, D, H, W] or [1, 1, H, W]
    
    :param path:
        complete/path/to/file.png

    :param ax:
        An instance of matplotlib ax.
        Used for overlaying the image on exisiting figures.
    
    :param alpha:
        The opacity of the mask.

    :param num_classes:
        The number of distinct labels in the mask.
        Useful for color coding.
    """
    pallete = [(0, 0, 0)] + sns.color_palette('Paired', n_colors=num_classes)
    cmap = ListedColormap(pallete)

    if len(image.size()) == 4:
        image = image[0, 0].detach().cpu().numpy()
    else:
        mid_size = image.size(3) // 2
        image = image[0, 0, :, mid_size, :].detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots()

        ax.imshow(image, cmap=cmap, interpolation='nearest', alpha=alpha)
        fig.savefig(path)
        plt.close()
    else:
        ax.imshow(image, cmap=cmap, interpolation='nearest', alpha=alpha)

def save_vector_field(
    deformation_grid: torch.Tensor,
    path: str,
    moving_keypoints: torch.Tensor,
    id_grid: torch.Tensor,
    ax: plt.Axes=None
) -> None:
    """Saves the 3D displacement field (used only for the LungCT dataset)
    It saves the middle slice along the y-axis.

    :param deformation_grid:
        The 3D deformation grid with shape [1, D, H, W, 3]

    :param path:
        complete/path/to/file.png

    :param moving_keypoints:
        The coordinates of the keyponits the vectors start from.
        Acceptable shape: [1, N, 3]
    
    :param id_grid:
        An instance of an identity grid with the same
        shape as the deformation grid.

    :param ax:
        An instance of matplotlib ax.
        Used for overlaying the vector field on exisiting figures.
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

    flow = flow.permute(0, 4, 1, 2, 3)
    kp2_grid = kp2_norm.view(1, -1, 1, 1, 3)

    # Interpolate displacements at kp2 locations
    # final shape: [1, N, 3]
    displacement_field = torch.nn.functional.grid_sample(flow,
                                                         kp2_grid,
                                                         align_corners=True)
    displacement_field = -displacement_field.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    if ax is None:
        fig, ax = plt.subplots()
        
        ax.quiver(kp2_norm[0, :, 0].cpu(),
                  kp2_norm[0, :, 2].cpu(),
                  displacement_field[0, :, 0].cpu(),
                  displacement_field[0, :, 2].cpu(),
                  color='#208080')
        
        fig.savefig(path)
        plt.close()
    else:
        ax.quiver(kp2_norm[0, :, 0].cpu(),
                  kp2_norm[0, :, 2].cpu(),
                  displacement_field[0, :, 0].cpu(),
                  displacement_field[0, :, 2].cpu())
    
def save_grid_overlayed_image(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    path: str,
    ax: plt.Axes=None,
    grid_color: str='#F80101',
    grid_alpha: float=0.6,
    grid_downfactor: int=2,
    cmap: str='bone'
) -> None:
    """Saves a 2D/3D warped image with the deformation grid overlaid on it.
    For 3D images it saves the middle slice along the y-axis.

    :param image:
        The warped image.
        Acceptable shapes: [1, 1, D, H, W] or [1, 1, H, W].

    :param deformation_grid:
        The [-1, 1] normalized grid used for warping the image.
        Shape: [1, D, H, W, 3] or [1, H, W, 2].

    :param path:
        complete/path/to/file.png
    
    :param grid_color:
        The color of the deformation grid.

    :param grid_alpha:
        The opacity of the deformation grid.

    :param grid_downfactor:
        The down sampling rate for the deformation grid
        for visualization purposes.
    
    :param cmap:
        The color map for displaying the image.
    """
    # denormalize the grid
    denormalized_grid = grid_denormalizer(deformation_grid)
    if ax is None:
        fig, ax = plt.subplots()

        save_image(image, None, ax, cmap=cmap)
        save_grid(denormalized_grid, None, ax,
                  color=grid_color,
                  alpha=grid_alpha,
                  down_factor=grid_downfactor)
        fig.savefig(path)
        plt.close()
    else:
        save_image(image, None, ax, cmap=cmap)
        save_grid(denormalized_grid, None, ax,
                  color=grid_color,
                  alpha=grid_alpha,
                  down_factor=grid_downfactor)
        ax.imshow(image, cmap=cmap)

def animate_warping(
    network: torch.nn.Module,
    fixed_img: torch.Tensor,
    moving_img: torch.Tensor, 
    id_grid: torch.Tensor,
    down_factor: int,
    save_path: str,
    num_frames: int=100,
    overlay_grid: bool=False,
    cmap: str='bone'
) -> None:
    """Animates the warping trajectory of SGDIR.

    :param network:
        An instance of SGDIR network.

    :param fixed_img:
        The fixed image.

    :param moving_img:
        The moving image.

    :param id_grid:
        An instance of the identity grid used in the input of
        SGDIR.

    :param down_factor:
        The down sampling rate of deformation grid used for
        visualization purposes.
    
    :param overlay_grid:
        If True, the animation includes the warping grid
        overlaid on the image.

    :param camp:
        The color map used for displaying the image.
    """
    tmp_dir = os.path.join(save_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    
    timesteps = torch.linspace(0, 1, num_frames).to(fixed_img.device)

    for index, t in enumerate(timesteps):
        forward_deformation = network(fixed_img, moving_img, id_grid, t.view(1))
        backward_deformation = network(fixed_img, moving_img, id_grid, -t.view(1))

        moving_warped = warp(moving_img, forward_deformation)
        fixed_warped = warp(fixed_img, backward_deformation)

        if overlay_grid:
            save_grid_overlayed_image(moving_warped,
                                      backward_deformation,
                                      os.path.join(tmp_dir, f'forward_frame_{index}.png'),
                                      grid_downfactor=down_factor,
                                      cmap=cmap)
            save_grid_overlayed_image(fixed_warped,
                                      forward_deformation,
                                      os.path.join(tmp_dir, f'backward_frame_{index}.png'),
                                      grid_downfactor=down_factor,
                                      cmap=cmap)
        else:
            save_image(moving_warped,
                       os.path.join(tmp_dir, f'forward_frame_{index}.png'),
                       cmap=cmap)
            save_image(fixed_warped,
                       os.path.join(tmp_dir, f'backward_frame_{index}.png'),
                       cmap=cmap)
    
    forward_frames = []
    backward_frames = []
    for index in range(num_frames):
        f_img = io.imread(os.path.join(tmp_dir, f'forward_frame_{index}.png')).astype(np.float32) / 255.
        b_img = io.imread(os.path.join(tmp_dir, f'backward_frame_{index}.png')).astype(np.float32) / 255.

        forward_frames.append(f_img)
        backward_frames.append(b_img)
    

    postfix = '_with_grid' if overlay_grid else ''

    forward_fig, forward_ax = plt.subplots()
    forward_ax.axis('off')

    first_forward_frame = forward_ax.imshow(forward_frames[0], cmap=cmap, animated=True)
    
    def forward_update(i):
        first_forward_frame.set_array(forward_frames[i])
        
        return first_forward_frame

    forward_animation = animation.FuncAnimation(forward_fig, forward_update, frames=num_frames, repeat_delay=10)
    forward_animation.save(os.path.join(save_path, f'moving_to_fixed{postfix}.gif'))


    backward_fig, backward_ax = plt.subplots()
    backward_ax.axis('off')

    first_backward_frame = backward_ax.imshow(backward_frames[0], cmap=cmap, animated=True)

    def backward_update(i):
        first_backward_frame.set_array(backward_frames[i])
        
        return first_backward_frame
    
    backward_animation = animation.FuncAnimation(backward_fig, backward_update, frames=num_frames, repeat_delay=10)
    backward_animation.save(os.path.join(save_path, f'fixed_to_moving{postfix}.gif'))

    plt.close()
    shutil.rmtree(tmp_dir)

def flow_snapshots(
    network: torch.nn.Module,
    fixed_img: torch.Tensor,
    moving_img: torch.Tensor,
    id_grid: torch.Tensor,
    save_path: str,
    num_frames: int=7
) -> None:
    """Saves the snapshots of warped images by SGDIR at
    different time points in [0, 1].

    :param network:
        An instance of SGDIR.

    :param fixed_img:
        The fixed image.

    :param moving_img:
        The moving image.

    :param id_grid:
        An instance of the identity grid used in the input of
        SGDIR.

    :param save_path:
        /path/to/directory/

    :param num_frames:
        The number of frames to save. The [0, 1] time interval
        is divided into num_frames segments and warpings will be
        saved at each segment.
    """
    base_path = os.path.join(save_path, 'frames')
    os.makedirs(f'{save_path}/frames', exist_ok=True)
    
    timesteps = torch.linspace(0, 1, num_frames).to(fixed_img.device)

    for index, t in enumerate(timesteps):
        deformation = network(fixed_img, moving_img, id_grid, t.view(1))

        moving_warped = warp(moving_img, deformation)

        save_image(moving_warped,
                   os.path.join(base_path, f'warped_{index}.png'),
                   cmap='bone')
