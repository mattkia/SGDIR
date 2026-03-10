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


def save_grid(deformation_grid: torch.Tensor,
              path: str,
              ax: plt.Axes=None,
              color: str='#404040',
              alpha: float=1.,
              down_factor: int=1) -> None:
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

def save_image(image: torch.Tensor,
               path: str,
               ax: plt.Axes=None,
               cmap: str='gray') -> None:
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
    
def save_mask(image: torch.Tensor,
              path: str,
              ax: plt.Axes=None,
              alpha: float=1.,
              num_classes: int=35) -> None:

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

def save_vector_field(deformation_grid: torch.Tensor,
                      path: str,
                      moving_keypoints: torch.Tensor,
                      id_grid: torch.Tensor,
                      ax: plt.Axes=None) -> None:
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
    
def save_grid_overlayed_image(image: torch.Tensor,
                              deformation_grid: torch.Tensor,
                              path: str,
                              ax: plt.Axes=None,
                              grid_color: str='#F80101',
                              grid_alpha: float=0.6,
                              grid_downfactor: int=2,
                              cmap: str='bone') -> None:
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

def animate_warping(network: torch.nn.Module,
                    fixed_img: torch.Tensor,
                    moving_img: torch.Tensor, 
                    id_grid: torch.Tensor,
                    down_factor: int,
                    save_path: str,
                    num_frames: int=100,
                    overlay_grid: bool=False,
                    cmap: str='bone') -> None:


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

def flow_snapshots(network: torch.nn.Module,
                   fixed_img: torch.Tensor,
                   moving_img: torch.Tensor,
                   id_grid: torch.Tensor,
                   save_path: str,
                   num_frames: int=7) -> None:
    base_path = os.path.join(save_path, 'frames')
    os.makedirs(f'{save_path}/frames', exist_ok=True)
    
    timesteps = torch.linspace(0, 1, num_frames).to(fixed_img.device)

    for index, t in enumerate(timesteps):
        deformation = network(fixed_img, moving_img, id_grid, t.view(1))

        moving_warped = warp(moving_img, deformation)

        save_image(moving_warped,
                   os.path.join(base_path, f'warped_{index}.png'),
                   cmap='bone')
