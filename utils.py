import os
import shutil
import torch
import scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F
import torch.nn as nn

from skimage import io
from typing import Tuple
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection
from torchvision.transforms.functional import rgb_to_grayscale

from config import Config
from data import OASISRegistrationV2
from data import CANDIRegistrationV2
from data import LPBA40Registration
from data import IXIRegistration



def jacobian_determinant_3d(deformed_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of the Jacobian numerically, given the deformed
    output grid and returns the percentage of negative values
    
    Args:
        deformed_grid (torch.Tensor): [B, D, H, W, 3]

    Returns:
        torch.Tensor: the percentage of negative determinants
    """
    dy = deformed_grid[:, 1:, :-1, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dx = deformed_grid[:, :-1, 1:, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dz = deformed_grid[:, :-1, :-1, 1:, :] - deformed_grid[:, :-1, :-1, :-1, :]

    det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy [:, :, :,:, 2] * dz[:, :, :, :, 0])
    det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy [:, :, :,:, 1] * dz[:, :, :, :, 0])

    determinants = det0 - det1 + det2
    
    num_neg_dets = len(determinants[determinants <= 0])
    total_points = torch.prod(torch.tensor(determinants.size(), device=determinants.device))
    
    neg_dets_percentage = num_neg_dets * 100 / total_points
    
    return neg_dets_percentage


def normalized_cross_correlation(input_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    """
    Computes the global Normalized Cross-Correlation (NCC) coefficient between a batch of input images and
    a batch of output images
    
    Args:
        input_image (torch.Tensor): [B, 1, D, H, W]
        target_image (torch.Tensor): [B, 1, D, H, W]

    Returns:
        torch.Tensor: the NCC between the input and target images
    """
    
    input_batch_mean = torch.mean(input_image.flatten(start_dim=1), dim=-1).view(-1, 1, 1, 1)
    target_batch_mean = torch.mean(target_image.flatten(start_dim=1), dim=-1).view(-1, 1, 1, 1)
    
    
    numerator = (input_image - input_batch_mean) * (target_image - target_batch_mean)
    numerator = torch.sum(numerator.flatten(start_dim=1), dim=-1)
    
    denum1 = torch.sum(((input_image - input_batch_mean) ** 2).flatten(start_dim=1), dim=-1)
    denum2 = torch.sum(((target_image - target_batch_mean) ** 2).flatten(start_dim=1), dim=-1)
    denum = torch.sqrt(denum1 * denum2)
    
    ncc = torch.mean(numerator / denum)
    
    return ncc


def save_grid(xy: torch.Tensor, xyr: torch.Tensor, save_path: str, num_rows: int=4, prefix: str=None, **kwargs) -> None:
    """
    Plots the grid given by x and y
    
    Args:
        xy (torch.Tensor): generated grids [B, h, w, 2]
        xyr (torch.Tensor): generated reverse grids [B, h, w, 2]
        save_path (str): the directory where the result is to be saved
        num_rows (int) - Defaults to 4: the maximum number of rows in the saved image
        prefix (str) - Defaults to None: the optional prefix to the file name to be saved
    """
    batch_size = xy.size(0)
    
    if batch_size % num_rows == 0:
        num_pages = batch_size // num_rows
    else:
        num_pages = batch_size // num_rows + 1
    
    for page in range(num_pages):
        if num_pages == 1:
            width = batch_size
        else:
            if page == num_pages - 1:
                width = batch_size - page * num_rows
            else:
                width = num_rows
                
        fig, ax = plt.subplots(nrows=width, ncols=2, figsize=(16, width * 4))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        fontsize = 35
        
        for row in range(width):
            if width > 1:
                if row == 0:
                    ax[row, 0].set_title('Forward', fontsize=fontsize)
                    ax[row, 1].set_title('Reverse', fontsize=fontsize)
                    
                segs1 = xy[row].cpu().numpy()
                segs2 = segs1.transpose(1, 0, 2)
                ax[row, 0].add_collection(LineCollection(segs1, **kwargs))
                ax[row, 0].add_collection(LineCollection(segs2, **kwargs))
                ax[row, 0].autoscale()
                
                segs1 = xyr[row].cpu().numpy()
                segs2 = segs1.transpose(1, 0, 2)
                ax[row, 1].add_collection(LineCollection(segs1, **kwargs))
                ax[row, 1].add_collection(LineCollection(segs2, **kwargs))
                ax[row, 1].autoscale()
            else:
                if row == 0:
                    ax[0].set_title('Forward', fontsize=fontsize)
                    ax[1].set_title('Reverse', fontsize=fontsize)
                    
                segs1 = xy[row].cpu().numpy()
                segs2 = segs1.transpose(1, 0, 2)
                ax[0].add_collection(LineCollection(segs1, **kwargs))
                ax[0].add_collection(LineCollection(segs2, **kwargs))
                ax[0].autoscale()
                
                segs1 = xyr[row].cpu().numpy()
                segs2 = segs1.transpose(1, 0, 2)
                ax[1].add_collection(LineCollection(segs1, **kwargs))
                ax[1].add_collection(LineCollection(segs2, **kwargs))
                ax[1].autoscale()
        
        
        filename = f'page_{page + 1}_grid'
        if prefix is not None:
            filename = 'batch_' + prefix + '_' + filename
        fig.savefig(f'{save_path}/{filename}.png')
        plt.close()
    

def save_images(fixed_img: torch.Tensor, moving_img: torch.Tensor, 
                warped_fixed: torch.Tensor, warped_moving: torch.Tensor, save_path: str, 
                show_diff: bool=True, num_rows: int=8, prefix: str=None):
    """
    This function is used to save the results of the FlowNet3D in batch form.
    
    Args:
        fixed_img (torch.Tensor): batch of fixed images [B, 1, D, H, W]
        moving_img (torch.Tensor): batch of moving images [B, 1, D, H, W]
        warped_fixed (torch.Tensor): batch of warped fixed images [B, 1, D, H, W]
        warped_moving (torch.Tensor): batch of warped moving images [B, 1, D, H, W]
        save_path (str): the directory where the results are to be saved
        show_diff (bool) - Defaults to True: if True, the absolute differences of warped and target are saved too
        num_rows (int) - Defaults to 8: The maximum number of rows of images in a single plot.
        prefix (str): the optional prefix to the file name to be saved
    """
    batch_size = fixed_img.size(0)
    
    if batch_size % num_rows == 0:
        num_pages = batch_size // num_rows
    else:
        num_pages = batch_size // num_rows + 1
    
    for page in range(num_pages):
        if num_pages == 1:
            width = batch_size
        else:
            if page == num_pages - 1:
                width = batch_size - page * num_rows
            else:
                width = num_rows
                
        fig, ax = plt.subplots(nrows=width, ncols=4, figsize=(16, width * 4))
        plt.subplots_adjust(wspace=0.5 if show_diff else 0.1, hspace=0.1)
        fontsize = 30 if show_diff else 35
        
        for row in range(width):
            I_ = fixed_img[row].permute(1, 2, 0).cpu().numpy()
            J_ = moving_img[row].permute(1, 2, 0).cpu().numpy()
            Iw = warped_fixed[row].permute(1, 2, 0).cpu().numpy()
            Jw = warped_moving[row].permute(1, 2, 0).cpu().numpy()
            
            if show_diff:
                first = J_ - I_ , 'J-I Before Reg'
                second = Jw - I_, 'Jw-I After Reg'
                third = I_ - J_, 'I-J Before Reg'
                fourth = Iw - J_, 'Iw-J After Reg'
            else:
                first = I_, 'I'
                second = Jw, 'Jw'
                third = J_, 'J'
                fourth = Iw, 'Iw'
            
            if width > 1:
                if row == 0:
                    ax[row, 0].set_title(first[1], fontsize=fontsize)
                    ax[row, 1].set_title(second[1], fontsize=fontsize)
                    ax[row, 2].set_title(third[1], fontsize=fontsize)
                    ax[row, 3].set_title(fourth[1], fontsize=fontsize)
                    
                ax[row, 0].set_xticks([])
                ax[row, 0].set_yticks([])
                ax[row, 0].imshow(first[0], cmap='gray')

                ax[row, 1].set_xticks([])
                ax[row, 1].set_yticks([])
                ax[row, 1].imshow(second[0], cmap='gray') 

                ax[row, 2].set_xticks([])
                ax[row, 2].set_yticks([])
                ax[row, 2].imshow(third[0], cmap='gray')

                ax[row, 3].set_xticks([])
                ax[row, 3].set_yticks([])
                ax[row, 3].imshow(fourth[0], cmap='gray')
            else:
                if row == 0:
                    ax[0].set_title(first[1], fontsize=fontsize)
                    ax[1].set_title(second[1], fontsize=fontsize)
                    ax[2].set_title(third[1], fontsize=fontsize)
                    ax[3].set_title(fourth[1], fontsize=fontsize)
                    
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[0].imshow(first[0], cmap='gray')

                ax[1].set_xticks([])
                ax[1].set_yticks([])
                ax[1].imshow(second[0], cmap='gray') 

                ax[2].set_xticks([])
                ax[2].set_yticks([])
                ax[2].imshow(third[0], cmap='gray')

                ax[3].set_xticks([])
                ax[3].set_yticks([])
                ax[3].imshow(fourth[0], cmap='gray')
        
        
        filename = f'page_{page + 1}_diff' if show_diff else f'page_{page + 1}'
        if prefix is not None:
            filename = 'batch_' + prefix + '_' + filename
        fig.savefig(f'{save_path}/{filename}.png')
        plt.close()


def save_warped(warped_img: torch.Tensor, warped_seg: torch.Tensor, 
                save_path: str, prefix: str=None):
    """
    This function receives an image and its segmentation mask and plots them in
    an overlayed fashion.
    
    Args:
        warped_img (torch.Tensor): the warped image with shape [1, D, H, W]
        warped_seg (torch.Tensor): the segmentation mask with shape [1, D, H, W]
        save_path (str): the directory in which the results is to be saved
        prefix (str): an optional prefix to the file name
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fontsize = 35
    
    I_ = warped_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    J_ = warped_seg.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    ax.set_title('warped and segmentation', fontsize=fontsize)
            
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(I_, cmap='gray')
    ax.imshow(J_, cmap='twilight', alpha=0.5)
    
    
    if prefix is not None:
        filename = 'batch_' + prefix
    fig.savefig(f'{save_path}/{filename}_warped_seg.png')
    plt.close()


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


def get_scheduler(optimizer: torch.optim, scheduler_type: str, **kwargs) -> torch.optim.lr_scheduler:
    """
    Receives an optimizer and an scheduler type along with its configuration
    and returns a schedure for the optimizer
    
    Args:
        optimizer (torch.optim): optimizer (e.g., Adam , SGD)
        scheduler_type (str): e.g., 'exponential', 'step', ...
    """
    
    if scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif scheduler_type == 'plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise Exception('[!] Specified scheduler not found!')
    
    return scheduler


def get_dataset(config: Config, train: bool=True, **kwargs):
    """
    This function receives the configuration dictionary and returns the dataset
    specified in the configs along with the determined specifications.
    
    Args:
        config (Config): an instance of the configs
        train (bool) - Defaults to True: dataset mode (train/val/test)
        
    Returns:
        Dataset: torch friendly dataset of the specified data.
    """
    name = getattr(config, 'name')
    if name == 'oasis':
        data_path = getattr(config, 'path')
        crop_x = getattr(config, 'crop_x')
        crop_x = int(crop_x) if crop_x is not None else None
        crop_y = getattr(config, 'crop_y')
        crop_y = int(crop_y) if crop_y is not None else None
        crop_z = getattr(config, 'crop_z')
        crop_z = int(crop_z) if crop_z is not None else None
        
        if 'val' in kwargs.keys():
            if kwargs['val']:
                mode = 'val'
        else:
            if train:
                mode = 'train'
            else:
                mode = 'test'
        dataset = OASISRegistrationV2(dataset_path=data_path, 
                                      crop_x=crop_x, crop_y=crop_y, crop_z=crop_z, 
                                      mode=mode)
            
        return dataset
    elif name == 'candi':
        data_path = getattr(config, 'path')
        crop_x = getattr(config, 'crop_x')
        crop_x = int(crop_x) if crop_x is not None else None
        crop_y = getattr(config, 'crop_y')
        crop_y = int(crop_y) if crop_y is not None else None
        crop_z = getattr(config, 'crop_z')
        crop_z = int(crop_z) if crop_z is not None else None
        
        if 'val' in kwargs.keys():
            if kwargs['val']:
                mode = 'val'
        else:
            if train:
                mode = 'train'
            else:
                mode = 'test'
        dataset = CANDIRegistrationV2(dataset_path=data_path, 
                                      crop_x=crop_x, crop_y=crop_y, crop_z=crop_z, 
                                      mode=mode)
        
        return dataset
    elif name == 'lpba40':
        data_path = getattr(config, 'path')
        crop_x = getattr(config, 'crop_x')
        crop_x = int(crop_x) if crop_x is not None else None
        crop_y = getattr(config, 'crop_y')
        crop_y = int(crop_y) if crop_y is not None else None
        crop_z = getattr(config, 'crop_z')
        crop_z = int(crop_z) if crop_z is not None else None
        
        if 'val' in kwargs.keys():
            if kwargs['val']:
                mode = 'val'
        else:
            if train:
                mode = 'train'
            else:
                mode = 'test'
        dataset = LPBA40Registration(dataset_path=data_path, 
                                     crop_x=crop_x, crop_y=crop_y, crop_z=crop_z, 
                                     mode=mode)
            
        return dataset
    elif name == 'ixi':
        data_path = getattr(config, 'path')
        
        crop_x = getattr(config, 'crop_x')
        crop_x = int(crop_x) if crop_x is not None else None
        crop_y = getattr(config, 'crop_y')
        crop_y = int(crop_y) if crop_y is not None else None
        crop_z = getattr(config, 'crop_z')
        crop_z = int(crop_z) if crop_z is not None else None
        
        if 'val' in kwargs.keys():
            if kwargs['val']:
                mode = 'val'
        else:
            if train:
                mode = 'train'
            else:
                mode = 'test'
        
        dataset = IXIRegistration(dataset_path=data_path,
                                  crop_x=crop_x, crop_y=crop_y, crop_z=crop_z,
                                  mode=mode)
        
        return dataset
    else:
        raise Exception(f'[!] Dataset {name} does not exist in the set of experiments. '
                        f'You need to implement your own data loader')


def animate_flow_3d(network: torch.nn.Module, fixed_img: torch.Tensor, moving_img: torch.Tensor, 
                    grid: torch.Tensor, down_factor: int, save_path: str, num_frames: int=100, prefix: str=None) -> None:
    """
    Receives the FlowNet3D, fixed image, moving image, and a number of steps, and animates the
    deformation of the moving image towards the fixed image. Saves the gifs after the 
    animations are created.
    
    Args:
        network (torch.nn.Module): an instanc of the PhiNet
        fixed_img (torch.Tensor): fixed image [1, 1, D, H, W]
        moving_img (torch.Tensor): moving image [1, 1, D, H, W]
        grid (torch.Tensor): the initial grid [1, D, H, W, 3]
        save_path (str): the directory in which the gifs will be saved
        num_frames (int) - Defaluts to 100: the number of frames to be generated for the gifs
        prefix (str) - Defaults to None: the prefix added to saved filename
    """
    # create a temporrary folder which will be removed after the gif is generated
    os.makedirs('evals/tmp', exist_ok=True)
    
    timesteps = torch.linspace(0, 1, num_frames)
    mid_frame_index = fixed_img.size(3) // 2
    
    for index, t in enumerate(timesteps):
        xyz = network(fixed_img, moving_img, grid, t.view(1).to(fixed_img.device))
        xyzr = network(fixed_img, moving_img, grid, -t.view(1).to(fixed_img.device))
        
        Jw = F.grid_sample(moving_img, xyz, padding_mode='reflection', align_corners=True)
        Iw = F.grid_sample(fixed_img, xyzr, padding_mode='reflection', align_corners=True)
        
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), tight_layout=True)
        ax[0].set_title('Fixed Image')
        ax[0].imshow(fixed_img[:, :, :, mid_frame_index, :].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[0].axis('off')
        ax[1].set_title('Moving Image')
        ax[1].imshow(Jw[:, :, :, mid_frame_index, :].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[1].axis('off')
        xy = torch.cat([xyz[:, :, mid_frame_index, :, 0].unsqueeze(-1), xyz[:, :, mid_frame_index, :, 2].unsqueeze(-1)], dim=-1)
        xy = xy[:, ::down_factor, ::down_factor, :]
        segs1 = xy[0].cpu().numpy()
        segs2 = segs1.transpose(1, 0, 2)
        ax[2].add_collection(LineCollection(segs1))
        ax[2].add_collection(LineCollection(segs2))
        ax[2].autoscale()
        ax[2].set_title('Deformation Grid')
        ax[2].axis('off')
        fig.savefig(f'evals/tmp/forward_{index}.png')
        plt.close()
        
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), tight_layout=True)
        ax[0].set_title('Fixed Image')
        ax[0].imshow(moving_img[:, :, :, mid_frame_index, :].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[0].axis('off')
        ax[1].set_title('Moving Image')
        ax[1].imshow(Iw[:, :, :, mid_frame_index, :].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[1].axis('off')
        xyr = torch.cat([xyzr[:, :, mid_frame_index, :, 0].unsqueeze(-1), xyzr[:, :, mid_frame_index, :, 2].unsqueeze(-1)], dim=-1)
        xyr = xyr[:, ::down_factor, ::down_factor, :]
        segs1 = xyr[0].cpu().numpy()
        segs2 = segs1.transpose(1, 0, 2)
        ax[2].add_collection(LineCollection(segs1))
        ax[2].add_collection(LineCollection(segs2))
        ax[2].autoscale()
        ax[2].set_title('Deformation Grid')
        ax[2].axis('off')
        fig.savefig(f'evals/tmp/reverse_{index}.png')
        plt.close()
    
    f_images = []
    r_images = []
    for index in range(num_frames):
        f_img = io.imread(f'evals/tmp/forward_{index}.png').astype(np.float32) / 255.
        r_img = io.imread(f'evals/tmp/reverse_{index}.png').astype(np.float32) / 255.
        f_images.append(f_img)
        r_images.append(r_img)
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    im = ax.imshow(f_images[0], cmap='gray', animated=True)
    
    def update(i):
        im.set_array(f_images[i])
        
        return im
    
    animation_fig = animation.FuncAnimation(fig, update, frames=num_frames, repeat_delay=10)
    filename = 'forward' if prefix is None else f'{prefix}_forward'
    animation_fig.save(f'{save_path}/{filename}.gif')
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    im = ax.imshow(r_images[0], cmap='gray', animated=True)
    
    def update(i):
        im.set_array(r_images[i])
        
        return im
    
    animation_fig = animation.FuncAnimation(fig, update, frames=num_frames, repeat_delay=10)
    filename = 'reverse' if prefix is None else f'{prefix}_reverse'
    animation_fig.save(f'{save_path}/{filename}.gif')
    plt.close()
    shutil.rmtree('evals/tmp')


def save_flow_sequence(network: torch.nn.Module, fixed_img: torch.Tensor, moving_img: torch.Tensor, 
                    init_grid: torch.Tensor, save_path: str, num_frames: int=7):
    """
    This function stores the temporal warping output by the network in a sequential manner.
    
    Args:
        network (torch.nn.Module): an instance of FlowNet3D by which the inference is done
        fixed_img (torch.Tensor): the fixed image with shape [1, 1, D, H, W]
        moving_img (torch.Tensor): the moving image with shape [1, 1, D, H, W]
        init_grid (torch.Tensor): the initial xyz grid with shape [1, D, H, W, 3]
        save_path (str): the directory where the results are saved
        num_frames (int) - Defaults to 7: the number of frames from t=0 to t=1 at which the warpings are calculated
    """
    # create a temporrary folder which will be removed after the gif is generated
    os.makedirs(f'{save_path}/seq_flow', exist_ok=True)
    
    timesteps = torch.linspace(0, 1, num_frames)
    mid_frame_index = fixed_img.size(3) // 2
    
    for index, t in enumerate(timesteps):
        xyz = network(fixed_img, moving_img, init_grid, t.view(1).to(fixed_img.device))
        
        Jw = F.grid_sample(moving_img, xyz, padding_mode='reflection', align_corners=True)
        
        fig, ax = plt.subplots(1, 1)
        ax.imshow(Jw[:, :, :, mid_frame_index, :].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        fig.savefig(f'{save_path}/seq_flow/forward_{index}.png')
        plt.close()


def dice(seg1: torch.Tensor, seg2: torch.Tensor, grid: torch.Tensor, grid_r: torch.Tensor, bg: bool=False, structured: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function receives the segmentation maps of the fixed and moving images along with the forward and reversed
    grids and warps the segmentation maps and computes the DICE score
    
    Args:
        seg1 (torch.Tensor): segmentation map of the fixed image [B, 1, D, H, W] or [B, 1, H, W]
        seg2 (torch.Tensor): segmentation map of the moving image [B, 1, D, H, W] or [B, 1, H, W]
        grid (torch.Tensor): the grid on which the moving image is sampled [B, D, H, W, 3] or [B, H, W, 2]
        gridr (torch.Tensor): the grid on which the fixed image is sampled [B, D, H, W, 3] or [B, H, W, 2]
        bg (bool) - Defaults to False: if False, background is excluded form the computation of the DICE score
        structured (bool) - Defaults to False: if True, the dice over each structure is calculated, otherwise the
                                               volumetric dice is computed
    
    Returns:
        Tuple[torch.float32, torch.float32]: the dice score between warped seg1 and seg2 and the dice score
                                             between the warped seg2 and seg1
    """
    warped_seg2 = F.grid_sample(seg2, grid, mode='nearest', padding_mode='reflection', align_corners=True)
    warped_seg1 = F.grid_sample(seg1, grid_r, mode='nearest', padding_mode='reflection', align_corners=True)
    
    if structured:
        dsc1 = dsc_score(seg1, warped_seg2, bg=bg, structured=True)
        dsc2 = dsc_score(seg2, warped_seg1, bg=bg, structured=True)
    else:
        dsc1 = dsc_score(seg1, warped_seg2, bg=bg, structured=False)
        dsc2 = dsc_score(seg2, warped_seg1, bg=bg, structured=False)
    
    return dsc1, dsc2


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


def compute_hd95(seg1: torch.Tensor, seg2: torch.Tensor, spacing=(1.0, 1.0, 1.0)):
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
    
    for t in timesteps:
        xyz = network(fixed_img, moving_img, grid, t.view(1).to(fixed_seg.device))
        xyzr = network(fixed_img, moving_img, grid, (t-1).view(1).to(fixed_seg.device))
        
        seg_Jw = F.grid_sample(moving_seg, xyz, mode='nearest', align_corners=True, padding_mode='reflection')
        seg_Iw = F.grid_sample(fixed_seg, xyzr, mode='nearest', align_corners=True, padding_mode='reflection')
        f_dice = dsc_score(seg_Jw, seg_Iw, structured=False)
    
        f_dice_scores.append(f_dice.item())
    
    np.savetxt(f'{save_path}/fwd_rolling_dice.txt', f_dice_scores)
        

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
            jacs[i] = jacobian_determinant_3d(grid_t)
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


class SSIM3d(nn.Module):
    """
    This class implements the Structural Similarity Index Measure (SSIM) for 3d images.
    The codes are mainly refactored from
    
    https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/
    """
    def __init__(self, window_size: int=11, size_average: bool=True) -> None:
        super().__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.kernel = self.__create_3d_kernel(window_size, self.channels)
    
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        n_channels = img1.size(1)
        
        if n_channels == self.channels and self.kernel.data.type == img1.data.type:
            window = self.kernel
        else:
            window = self.__create_3d_kernel(self.window_size, n_channels)
            
        window = window.to(img1.device)
        window = window.type_as(img1)
        
        ssim = self.__ssim(img1, img2, window, self.window_size, n_channels, self.size_average)
        
        return ssim
       
    def __create_3d_kernel(self, window_size: int, channels: int) -> torch.Tensor:
        kernel1d = self.__gaussian(window_size, 1.5).unsqueeze(1)
        kernel2d = torch.mm(kernel1d, kernel1d.t()).unsqueeze(0).unsqueeze(0)
        kernel3d = torch.mm(kernel1d, kernel2d.reshape(1, -1))
        kernel3d = kernel3d.reshape(window_size, window_size, window_size).unsqueeze(0).unsqueeze(0)
        
        kernel = kernel3d.expand(channels, 1, window_size, window_size, window_size).contiguous()
        
        return kernel    
    
    def __gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        kernel = torch.Tensor([torch.exp(-(x - torch.tensor(window_size // 2))**2 / (2 * torch.tensor(sigma ** 2))) for x in range(window_size)])
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def __ssim(self, img1: torch.Tensor, img2: torch.Tensor, kernel: torch.Tensor, 
               kernel_size: int, channels: int, size_average: bool) -> torch.Tensor:
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

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        assert len(I.shape) == 4 or len(I.shape) == 5, '[!] Expected shape of [B, C, H. W] or [B, C, D, H, W]'
        assert len(J.shape) == 4 or len(J.shape) == 5, '[!] Expected shape of [B, C, H. W] or [B, C, D, H, W]'
        
        # converto to grayscale if the images are colored
        if I.size(1) > 1:
            I = rgb_to_grayscale(I)
        if J.size(1) > 1:
            J = rgb_to_grayscale(J)
        
        ndims = len(I.shape) - 2
        
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        
        if ndims == 3:
            weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        else:
            weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
            
        conv_fn = F.conv3d if ndims == 3 else F.conv2d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        loss = -1.0 * torch.mean(cc)
        
        return loss

