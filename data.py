import os
import json
import torch
import random

import numpy as np
import nibabel as nib

from itertools import product
from typing import List, Tuple
from torch.utils.data import Dataset


class OASISRegistrationV2(Dataset):
    def __init__(self, dataset_path: str, 
                 crop_x: int=None, crop_y: int=None, crop_z: int=None, 
                 normalize: bool=True, mode: str='train') -> None:
        
        super().__init__()
        
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.mode = mode
        
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z
        
        available_subjects = os.listdir(dataset_path)
        available_subjects = list(filter(lambda x: x.startswith('OASIS'), available_subjects))
        subject_ids = list(map(lambda x: x.split('_')[-2].strip(), available_subjects))
        
        if os.path.exists('tmp/oasis_train_val_test.json'):
            # loading the existing information
            with open('tmp/oasis_train_val_test.json', 'r') as handle:
                if mode == 'train':
                    self.fixed_moving_ids = json.load(handle)['train']
                elif mode == 'val':
                    self.fixed_moving_ids = json.load(handle)['val']
                else:
                    self.fixed_moving_ids = json.load(handle)['test']
                    
        else:
            # find the (fixed image, moving image) training pairs
            train_pairs = self.randomized_pairs(subject_ids[:256], 40, 40)
            # find the (fixed image, moving image) validation pairs
            val_pairs = self.randomized_pairs(subject_ids[256: 306], 10, 10)
            # find the (fixed image, moving image) test pairs
            test_pairs = self.randomized_pairs(subject_ids[306:], 15, 15)
            # saving the information for future use
            oasis_train_test = {'train': train_pairs,
                                'val': val_pairs,
                                'test': test_pairs}
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/oasis_train_val_test.json', 'w') as handle:
                json.dump(oasis_train_test, handle)
            
            if mode == 'train':
                self.fixed_moving_ids = train_pairs
            elif mode == 'val':
                self.fixed_moving_ids = val_pairs
            else:
                self.fixed_moving_ids = test_pairs
    
    def randomized_pairs(self, ids: List[str], num_moving_imgs: int, num_atlases: int) -> List[Tuple]:
        """
        This method receives the ids of all images, along with the nnumber of images to be considered as the 
        moving image and the number of images to be considered as atlases (or fixed images). The method then
        randomly samples from the ids and pairs up the ids of the moving images and the fixed/atlas images
        
        Args:
            ids (List[str]): a list of strings containing the ids of the images; e.g., 0001, 0034, 0411
            num_moving_imgs (int): the number of images to be considered as the moving image
            num_atlases (int): the number of images to be considered as the atlas or fixed images
            
        Returns:
            List[Tuple]: a list of the form [(f_id1, m_id1), (f_id2, m_id2), ...], where m_id is the id
                         of the moving image and the f_id is the id of the fixed image
        """
        # randomly choose the moving images
        moving_imgs_ids = random.sample(ids, num_moving_imgs)
        remained_ids = [i for i in ids if i not in moving_imgs_ids]
        atlas_ids = random.sample(remained_ids, num_atlases)
        
        fixed_moving_ids = list(product(atlas_ids, moving_imgs_ids))
        
        return fixed_moving_ids

    def load_image_pair(self, fixed_id: str, moving_id: str, crop_x: int, crop_y: int, crop_z: int) -> Tuple[np.ndarray]:
        """
        This method receives the ids of the fixed and moving images along with croping sizes along each axis
        and returns the center-cropped fixed and moving images with the segmentation masks.
        
        Args:
            fixed_id (str): the id of the fixed image (e.g., 0001)
            moving_id (str): the id of the moving image (e.g., 0002)
            crop_x (int): the crop size along the x axis
            crop_y (int): the crop size along the y axis
            crop_z (int): the crop size along the z axis
        Returns
            Tuple[nd.array]: a four-tupple containing the fixed image, moving image, fixed mask, and moving mask
        """
        fixed_path = os.path.join(os.path.join(self.dataset_path, f'OASIS_OAS1_{fixed_id}_MR1'))
        moving_path = os.path.join(os.path.join(self.dataset_path, f'OASIS_OAS1_{moving_id}_MR1'))
        
        # loading the fixed image
        fixed_img = nib.load(os.path.join(fixed_path, 'aligned_norm.nii.gz')).get_fdata()
        fixed_seg35 = nib.load(os.path.join(fixed_path, 'aligned_seg35.nii.gz')).get_fdata()
        
        # loading the moving image
        moving_img = nib.load(os.path.join(moving_path, 'aligned_norm.nii.gz')).get_fdata()
        moving_seg35 = nib.load(os.path.join(moving_path, 'aligned_seg35.nii.gz')).get_fdata()
        
        # croppin the image if specified
        fixed_img, fixed_seg35, moving_img, moving_seg35 = self.__crop([fixed_img, fixed_seg35, moving_img, moving_seg35], crop_x, crop_y, crop_z)
        self.d, self.h, self.w = fixed_img.shape
        
        return fixed_img, moving_img, fixed_seg35, moving_seg35
    
    def get_grid(self) -> torch.Tensor:
        """
        This method constructs a 3D grid
        
        Args:
            jitter (bool, optional): whether to jitter the grid. Defaults to True.

        Returns:
            torch.Tensor: [B, w, h, d, 3] a grid.
        """
        z, y, x = np.meshgrid(np.arange(0, self.d), 
                              np.arange(0, self.h), 
                              np.arange(0, self.w), indexing='ij')
        
        return torch.tensor(np.stack([x, y, z], 3), dtype=torch.float32)
    
    def __crop(self, imgs_list: List[np.ndarray], crop_x: int, crop_y: int, crop_z: int) -> List[np.ndarray]:
        """
        This method receives a list of images (fixed image, fixed segmentation, moving image, moving segmentation) and the
        cropping information and crops the images along each axis (if specified)
        
        Args:
            imgs_list (List[np.ndarray]): a list containing the 3D images as numpy arrays
            crop_x (int): the new size of the image along the x direction
            crop_y (int): the new size of the image along the y direction
            crop_z (int): the new size of the image along the z direction
            
        Returns:
            List[np.ndarray]: a list of the same images as input but cropped
        """
        if crop_x is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_x, 0)
            
        if crop_y is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_y, 1)
            
        if crop_z is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_z, 2)
        
        return imgs_list
    
    def __crop_axis(self, imgs_list: List[np.ndarray], crop_size: int, axis: int=0) -> List[np.ndarray]:
        """
        This method crops the given list of image along the specified axis.
        
        Args:
            imgs_list (List[np.ndarray]): a list of images to be cropped.
            crop_size (int): the new size of the image along the given axis.
            axis (int) - Defaults to 0: the axis along which the cropping should be done (axis must be eiher 0, 1, or 2)
            
        Returns:
            List[np.ndarray]: a list of the same images as the input but cropped along the specified axis
        """
        
        size = imgs_list[0].shape[axis]
        start = size // 2 - crop_size // 2
        start = start if start > 0 else 0
        
        end = start + crop_size
        end = end if end <= size else size
        if axis == 0:
            cropped_list = [img[start: end, :, :] for img in imgs_list]
        elif axis == 1:
            cropped_list = [img[:, start: end, :] for img in imgs_list]
        else:
            cropped_list = [img[:, :, start: end] for img in imgs_list]
        
        return cropped_list
    
    def __image_norm(self, img: np.ndarray) -> torch.Tensor:
        """
        This method implements a simple min-max normalization on the images
        """
        
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def __len__(self):
        return len(self.fixed_moving_ids)
    
    def __getitem__(self, index):
        fixed_id, moving_id = self.fixed_moving_ids[index]
        fixed, moving, fixed_seg, moving_seg = self.load_image_pair(fixed_id, moving_id, self.crop_x, self.crop_y, self.crop_z)
        
        if self.mode == 'train':
            fixed = fixed + np.random.normal(scale=0.01, size=fixed.shape)
            moving = moving + np.random.normal(scale=0.01, size=moving.shape)
        
        if self.normalize:
            fixed = self.__image_norm(fixed)
            moving = self.__image_norm(moving)
        
        fixed = torch.tensor(fixed, dtype=torch.float32).unsqueeze(0)
        fixed_seg = torch.tensor(fixed_seg, dtype=torch.float32).unsqueeze(0)
        moving = torch.tensor(moving, dtype=torch.float32).unsqueeze(0)
        moving_seg = torch.tensor(moving_seg, dtype=torch.float32).unsqueeze(0)
        
        xyz = self.get_grid()
        
        return (fixed, moving, xyz, fixed_seg, moving_seg)


class CANDIRegistrationV2(Dataset):
    def __init__(self, dataset_path: str, 
                 crop_x: int=None, crop_y: int=None, crop_z: int=None, 
                 normalize: bool=True, 
                 mode: str='train') -> None:
        
        super().__init__()
        
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.mode = mode
        
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z
        
        
        if os.path.exists('tmp/candi_train_val_test.json'):
            # loading the existing information
            with open('tmp/candi_train_val_test.json', 'r') as handle:
                self.ids_info = json.load(handle)
                    
        else:
            available_cats = ['BPDwithoutPsy', 'BPDwithPsy', 'HC', 'SS']
            candi_train_test = {}
            all_ids = []
            for category in available_cats:
                path = os.path.join(dataset_path, f'SchizBull_2008/{category}')
                subject_ids = os.listdir(path)
                subject_ids = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), subject_ids))
                all_ids.extend(subject_ids)
            random.shuffle(all_ids)
            train_length = int(0.7 * len(all_ids))
            val_length = int(0.1 * len(all_ids)) 
            
            train_pairs = self.randomized_pairs(all_ids[:train_length], 20, 20)
            val_pairs = self.randomized_pairs(all_ids[train_length: train_length + val_length], 5, 5)
            test_pairs = self.randomized_pairs(all_ids[train_length + val_length:], 5, 5)
            
            candi_train_test = {'train': train_pairs,
                                'val': val_pairs,
                                'test': test_pairs}
            
            # saving the information for future use
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/candi_train_val_test.json', 'w') as handle:
                json.dump(candi_train_test, handle)
            
            self.ids_info = candi_train_test
    
    def randomized_pairs(self, ids: List[str], num_moving_imgs: int, num_atlases: int) -> List[Tuple]:
        """
        This method receives the ids of all images, along with the nnumber of images to be considered as the 
        moving image and the number of images to be considered as atlases (or fixed images). The method then
        randomly samples from the ids and pairs up the ids of the moving images and the fixed/atlas images
        
        Args:
            ids (List[str]): a list of strings containing the ids of the images; e.g., 0001, 0034, 0411
            num_moving_imgs (int): the number of images to be considered as the moving image
            num_atlases (int): the number of images to be considered as the atlas or fixed images
            
        Returns:
            List[Tuple]: a list of the form [(f_id1, m_id1), (f_id2, m_id2), ...], where m_id is the id
                         of the moving image and the f_id is the id of the fixed image
        """
        # randomly choose the moving images
        moving_imgs_ids = random.sample(ids, num_moving_imgs)
        remained_ids = [i for i in ids if i not in moving_imgs_ids]
        atlas_ids = random.sample(remained_ids, num_atlases)
        
        fixed_moving_ids = list(product(atlas_ids, moving_imgs_ids))
        
        return fixed_moving_ids
    
    def load_image_pair(self, fixed_id: str, moving_id: str, crop_x: int, crop_y: int, crop_z: int) -> Tuple[np.ndarray]:
        """
        This method receives the ids of the fixed and moving images along with croping sizes along each axis
        and returns the center-cropped fixed and moving images with the segmentation masks.
        
        Args:
            fixed_id (str): the id of the fixed image (e.g., 0001)
            moving_id (str): the id of the moving image (e.g., 0002)
            crop_x (int): the crop size along the x axis
            crop_y (int): the crop size along the y axis
            crop_z (int): the crop size along the z axis
        Returns
            Tuple[nd.array]: a four-tupple containing the fixed image, moving image, fixed mask, and moving mask
        """
        if 'BPDwoPsy' in fixed_id:
            fixed_cat = 'BPDwithoutPsy'
        elif 'BPDwPsy' in fixed_id:
            fixed_cat = 'BPDwithPsy'
        elif 'HC' in fixed_id:
            fixed_cat = 'HC'
        else:
            fixed_cat = 'SS'
        
        if 'BPDwoPsy' in moving_id:
            moving_cat = 'BPDwithoutPsy'
        elif 'BPDwPsy' in moving_id:
            moving_cat = 'BPDwithPsy'
        elif 'HC' in moving_id:
            moving_cat = 'HC'
        else:
            moving_cat = 'SS'
            
            
        fixed_path = os.path.join(self.dataset_path, f'SchizBull_2008/{fixed_cat}/{fixed_id}/MNI152_2mm_Linear')
        moving_path = os.path.join(self.dataset_path, f'SchizBull_2008/{moving_cat}/{moving_id}/MNI152_2mm_Linear')
        
        # loading the fixed image
        fixed_img = nib.load(os.path.join(fixed_path, f'{fixed_id}_linear_MRI.nii.gz')).get_fdata()
        fixed_seg = nib.load(os.path.join(fixed_path, f'{fixed_id}_linear_SEG.nii.gz')).get_fdata()
        
        # loading the moving image
        moving_img = nib.load(os.path.join(moving_path, f'{moving_id}_linear_MRI.nii.gz')).get_fdata()
        moving_seg = nib.load(os.path.join(moving_path, f'{moving_id}_linear_SEG.nii.gz')).get_fdata()
        
        # croppin the image if specified
        fixed_img = np.pad(fixed_img, ((2, 3), (2, 1), (2, 3)))
        fixed_seg = np.pad(fixed_seg, ((2, 3), (2, 1), (2, 3)))
        moving_img = np.pad(moving_img, ((2, 3), (2, 1), (2, 3)))
        moving_seg = np.pad(moving_seg, ((2, 3), (2, 1), (2, 3)))
        
        fixed_img, fixed_seg, moving_img, moving_seg = self.__crop([fixed_img, fixed_seg, moving_img, moving_seg], crop_x, crop_y, crop_z)
        self.d, self.h, self.w = fixed_img.shape
        
        return fixed_img, moving_img, fixed_seg, moving_seg
    
    def get_grid(self) -> torch.Tensor:
        """
        This method constructs an empty 3D grid

        Returns:
            torch.Tensor: [B, w, h, d, 3] an empty grid.
        """
        z, y, x = np.meshgrid(np.arange(0, self.d), 
                              np.arange(0, self.h), 
                              np.arange(0, self.w), indexing='ij')
        
        
        return torch.tensor(np.stack([x, y, z], 3), dtype=torch.float32)
    
    def __crop(self, imgs_list: List[np.ndarray], crop_x: int, crop_y: int, crop_z: int) -> List[np.ndarray]:
        """
        This method receives a list of images (fixed image, fixed segmentation, moving image, moving segmentation) and the
        cropping information and crops the images along each axis (if specified)
        
        Args:
            imgs_list (List[np.ndarray]): a list containing the 3D images as numpy arrays
            crop_x (int): the new size of the image along the x direction
            crop_y (int): the new size of the image along the y direction
            crop_z (int): the new size of the image along the z direction
            
        Returns:
            List[np.ndarray]: a list of the same images as input but cropped
        """
        if crop_x is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_x, 0)
            
        if crop_y is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_y, 1)
            
        if crop_z is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_z, 2)
        
        return imgs_list
    
    def __crop_axis(self, imgs_list: List[np.ndarray], crop_size: int, axis: int=0) -> List[np.ndarray]:
        """
        This method crops the given list of image along the specified axis.
        
        Args:
            imgs_list (List[np.ndarray]): a list of images to be cropped.
            crop_size (int): the new size of the image along the given axis.
            axis (int) - Defaults to 0: the axis along which the cropping should be done (axis must be eiher 0, 1, or 2)
            
        Returns:
            List[np.ndarray]: a list of the same images as the input but cropped along the specified axis
        """
        
        size = imgs_list[0].shape[axis]
        start = size // 2 - crop_size // 2
        start = start if start > 0 else 0
        
        end = start + crop_size
        end = end if end <= size else size
        
        if axis == 0:
            cropped_list = [img[start: end, :, :] for img in imgs_list]
        elif axis == 1:
            cropped_list = [img[:, start: end, :] for img in imgs_list]
        else:
            cropped_list = [img[:, :, start: end] for img in imgs_list]
        
        return cropped_list
    
    def __image_norm(self, img: np.ndarray) -> torch.Tensor:
        """
        This method implements a simple min-max normalization on the images
        """
        
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def __len__(self):
        return len(self.ids_info[self.mode])
    
    def __getitem__(self, index):
        fixed_id, moving_id = self.ids_info[self.mode][index]
        fixed, moving, fixed_seg, moving_seg = self.load_image_pair(fixed_id, moving_id, self.crop_x, self.crop_y, self.crop_z)
        
        if self.mode == 'train':
            fixed = fixed + np.random.normal(scale=0.001, size=fixed.shape)
            moving = moving + np.random.normal(scale=0.001, size=moving.shape)
        
        if self.normalize:
            fixed = self.__image_norm(fixed)
            moving = self.__image_norm(moving)
        
        fixed = torch.tensor(fixed, dtype=torch.float32).unsqueeze(0)
        fixed_seg = torch.tensor(fixed_seg, dtype=torch.float32).unsqueeze(0)
        moving = torch.tensor(moving, dtype=torch.float32).unsqueeze(0)
        moving_seg = torch.tensor(moving_seg, dtype=torch.float32).unsqueeze(0)
        
        xyz = self.get_grid()
        
        return (fixed, moving, xyz, fixed_seg, moving_seg)


class LPBA40Registration(Dataset):
    def __init__(self, dataset_path: str, 
                 crop_x: int=None, crop_y: int=None, crop_z: int=None, 
                 normalize: bool=True, 
                 mode: str='train') -> None:
        
        super().__init__()
        
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.mode = mode
        
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z
        
        if os.path.exists('tmp/lpba_train_val_test.json'):
            # loading the existing information
            with open('tmp/lpba_train_val_test.json', 'r') as handle:
                self.fixed_moving_ids = json.load(handle)[mode]
                    
        else:
            subject_ids = os.listdir(os.path.join(dataset_path, 'Delineation'))
            
            lpba_train_test = {}
            
            random.shuffle(subject_ids)
            
            train_pairs = self.randomized_pairs(subject_ids[:20], 10, 10)
            val_pairs = self.randomized_pairs(subject_ids[20: 25], 3, 2)
            test_pairs = self.randomized_pairs(subject_ids[25:], 8, 7)
            
            lpba_train_test['train'] = train_pairs
            lpba_train_test['val'] = val_pairs
            lpba_train_test['test'] = test_pairs
            # saving the information for future use
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/lpba_train_val_test.json', 'w') as handle:
                json.dump(lpba_train_test, handle)
            
            self.fixed_moving_ids = lpba_train_test[mode]
    
    def randomized_pairs(self, ids: List[str], num_moving_imgs: int, num_atlases: int) -> List[Tuple]:
        """
        This method receives the ids of all images, along with the nnumber of images to be considered as the 
        moving image and the number of images to be considered as atlases (or fixed images). The method then
        randomly samples from the ids and pairs up the ids of the moving images and the fixed/atlas images
        
        Args:
            ids (List[str]): a list of strings containing the ids of the images; e.g., 0001, 0034, 0411
            num_moving_imgs (int): the number of images to be considered as the moving image
            num_atlases (int): the number of images to be considered as the atlas or fixed images
            
        Returns:
            List[Tuple]: a list of the form [(f_id1, m_id1), (f_id2, m_id2), ...], where m_id is the id
                         of the moving image and the f_id is the id of the fixed image
        """
        # randomly choose the moving images
        moving_imgs_ids = random.sample(ids, num_moving_imgs)
        remained_ids = [i for i in ids if i not in moving_imgs_ids]
        atlas_ids = random.sample(remained_ids, num_atlases)
        
        fixed_moving_ids = list(product(atlas_ids, moving_imgs_ids))
        
        return fixed_moving_ids
    
    def load_image_pair(self, fixed_id, moving_id, crop_x, crop_y, crop_z):
        image_path = os.path.join(self.dataset_path, f'Delineation')
        
        # loading the fixed image
        fixed_img = nib.load(os.path.join(image_path, f'{fixed_id}/{fixed_id}.delineation.skullstripped.img')).get_fdata()[..., 0]
        fixed_seg = nib.load(os.path.join(image_path, f'{fixed_id}/{fixed_id}.delineation.structure.label.img')).get_fdata()[..., 0]
        
        # loading the moving image
        moving_img = nib.load(os.path.join(image_path, f'{moving_id}/{moving_id}.delineation.skullstripped.img')).get_fdata()[..., 0]
        moving_seg = nib.load(os.path.join(image_path, f'{moving_id}/{moving_id}.delineation.structure.label.img')).get_fdata()[..., 0]
        
        # croppin the image if specified
        fixed_img, fixed_seg, moving_img, moving_seg = self.__crop([fixed_img, fixed_seg, moving_img, moving_seg], crop_x, crop_y, crop_z)

        self.d, self.h, self.w = fixed_img.shape
        
        return fixed_img, moving_img, fixed_seg, moving_seg
    
    def get_grid(self) -> torch.Tensor:
        """
        This method constructs an empty 3D grid

        Returns:
            torch.Tensor: [B, w, h, d, 3] an empty grid.
        """
        z, y, x = np.meshgrid(np.arange(0, self.d), 
                              np.arange(0, self.h), 
                              np.arange(0, self.w), indexing='ij')
        
        
        return torch.tensor(np.stack([x, y, z], 3), dtype=torch.float32)
        
    def __crop(self, imgs_list: List[np.ndarray], crop_x: int, crop_y: int, crop_z: int) -> List[np.ndarray]:
        """
        This method receives a list of images (fixed image, fixed segmentation, moving image, moving segmentation) and the
        cropping information and crops the images along each axis (if specified)
        
        Args:
            imgs_list (List[np.ndarray]): a list containing the 3D images as numpy arrays
            crop_x (int): the new size of the image along the x direction
            crop_y (int): the new size of the image along the y direction
            crop_z (int): the new size of the image along the z direction
            
        Returns:
            List[np.ndarray]: a list of the same images as input but cropped
        """
        if crop_x is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_x, 0)
            
        if crop_y is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_y, 1)
            
        if crop_z is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_z, 2)
        
        return imgs_list
    
    def __crop_axis(self, imgs_list: List[np.ndarray], crop_size: int, axis: int=0) -> List[np.ndarray]:
        """
        This method crops the given list of image along the specified axis.
        
        Args:
            imgs_list (List[np.ndarray]): a list of images to be cropped.
            crop_size (int): the new size of the image along the given axis.
            axis (int) - Defaults to 0: the axis along which the cropping should be done (axis must be eiher 0, 1, or 2)
            
        Returns:
            List[np.ndarray]: a list of the same images as the input but cropped along the specified axis
        """
        
        size = imgs_list[0].shape[axis]
        start = size // 2 - crop_size // 2
        end = start + crop_size
        if axis == 0:
            cropped_list = [img[start: end, :, :] for img in imgs_list]
        elif axis == 1:
            cropped_list = [img[:, start: end, :] for img in imgs_list]
        else:
            cropped_list = [img[:, :, start: end] for img in imgs_list]
        
        return cropped_list
    
    def __image_norm(self, img: np.ndarray) -> torch.Tensor:
        """
        This method implements a simple min-max normalization on the images
        """
        
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def __len__(self):
        return len(self.fixed_moving_ids)
    
    def __getitem__(self, index):
        fixed_id, moving_id = self.fixed_moving_ids[index]
        fixed, moving, fixed_seg, moving_seg = self.load_image_pair(fixed_id, moving_id, self.crop_x, self.crop_y, self.crop_z)
        
        if self.normalize:
            fixed = self.__image_norm(fixed)
            moving = self.__image_norm(moving)
        
        fixed = torch.tensor(fixed, dtype=torch.float32).unsqueeze(0)
        fixed_seg = torch.tensor(fixed_seg, dtype=torch.float32).unsqueeze(0)
        moving = torch.tensor(moving, dtype=torch.float32).unsqueeze(0)
        moving_seg = torch.tensor(moving_seg, dtype=torch.float32).unsqueeze(0)
        
        xyz = self.get_grid()
        
        return (fixed, moving, xyz, fixed_seg, moving_seg)


class IXIRegistration(Dataset):
    def __init__(self, dataset_path: str, 
                 crop_x: int=None, crop_y: int=None, crop_z: int=None, 
                 normalize: bool=True, 
                 mode: str='train') -> None:
        
        super().__init__()
        
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.mode = mode
        
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z
        
        if os.path.exists('tmp/ixi_train_val_test.json'):
            # loading the existing information
            with open('tmp/ixi_train_val_test.json', 'r') as handle:
                self.fixed_moving_ids = json.load(handle)[mode]
                    
        else:
            subject_ids = os.listdir(dataset_path)
            
            lpba_train_test = {}
            
            random.shuffle(subject_ids)
            
            train_pairs = self.randomized_pairs(subject_ids[:20], 10, 10)
            val_pairs = self.randomized_pairs(subject_ids[20: 25], 3, 2)
            test_pairs = self.randomized_pairs(subject_ids[25:], 8, 7)
            
            lpba_train_test['train'] = train_pairs
            lpba_train_test['val'] = val_pairs
            lpba_train_test['test'] = test_pairs
            # saving the information for future use
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/ixi_train_val_test.json', 'w') as handle:
                json.dump(lpba_train_test, handle)
            
            self.fixed_moving_ids = lpba_train_test[mode]
    
    def randomized_pairs(self, ids: List[str], num_moving_imgs: int, num_atlases: int) -> List[Tuple]:
        """
        This method receives the ids of all images, along with the nnumber of images to be considered as the 
        moving image and the number of images to be considered as atlases (or fixed images). The method then
        randomly samples from the ids and pairs up the ids of the moving images and the fixed/atlas images
        
        Args:
            ids (List[str]): a list of strings containing the ids of the images; e.g., 0001, 0034, 0411
            num_moving_imgs (int): the number of images to be considered as the moving image
            num_atlases (int): the number of images to be considered as the atlas or fixed images
            
        Returns:
            List[Tuple]: a list of the form [(f_id1, m_id1), (f_id2, m_id2), ...], where m_id is the id
                         of the moving image and the f_id is the id of the fixed image
        """
        # randomly choose the moving images
        moving_imgs_ids = random.sample(ids, num_moving_imgs)
        remained_ids = [i for i in ids if i not in moving_imgs_ids]
        atlas_ids = random.sample(remained_ids, num_atlases)
        
        fixed_moving_ids = list(product(atlas_ids, moving_imgs_ids))
        
        return fixed_moving_ids
    
    def load_image_pair(self, fixed_id, moving_id, crop_x, crop_y, crop_z):

        # loading the fixed image
        fixed_img = nib.load(os.path.join(self.dataset_path, f'{fixed_id}/norm.mgz')).get_fdata()
        fixed_seg = nib.load(os.path.join(self.dataset_path, f'{fixed_id}/aseg.mgz')).get_fdata()
        
        # loading the moving image
        moving_img = nib.load(os.path.join(self.dataset_path, f'{moving_id}/norm.mgz')).get_fdata()
        moving_seg = nib.load(os.path.join(self.dataset_path, f'{moving_id}/aseg.mgz')).get_fdata()
        
        # croppin the image if specified
        fixed_img, fixed_seg, moving_img, moving_seg = self.__crop([fixed_img, fixed_seg, moving_img, moving_seg], crop_x, crop_y, crop_z)
        
        self.d, self.h, self.w = fixed_img.shape
        
        return fixed_img, moving_img, fixed_seg, moving_seg
    
    def get_grid(self) -> torch.Tensor:
        """
        This method constructs an empty 3D grid

        Returns:
            torch.Tensor: [B, w, h, d, 3] an empty grid.
        """
        z, y, x = np.meshgrid(np.arange(0, self.d), 
                              np.arange(0, self.h), 
                              np.arange(0, self.w), indexing='ij')
        
        
        return torch.tensor(np.stack([x, y, z], 3), dtype=torch.float32)
        
    def __crop(self, imgs_list: List[np.ndarray], crop_x: int, crop_y: int, crop_z: int) -> List[np.ndarray]:
        """
        This method receives a list of images (fixed image, fixed segmentation, moving image, moving segmentation) and the
        cropping information and crops the images along each axis (if specified)
        
        Args:
            imgs_list (List[np.ndarray]): a list containing the 3D images as numpy arrays
            crop_x (int): the new size of the image along the x direction
            crop_y (int): the new size of the image along the y direction
            crop_z (int): the new size of the image along the z direction
            
        Returns:
            List[np.ndarray]: a list of the same images as input but cropped
        """
        if crop_x is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_x, 0)
            
        if crop_y is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_y, 1)
            
        if crop_z is not None:
            imgs_list = self.__crop_axis(imgs_list, crop_z, 2)
        
        return imgs_list
    
    def __crop_axis(self, imgs_list: List[np.ndarray], crop_size: int, axis: int=0) -> List[np.ndarray]:
        """
        This method crops the given list of image along the specified axis.
        
        Args:
            imgs_list (List[np.ndarray]): a list of images to be cropped.
            crop_size (int): the new size of the image along the given axis.
            axis (int) - Defaults to 0: the axis along which the cropping should be done (axis must be eiher 0, 1, or 2)
            
        Returns:
            List[np.ndarray]: a list of the same images as the input but cropped along the specified axis
        """
        
        size = imgs_list[0].shape[axis]
        start = size // 2 - crop_size // 2
        end = start + crop_size
        if axis == 0:
            cropped_list = [img[start: end, :, :] for img in imgs_list]
        elif axis == 1:
            cropped_list = [img[:, start: end, :] for img in imgs_list]
        else:
            cropped_list = [img[:, :, start: end] for img in imgs_list]
        
        return cropped_list
    
    def __image_norm(self, img: np.ndarray) -> torch.Tensor:
        """
        This method implements a simple min-max normalization on the images
        """
        
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def __len__(self):
        return len(self.fixed_moving_ids)
    
    def __getitem__(self, index):
        fixed_id, moving_id = self.fixed_moving_ids[index]
        fixed, moving, fixed_seg, moving_seg = self.load_image_pair(fixed_id, moving_id, self.crop_x, self.crop_y, self.crop_z)
        
        if self.normalize:
            fixed = self.__image_norm(fixed)
            moving = self.__image_norm(moving)
        
        fixed = torch.tensor(fixed, dtype=torch.float32).unsqueeze(0)
        fixed_seg = torch.tensor(fixed_seg, dtype=torch.float32).unsqueeze(0)
        moving = torch.tensor(moving, dtype=torch.float32).unsqueeze(0)
        moving_seg = torch.tensor(moving_seg, dtype=torch.float32).unsqueeze(0)
        
        xyz = self.get_grid()
        
        return (fixed, moving, xyz, fixed_seg, moving_seg)

