"""Implementations of different datasets and loaders
"""

import os
import json
import torch
import random

import numpy as np
import pandas as pd
import nibabel as nib

from abc import ABC
from abc import abstractmethod
from torch.utils.data import Dataset

from preprocess import crop
from preprocess import image_norm
from preprocess import get_id_grid
from preprocess import resampler_sitk
from preprocess import affine_register
from preprocess import randomized_pairs


def get_dataset(
    config: dict,
    train: bool=True,
    **kwargs
) -> Dataset:
    """Receives the configuration dictionary and returns the dataset
    specified in the configs along with the determined specifications.
    
    :param config:
        An instance of the configs

    :param train:
        Defaults to True; dataset mode (train/val/test)
        
    :returns:
        Torch friendly dataset of the specified data.
    """
    name = config.get('name')
    dataset_path = config.get('path')
    crop_x = config.get('crop_x', None)
    crop_y = config.get('crop_y', None)
    crop_z = config.get('crop_z', None)
    
    if 'val' in kwargs.keys():
        if kwargs['val']:
            mode = 'val'
    else:
        if train:
            mode = 'train'
        else:
            mode = 'test'

    if name == 'oasis':
        target_dataset = OASISRegistration
    elif name == 'candi':
        target_dataset = CANDIRegistration
    elif name == 'lpba40':
        target_dataset = LPBA40Registration
    elif name == 'ixi':
        target_dataset = IXIRegistration
    elif name == 'mindboggle':
        target_dataset = MindboggleRegistration
    elif name == 'acdc':
        target_dataset = ACDCRegistration
    elif name == 'abdomen':
        target_dataset = AbdomenCTRegistration
    elif name == 'lungct':
        target_dataset = LungCTRegistration
    else:
        raise Exception(f'[!] Dataset {name} does not exist in the set of experiments. '
                        f'You need to implement your own data loader')
    
    dataset = target_dataset(dataset_path=dataset_path,
                             crop_x=crop_x,
                             crop_y=crop_y,
                             crop_z=crop_z,
                             mode=mode)
    
    return dataset


class RegistrationDataset(ABC, Dataset):
    """Registration dataset interface
    """
    def __init__(
        self,
        dataset_path: str,
        crop_x: int=None,
        crop_y: int=None,
        crop_z: int=None,
        mode: str='train'
    ):
        """
        :param dataset_path:
            Path to where the OASIS dataset is stored.

        :param crop_x:
            The size of image along x direction after center cropping
            [D, H, W] -> [crop_x, H, W]

        :param crop_y:
            The size of image along y direction after center cropping
            [D, H, W] -> [D, crop_y, W]

        :param crop_z:
            The size of image along z direction after center cropping
            [D, H, W] -> [D, H, crop_z]

        :param normalize:
            If true, normalizes the intesities into [0, 1] range.
        
        :param mode:
            The dataset to return (train/val/test)
        """
        
        self.dataset_path = dataset_path
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z
        self.mode = mode

        self.fixed_moving_ids = self.setup_fixed_moving_ids()
    
    @abstractmethod
    def setup_fixed_moving_ids(self) -> list[tuple[str, str]]:
        """Implements dataset-specific logic to determine
        the (fixed, moving) ids for train, val. and test
        modes.

        :returns:
            A list of (fixed_id, moving_id) corresponding to
            the self.mode (i.e., train, val, test)
        """
        pass

    @abstractmethod
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str
    ) -> tuple[np.ndarray]:
        """Receives the ids of the fixed and moving images
        and returns the center-cropped fixed and moving images 
        with the segmentation masks or keypoints.
        
        :param fixed_id:
            The id of the fixed image (e.g., 0001)

        :param moving_id:
            The id of the moving image (e.g., 0002)

        :returns:
            A four-tupple containing the fixed image,
            moving image, fixed mask, and moving mask
        """
        pass

    def __len__(self):
        return len(self.fixed_moving_ids)
    
    def __getitem__(self, index):
        fixed_id, moving_id = self.fixed_moving_ids[index]
        fixed, moving, fixed_seg, moving_seg = self.load_image_pair(fixed_id,
                                                                    moving_id)
        
        fixed = image_norm(fixed)
        moving = image_norm(moving)
        
        fixed = torch.tensor(fixed, dtype=torch.float32).unsqueeze(0)
        fixed_seg = torch.tensor(fixed_seg, dtype=torch.float32).unsqueeze(0)
        moving = torch.tensor(moving, dtype=torch.float32).unsqueeze(0)
        moving_seg = torch.tensor(moving_seg, dtype=torch.float32).unsqueeze(0)
        
        img_shape = fixed.shape[1:]
        id_grid = get_id_grid(img_shape)
        
        return fixed, moving, id_grid, fixed_seg, moving_seg


class OASISRegistration(RegistrationDataset):
    """Dataset definition of the OASIS dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str
    ) -> tuple[np.ndarray]:
        fixed_path = os.path.join(os.path.join(self.dataset_path, f'OASIS_OAS1_{fixed_id}_MR1'))
        moving_path = os.path.join(os.path.join(self.dataset_path, f'OASIS_OAS1_{moving_id}_MR1'))
        
        # loading the fixed image
        fixed_img = nib.load(os.path.join(fixed_path, 'aligned_norm.nii.gz')).get_fdata()
        fixed_seg35 = nib.load(os.path.join(fixed_path, 'aligned_seg35.nii.gz')).get_fdata()
        
        # loading the moving image
        moving_img = nib.load(os.path.join(moving_path, 'aligned_norm.nii.gz')).get_fdata()
        moving_seg35 = nib.load(os.path.join(moving_path, 'aligned_seg35.nii.gz')).get_fdata()
        
        # croppin the image if specified
        fixed_img, fixed_seg35, moving_img, moving_seg35 = crop([fixed_img, fixed_seg35, moving_img, moving_seg35],
                                                                self.crop_x, self.crop_y, self.crop_z)
        
        return fixed_img, moving_img, fixed_seg35, moving_seg35

    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/oasis_train_val_test.json'):
            with open('tmp/oasis_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]
                    
        else:
            available_subjects = os.listdir(self.dataset_path)
            available_subjects = list(filter(lambda x: x.startswith('OASIS'), available_subjects))
            subject_ids = list(map(lambda x: x.split('_')[-2].strip(), available_subjects))

            random.shuffle(subject_ids)
            
            train_pairs = randomized_pairs(subject_ids[:256], 40, 40)
            val_pairs = randomized_pairs(subject_ids[256: 306], 10, 10)
            test_pairs = randomized_pairs(subject_ids[306:], 15, 15)

            # saving the information for future use
            oasis_train_test = {'train': train_pairs,
                                'val': val_pairs,
                                'test': test_pairs}
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/oasis_train_val_test.json', 'w') as handle:
                json.dump(oasis_train_test, handle)

            return oasis_train_test[self.mode]


class CANDIRegistration(RegistrationDataset):
    """Dataset definition of the CANDI dataset
    """    
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str,
    ) -> tuple[np.ndarray]:
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
        
        # cropping the image if specified
        fixed_img = np.pad(fixed_img, ((2, 3), (2, 1), (2, 3)))
        fixed_seg = np.pad(fixed_seg, ((2, 3), (2, 1), (2, 3)))
        moving_img = np.pad(moving_img, ((2, 3), (2, 1), (2, 3)))
        moving_seg = np.pad(moving_seg, ((2, 3), (2, 1), (2, 3)))
        
        fixed_img, fixed_seg, moving_img, moving_seg = crop([fixed_img, fixed_seg, moving_img, moving_seg],
                                                            self.crop_x, self.crop_y, self.crop_z)
        
        return fixed_img, moving_img, fixed_seg, moving_seg

    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/candi_train_val_test.json'):
            # loading the existing information
            with open('tmp/candi_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]
                    
        else:
            available_cats = ['BPDwithoutPsy', 'BPDwithPsy', 'HC', 'SS']
            all_ids = []

            for category in available_cats:
                path = os.path.join(self.dataset_path, f'SchizBull_2008/{category}')
                subject_ids = os.listdir(path)
                subject_ids = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), subject_ids))
                all_ids.extend(subject_ids)

            random.shuffle(all_ids)
            train_length = int(0.7 * len(all_ids))
            val_length = int(0.1 * len(all_ids)) 
            
            train_pairs = randomized_pairs(all_ids[:train_length], 20, 20)
            val_pairs = randomized_pairs(all_ids[train_length: train_length + val_length], 5, 5)
            test_pairs = randomized_pairs(all_ids[train_length + val_length:], 5, 5)

            # saving the information for future use
            candi_train_test = {'train': train_pairs,
                                'val': val_pairs,
                                'test': test_pairs}
            
            # saving the information for future use
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/candi_train_val_test.json', 'w') as handle:
                json.dump(candi_train_test, handle)
            
            return candi_train_test[self.mode]


class LPBA40Registration(RegistrationDataset):
    """Dataset definition of the LPBA40 dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str
    ) -> tuple[np.ndarray]:
        image_path = os.path.join(self.dataset_path, f'Delineation')
        
        # loading the fixed image
        fixed_img = nib.load(os.path.join(image_path, f'{fixed_id}/{fixed_id}.delineation.skullstripped.img')).get_fdata()[..., 0]
        fixed_seg = nib.load(os.path.join(image_path, f'{fixed_id}/{fixed_id}.delineation.structure.label.img')).get_fdata()[..., 0]
        
        # loading the moving image
        moving_img = nib.load(os.path.join(image_path, f'{moving_id}/{moving_id}.delineation.skullstripped.img')).get_fdata()[..., 0]
        moving_seg = nib.load(os.path.join(image_path, f'{moving_id}/{moving_id}.delineation.structure.label.img')).get_fdata()[..., 0]

        # modifying the image shape
        fixed_img = fixed_img.transpose(0, 2, 1)
        fixed_seg = fixed_seg.transpose(0, 2, 1)
        moving_img = moving_img.transpose(0, 2, 1)
        moving_seg = moving_seg.transpose(0, 2, 1)
        
        # croppin the image if specified
        fixed_img, fixed_seg, moving_img, moving_seg = crop([fixed_img, fixed_seg, moving_img, moving_seg],
                                                            self.crop_x, self.crop_y, self.crop_z)
        moving_img, moving_seg = affine_register(fixed_img, moving_img, moving_seg)
        
        return fixed_img, moving_img, fixed_seg, moving_seg

    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/lpba_train_val_test.json'):
            # loading the existing information
            with open('tmp/lpba_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]
                    
        else:
            subject_ids = os.listdir(os.path.join(self.dataset_path, 'Delineation'))

            random.shuffle(subject_ids)
            
            train_pairs = randomized_pairs(subject_ids[:20], 10, 10)
            val_pairs = randomized_pairs(subject_ids[20: 25], 3, 2)
            test_pairs = randomized_pairs(subject_ids[25:], 8, 7)
            
            # saving the information for future use
            lpba_train_test = {'train': train_pairs,
                               'val': val_pairs,
                               'test': test_pairs}

            os.makedirs('tmp', exist_ok=True)
            with open('tmp/lpba_train_val_test.json', 'w') as handle:
                json.dump(lpba_train_test, handle)
            
            return lpba_train_test[self.mode]


class IXIRegistration(RegistrationDataset):
    """Dataset definition of the IXI dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str,
    ) -> tuple[np.ndarray]:
        # loading the fixed image
        fixed_img = nib.load(os.path.join(self.dataset_path, f'{fixed_id}/norm.mgz')).get_fdata()
        fixed_seg = nib.load(os.path.join(self.dataset_path, f'{fixed_id}/aseg.mgz')).get_fdata()
        
        # loading the moving image
        moving_img = nib.load(os.path.join(self.dataset_path, f'{moving_id}/norm.mgz')).get_fdata()
        moving_seg = nib.load(os.path.join(self.dataset_path, f'{moving_id}/aseg.mgz')).get_fdata()
        
        # croppin the image if specified
        fixed_img, fixed_seg, moving_img, moving_seg = crop([fixed_img, fixed_seg, moving_img, moving_seg],
                                                            self.crop_x, self.crop_y, self.crop_z)
        
        return fixed_img, moving_img, fixed_seg, moving_seg
    
    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/ixi_train_val_test.json'):
            # loading the existing information
            with open('tmp/ixi_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]
                    
        else:
            subject_ids = os.listdir(self.dataset_path)
            
            random.shuffle(subject_ids)
            
            train_pairs = randomized_pairs(subject_ids[:20], 10, 10)
            val_pairs = randomized_pairs(subject_ids[20: 25], 3, 2)
            test_pairs = randomized_pairs(subject_ids[25:], 8, 7)
            
            # saving the information for future use
            ixi_train_test = {'train': train_pairs,
                              'val': val_pairs,
                              'test': test_pairs}
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/ixi_train_val_test.json', 'w') as handle:
                json.dump(ixi_train_test, handle)
            
            return ixi_train_test[self.mode]


class MindboggleRegistration(RegistrationDataset):
    """Dataset definition of the Mindboggle dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str
    ) -> tuple[np.ndarray]:
        img_path = os.path.join(self.dataset_path, 'images')
        label_path = os.path.join(self.dataset_path, 'labels')
        # loading the fixed image
        fixed_img = nib.load(os.path.join(img_path, f'{fixed_id}.nii.gz')).get_fdata().transpose(0, 2, 1)
        fixed_seg = nib.load(os.path.join(label_path, f'{fixed_id}.nii.gz')).get_fdata().transpose(0, 2, 1)
        
        # loading the moving image
        moving_img = nib.load(os.path.join(img_path, f'{moving_id}.nii.gz')).get_fdata().transpose(0, 2, 1)
        moving_seg = nib.load(os.path.join(label_path, f'{moving_id}.nii.gz')).get_fdata().transpose(0, 2, 1)
        
        # croppin the image if specified
        fixed_img, fixed_seg, moving_img, moving_seg = crop([fixed_img, fixed_seg, moving_img, moving_seg],
                                                            self.crop_x, self.crop_y, self.crop_z)
        
        return fixed_img, moving_img, fixed_seg, moving_seg
    
    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/mindboggle_train_val_test.json'):
            # loading the existing information
            with open('tmp/mindboggle_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]
            
        else:
            available_subjects = os.listdir(os.path.join(self.dataset_path, 'images'))
            available_subjects = list(filter(lambda x: 'flipped' not in x, available_subjects))
            subject_ids = list(map(lambda x: x[:-7].strip(), available_subjects))

            random.shuffle(subject_ids)

            train_pairs = randomized_pairs(subject_ids[:80], 20, 20)
            val_pairs = randomized_pairs(subject_ids[80: 86], 2, 2)
            test_pairs = randomized_pairs(subject_ids[86:], 7, 7)

            # saving the information for future use
            mindboggle_train_test = {'train': train_pairs,
                                     'val': val_pairs,
                                     'test': test_pairs}
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/mindboggle_train_val_test.json', 'w') as handle:
                json.dump(mindboggle_train_test, handle)

            return mindboggle_train_test[self.mode]


class LungCTRegistration(RegistrationDataset):
    """Dataset definition of the LungCT dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str
    ) -> tuple[np.ndarray]:
        fixed_path = fixed_id.split('./')[1]
        fixed_path = os.path.join(self.dataset_path, fixed_path)

        fixed_keypoints_path = fixed_path.replace('images', 'keypoints')
        fixed_keypoints_path = fixed_keypoints_path.replace('.nii.gz', '.csv')

        # settin up the moving image, mask, and keypoints path
        moving_path = moving_id.split('./')[1]
        moving_path = os.path.join(self.dataset_path, moving_path)

        moving_keypoints_path = moving_path.replace('images', 'keypoints')
        moving_keypoints_path = moving_keypoints_path.replace('.nii.gz', '.csv')

        # loading the fixed image, mask, and keypoints
        fixed_img = nib.load(fixed_path).get_fdata().clip(-800, 700)

        fixed_keypoints = pd.read_csv(fixed_keypoints_path)
        fixed_keypoints.columns = ['x', 'y', 'z']
        fixed_keypoints[['x', 'y', 'z']] = fixed_keypoints[['z', 'y', 'x']]
        fixed_keypoints = fixed_keypoints.values

        # loading the moving image, mask, and keypoints
        moving_img = nib.load(moving_path).get_fdata().clip(-800, 700)

        moving_keypoints = pd.read_csv(moving_keypoints_path)
        moving_keypoints.columns = ['x', 'y', 'z']
        moving_keypoints[['x', 'y', 'z']] = moving_keypoints[['z', 'y', 'x']]
        moving_keypoints = moving_keypoints.values

        fixed_img = resampler_sitk(fixed_img, spacing=[1, 1, 1])
        moving_img = resampler_sitk(moving_img, spacing=[1, 1, 1])

        fixed_img, fixed_keypoints = self.__half_resize(fixed_img, fixed_keypoints)
        moving_img, moving_keypoints = self.__half_resize(moving_img, moving_keypoints)

        return fixed_img, moving_img, fixed_keypoints, moving_keypoints
    
    def setup_fixed_moving_ids(self):
        with open(os.path.join(self.dataset_path, 'LungCT_dataset.json'), 'r') as handle:
            fixed_moving_ids = json.load(handle)

        if self.mode == 'train':
            fixed_moving_ids = fixed_moving_ids['training_paired_images']
        elif self.mode == 'val':
            fixed_moving_ids = fixed_moving_ids['registration_val']
        else:
            fixed_moving_ids = fixed_moving_ids['registration_test']

        fixed_moving_ids = [(item['fixed'], item['moving']) for item in fixed_moving_ids]

        return fixed_moving_ids

    def __half_resize(self, img, kp):
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        img = torch.nn.functional.interpolate(img, scale_factor=0.5, mode='trilinear')[0, 0].numpy()
        kp = kp / 2

        return img, kp


class AbdomenCTRegistration(RegistrationDataset):
    """Dataset definition of the AbdomenCT dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str,
    ):
        img_path = os.path.join(self.dataset_path, 'imagesTr')
        labels_path = os.path.join(self.dataset_path, 'labelsTr')

        fixed_path = os.path.join(img_path, f'AbdomenCTCT_{fixed_id}_0000.nii.gz')
        fixed_seg_path = os.path.join(labels_path, f'AbdomenCTCT_{fixed_id}_0000.nii.gz')
        moving_path = os.path.join(img_path, f'AbdomenCTCT_{moving_id}_0000.nii.gz')
        moving_seg_path = os.path.join(labels_path, f'AbdomenCTCT_{moving_id}_0000.nii.gz')

        fixed_img = nib.load(fixed_path).get_fdata().clip(-1000, 1000)
        fixed_seg = nib.load(fixed_seg_path).get_fdata()

        moving_img = nib.load(moving_path).get_fdata().clip(-1000, 1000)
        moving_seg = nib.load(moving_seg_path).get_fdata()

        fixed_img, fixed_seg, moving_img, moving_seg = crop([fixed_img, fixed_seg, moving_img, moving_seg],
                                                            self.crop_x, self.crop_y, self.crop_z)

        return fixed_img, moving_img, fixed_seg, moving_seg

    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/abdomen_train_val_test.json'):
            # loading the existing information
            with open('tmp/abdomen_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]

        else:
            subject_ids = [f'{i + 1:04d}' for i in range(30)]

            train_pairs = randomized_pairs(subject_ids[:18], 8, 9)
            val_pairs = randomized_pairs(subject_ids[18: 20], 1, 1)
            test_pairs = randomized_pairs(subject_ids[20:], 5, 5)

            # saving the information for future use
            abdomen_train_test = {'train': train_pairs,
                                  'val': val_pairs,
                                  'test': test_pairs}
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/abdomen_train_val_test.json', 'w') as handle:
                json.dump(abdomen_train_test, handle)

            return abdomen_train_test[self.mode]


class ACDCRegistration(RegistrationDataset):
    """Dataset definition of the ACDC dataset
    """
    def load_image_pair(
        self,
        fixed_id: str,
        moving_id: str
    ) -> tuple[np.ndarray]:
        subject_id = fixed_id
        data_path = os.path.join(self.dataset_path, subject_id)

        fixed_id, moving_id = self.__find_fixed_moving_ids(data_path)

        fixed_img = nib.load(os.path.join(data_path, f'{subject_id}_frame{fixed_id}.nii.gz')).get_fdata()
        fixed_seg = nib.load(os.path.join(data_path, f'{subject_id}_frame{fixed_id}_gt.nii.gz')).get_fdata()
        
        moving_img = nib.load(os.path.join(data_path, f'{subject_id}_frame{moving_id}.nii.gz')).get_fdata()
        moving_seg = nib.load(os.path.join(data_path, f'{subject_id}_frame{moving_id}_gt.nii.gz')).get_fdata()

        fixed_img, fixed_seg, moving_img, moving_seg = crop([fixed_img, fixed_seg, moving_img, moving_seg],
                                                            self.crop_x, self.crop_y, None)
        
        mid_frame = fixed_img.shape[-1] // 2
        fixed_img = fixed_img[..., mid_frame]
        fixed_seg = fixed_seg[..., mid_frame]
        moving_img = moving_img[..., mid_frame]
        moving_seg = moving_seg[..., mid_frame]

        return fixed_img, moving_img, fixed_seg, moving_seg
    
    def setup_fixed_moving_ids(self):
        if os.path.exists('tmp/acdc_train_val_test.json'):
            # loading the existing information
            with open('tmp/acdc_train_val_test.json', 'r') as handle:
                return json.load(handle)[self.mode]
            
        else:
            available_subjects = os.listdir(self.dataset_path)
            available_subjects = list(filter(lambda x: os.path.isdir(os.path.join(self.dataset_path, x)), available_subjects))
            available_subjects = sorted(available_subjects)

            train_pairs = available_subjects[:90]
            val_pairs = available_subjects[90: 101]
            test_pairs = available_subjects[101:]

            train_pairs = [(subject, subject) for subject in train_pairs]
            val_pairs = [(subject, subject) for subject in val_pairs]
            test_pairs = [(subject, subject) for subject in test_pairs]

            # saving the information for future use
            acdc_train_val_test = {'train': train_pairs,
                                   'val': val_pairs,
                                   'test': test_pairs}
            
            os.makedirs('tmp', exist_ok=True)
            with open('tmp/acdc_train_val_test.json', 'w') as handle:
                json.dump(acdc_train_val_test, handle)

            return acdc_train_val_test[self.mode]

    def __find_fixed_moving_ids(self, data_path: str):
        data_files = os.listdir(data_path)

        image_files = list(filter(lambda x: 'frame' in x and 'gt' not in x, data_files))

        all_ids = set(map(lambda x: x.split('_')[1].split('.')[0][-2:], image_files))

        sorted_ids = sorted(list(all_ids))

        fixed_id, moving_id = sorted_ids

        return fixed_id, moving_id
