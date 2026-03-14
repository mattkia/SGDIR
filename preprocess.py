"""Implementations of image preprocessing functions
"""

import math
import torch
import random

import numpy as np
import SimpleITK as sitk

from itertools import product


def randomized_pairs(
    ids: list[str],
    num_moving_imgs: int,
    num_fixed_imgs: int
) -> list[tuple]:
    """Receives the ids of all images, along with the number
    of images to be considered as the moving and fixed images.
    The method then randomly samples from the ids and pairs up
    the ids of the moving and fixed images
    
    :param ids:
        A list of strings containing the ids of all images;
        e.g., 0001, 0034, 0411

    :param num_moving_imgs:
        The number of images to be considered as the moving
        image

    :param num_fixed_imgs:
        The number of images to be considered as the atlas
        or fixed images
        
    :returns:
        A list of the form [(f_id1, m_id1), (f_id2, m_id2), ...],
        where m_id is the id of the moving image and the f_id is
        the id of the fixed image
    """
    # randomly choose the moving images
    moving_imgs_ids = random.sample(ids, num_moving_imgs)
    remained_ids = [i for i in ids if i not in moving_imgs_ids]
    fixed_ids = random.sample(remained_ids, num_fixed_imgs)
    
    fixed_moving_ids = list(product(fixed_ids, moving_imgs_ids))
    
    return fixed_moving_ids

def get_id_grid(shape: tuple) -> torch.Tensor:
    """Constructs a 2D or 3D identity grid

    :param shape:
        Spatial dimensions of the image.
        [D, H, W] or [H, W]

    :returns:
        An identity grid with shape [D, H, W, 3].
    """
    if len(shape) == 3:
        d, h, w = shape
        z, y, x = np.meshgrid(np.arange(d),
                              np.arange(h),
                              np.arange(w),
                              indexing='ij')
        id_grid = np.stack([x, y, z], 3)
    else:
        h, w = shape
        y, x = np.meshgrid(np.arange(h),
                           np.arange(w),
                              indexing='ij')
        id_grid = np.stack([x, y], 2)

    id_grid = torch.tensor(id_grid, dtype=torch.float32)
    
    return id_grid

def crop(
    imgs_list: list[np.ndarray],
    crop_x: int,
    crop_y: int,
    crop_z: int
) -> list[np.ndarray]:
    """Receives a list of images (fixed image, fixed segmentation,
    moving image, moving segmentation) and thecropping information
    and crops the images along each axis (if specified).
    
    :param imgs_list:
        A list containing the 3D images as numpy arrays.

    :param crop_x:
        The new size of the image along the x direction.

    :param crop_y:
        The new size of the image along the y direction.

    :param crop_z:
        The new size of the image along the z direction.
        
    :returns:
        A list of cropped images.
    """
    if crop_x is not None:
        imgs_list = crop_axis(imgs_list, crop_x, 0)
        
    if crop_y is not None:
        imgs_list = crop_axis(imgs_list, crop_y, 1)
        
    if crop_z is not None:
        imgs_list = crop_axis(imgs_list, crop_z, 2)
    
    return imgs_list
    
def crop_axis(
    imgs_list: list[np.ndarray],
    crop_size: int,
    axis: int=0
) -> list[np.ndarray]:
    """Crops the given list of image along the specified axis.
    
    :param imgs_list:
        A list of images to be cropped.

    :param crop_size:
        The new size of the image along the given axis.

    :param axis:
        Defaults to 0; The axis along which the cropping
        should be done (axis must be eiher 0, 1, or 2)

    :returns:
        A list of the cropped images along the specified axis
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

def image_norm(img: np.ndarray) -> np.ndarray:
    """Applies min-max normalization on an image.

    :param img:
        The unnormalized input image

    :returns:
        The min-max intensity normalized image.
    """
    
    img = (img - img.min()) / (img.max() - img.min())
    
    return img

def resampler_by_transform(
    im_sitk: sitk.Image,
    dvf_t: sitk.Transform,
    im_ref: sitk.Image=None, 
    default_pixel_value: float=0, 
    interpolator=sitk.sitkBSpline
) -> sitk.Image:
    """Resamples an image using a given displacement vector field transform.

    :param im_sitk:
        The input SimpleITK image to be resampled.

    :param dvf_t:
        The displacement vector field transform to apply.

    :param im_ref:
        The reference image for resampling. If None, a default reference
        image is created based on the transform's displacement field.

    :param default_pixel_value:
        The default pixel value to use for pixels outside the input image.

    :param interpolator:
        The interpolation method to use for resampling.
        
    :returns:
        The resampled SimpleITK image.
    """
    if im_ref is None:
        im_ref = sitk.Image(dvf_t.GetDisplacementField().GetSize(), sitk.sitkInt8)
        im_ref.SetOrigin(dvf_t.GetDisplacementField().GetOrigin())
        im_ref.SetSpacing(dvf_t.GetDisplacementField().GetSpacing())
        im_ref.SetDirection(dvf_t.GetDisplacementField().GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(dvf_t)
    out_im = resampler.Execute(im_sitk)
    return out_im

def resampler_sitk(
    image: np.ndarray,
    spacing: tuple=None,
    scale: float=None, 
    im_ref: sitk.Image=None,
    im_ref_size: tuple=None,
    default_pixel_value: float=0,
    interpolator=sitk.sitkBSpline,
    dimension: int=3
) -> np.ndarray:
    """Resamples a numpy array image using SimpleITK with specified spacing or scale.

    :param image:
        The input image as a numpy array.

    :param spacing:
        The desired spacing for the resampled image. If None, scale must be provided.

    :param scale:
        The scale factor for resampling. If None, spacing must be provided.

    :param im_ref:
        The reference SimpleITK image. If None, a default one is created.

    :param im_ref_size:
        The size of the reference image. If None, calculated based on spacing or scale.

    :param default_pixel_value:
        The default pixel value for pixels outside the input image.

    :param interpolator:
        The interpolation method to use.

    :param dimension:
        The dimension of the image (2 or 3).
        
    :returns:
        The resampled image as a numpy array.
    """
    image = image + 1024

    image_sitk = sitk.GetImageFromArray(image)

    if spacing is None and scale is None:
        raise ValueError('spacing and scale cannot be both None')

    if spacing is None:
        spacing = tuple(i * scale for i in image_sitk.GetSpacing())
        if im_ref_size is None:
            im_ref_size = tuple(round(i / scale) for i in image_sitk.GetSize())

    elif scale is None:
        ratio = [spacing_dim / spacing[i] for i, spacing_dim in enumerate(image_sitk.GetSpacing())]
        if im_ref_size is None:
            im_ref_size = tuple(math.ceil(size_dim * ratio[i]) for i, size_dim in enumerate(image_sitk.GetSize()))
    else:
        raise ValueError('spacing and scale cannot both have values')

    if im_ref is None:
        im_ref = sitk.Image(im_ref_size, sitk.sitkInt8)
        im_ref.SetOrigin(image_sitk.GetOrigin())
        im_ref.SetDirection(image_sitk.GetDirection())
        im_ref.SetSpacing(spacing)
    identity = sitk.Transform(dimension, sitk.sitkIdentity)
    resampled_sitk = resampler_by_transform(image_sitk, identity, im_ref=im_ref,
                                            default_pixel_value=default_pixel_value,
                                            interpolator=interpolator)
    
    resampled_img = sitk.GetArrayFromImage(resampled_sitk)

    return resampled_img

def affine_register(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
    moving_seg_np: np.ndarray,
    spacing: tuple=(1.0, 1.0, 1.0),
    origin: tuple=(0, 0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Performs affine registration between fixed and moving images using SimpleITK.

    :param fixed_np:
        The fixed image as a numpy array.

    :param moving_np:
        The moving image as a numpy array.

    :param moving_seg_np:
        The moving segmentation as a numpy array.

    :param spacing:
        The spacing of the images.

    :param origin:
        The origin of the images.
        
    :returns:
        A tuple containing the registered image and registered segmentation as numpy arrays.
    """
    # Convert numpy arrays to SimpleITK images
    fixed_img = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    moving_img = sitk.GetImageFromArray(moving_np.astype(np.float32))
    moving_seg = sitk.GetImageFromArray(moving_seg_np.astype(np.int16))

    for im in [fixed_img, moving_img, moving_seg]:
        im.SetSpacing(spacing)
        im.SetOrigin(origin)

    # Registration method
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(learningRate=1.0,
                                               numberOfIterations=100,
                                               convergenceMinimumValue=1e-6,
                                               convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.AffineTransform(fixed_img.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Run registration
    final_transform = registration.Execute(fixed_img, moving_img)

    # Resample moving image
    registered_img = sitk.Resample(
        moving_img, fixed_img, final_transform,
        sitk.sitkLinear, 0.0, moving_img.GetPixelID()
    )
    # Resample segmentation (nearest neighbor)
    registered_seg = sitk.Resample(
        moving_seg, fixed_img, final_transform,
        sitk.sitkNearestNeighbor, 0, moving_seg.GetPixelID()
    )

    # Convert back to numpy
    registered_img_np = sitk.GetArrayFromImage(registered_img)
    registered_seg_np = sitk.GetArrayFromImage(registered_seg)

    return registered_img_np, registered_seg_np
