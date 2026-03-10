import math

import numpy as np
import SimpleITK as sitk


def resampler_by_transform(im_sitk, dvf_t, im_ref=None, 
                           default_pixel_value=0, 
                           interpolator=sitk.sitkBSpline):
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

def resampler_sitk(image, spacing=None, scale=None, 
                   im_ref=None, im_ref_size=None, 
                   default_pixel_value=0, 
                   interpolator=sitk.sitkBSpline, 
                   dimension=3):
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
            # im_ref_size = tuple(math.ceil(size_dim * ratio[i]) - 1 for i, size_dim in enumerate(image_sitk.GetSize()))
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

def affine_register(fixed_np, moving_np, moving_seg_np, spacing=(1.0,1.0,1.0), origin=(0,0,0)):
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