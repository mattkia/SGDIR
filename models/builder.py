"""Implementations of model builders used in trainers
"""

from models.layers import UNet3D
from models.layers import UNet2D
from models.layers import LatentDiT3D
from models.layers import LatentDiT2D

from models.models import SGDIR
from models.models import SGDIR2D
from models.models import SGDIRDiT
from models.models import SGDIRDiT2D


def build_model(config: dict) -> SGDIR | SGDIRDiT:
    """Construct an instance of 3D SGDIR or SGDIRDiT

    :param config:
        A dictionary specifying the architecture of the backbone;
        examples can be seen in architecture field of the config
        files

    :returns:
        An instance of 3D SGDIR or SGDIRDiT
    """
    model_type = config.get('model_type')
    loss_type = config.get('loss_type', 'ncc')
    architecture_configs = config.get('architecture')

    assert model_type in ['unet', 'dit'], f'{model_type} is not implemented.'

    
    if model_type == 'unet':
        backbone = UNet3D(**architecture_configs)
    else:
        backbone = LatentDiT3D(**architecture_configs)

    
    if model_type == 'dit':
        sgdir = SGDIRDiT(backbone=backbone, loss_type=loss_type)
    else:
        sgdir = SGDIR(backbone=backbone, loss_type=loss_type)

    return sgdir

def build_model_2d(config: dict) -> SGDIR | SGDIRDiT:
    """Construct an instance of 2D SGDIR or SGDIRDiT

    :param config:
        A dictionary specifying the architecture of the backbone;
        examples can be seen in architecture field of the config
        files

    :returns:
        An instance of 2D SGDIR or SGDIRDiT
    """
    model_type = config.get('model_type')
    loss_type = config.get('loss_type', 'ncc')
    architecture_configs = config.get('architecture')

    assert model_type in ['unet', 'dit'], f'{model_type} is not implemented.'
    
    if model_type == 'unet':
        backbone = UNet2D(**architecture_configs)
    else:
        backbone = LatentDiT2D(**architecture_configs)

    
    if model_type == 'dit':
        sgdir = SGDIRDiT2D(backbone=backbone, loss_type=loss_type)
    else:
        sgdir = SGDIR2D(backbone=backbone, loss_type=loss_type)

    return sgdir
