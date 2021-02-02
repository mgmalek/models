import torch.nn as nn
from torchvision import models
from .utils import get_conv_block


def ssd_body_mobilenet_v2(pretrained: bool = True) -> nn.Module:
    """Return a version of the the mobilenet_v2 architecture prepared for use
    in an SSD"""
    model = models.mobilenet_v2(pretrained=pretrained)
    model = model.features  # Remove classifier layers
    model = model[:-1]  # Remove final layer (with 1280-channel output)
    
    model = nn.Sequential(
        *model,
        get_conv_block(320, 320, depthwise=True),
        get_conv_block(320, 320, depthwise=True),
        get_conv_block(320, 320, depthwise=True),
        get_conv_block(320, 320, depthwise=True),
    )

    return model
