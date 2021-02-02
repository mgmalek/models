import torch.nn as nn
from torchvision import models
from .utils import generate_map_data, get_conv_block
from .layers import ConvBNRelu


def prepare_resnet_for_ssd(base_model: nn.Module, last_channels: int) -> nn.Module:
    """Prepare a resnet model for use in an SSD by removing classification
    layers and adding layers required to produce smaller feature maps"""
    # Source for ResNet structure with SSD is:
    # https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
    body = list(base_model.children())[:7] # Remove avgpool, fc layer and final resnet block
    body[-1][0].conv1.stride = (1, 1)
    body[-1][0].conv2.stride = (1, 1)
    body[-1][0].downsample[0].stride = (1, 1)
    
    body = nn.Sequential(
        *body,
        get_conv_block(last_channels, 512),
        get_conv_block(512, 512),
        get_conv_block(512, 256),
        get_conv_block(256, 256, stride=1, padding=0),
        get_conv_block(256, 128, stride=1, padding=0),
    )

    return body


def ssd_body_resnet18(pretrained: bool = True, layers_to_add: int = 5) -> nn.Module:
    """Return a modified version of the resnet18 model ready for use as the
    body of an SSD"""
    base_model = models.resnet18(pretrained=pretrained)
    prepared_model = prepare_resnet_for_ssd(base_model, 256)
    return prepared_model


def ssd_body_resnet34(pretrained: bool = True, layers_to_add: int = 5) -> nn.Module:
    """Return a modified version of the resnet34 model ready for use as the
    body of an SSD"""
    base_model = models.resnet34(pretrained=pretrained)
    prepared_model = prepare_resnet_for_ssd(base_model, 256)
    return prepared_model


def ssd_body_resnet50(pretrained: bool = True, layers_to_add: int = 5) -> nn.Module:
    """Return a modified version of the resnet50 model ready for use as the
    body of an SSD"""
    base_model = models.resnet50(pretrained=pretrained)
    prepared_model = prepare_resnet_for_ssd(base_model, 1024)
    return prepared_model


def ssd_body_resnet101(pretrained: bool = True, layers_to_add: int = 5) -> nn.Module:
    """Return a modified version of the resnet101 model ready for use as the
    body of an SSD"""
    base_model = models.resnet101(pretrained=pretrained)
    prepared_model = prepare_resnet_for_ssd(base_model, 1024)
    return prepared_model


def ssd_body_resnet152(pretrained: bool = True, layers_to_add: int = 5) -> nn.Module:
    """Return a modified version of the resnet152 model ready for use as the
    body of an SSD"""
    base_model = models.resnet152(pretrained=pretrained)
    prepared_model = prepare_resnet_for_ssd(base_model, 1024)
    return prepared_model
