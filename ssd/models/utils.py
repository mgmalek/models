import torch
import torch.nn as nn
from .layers import ConvBNRelu
from typing import List, Dict


def generate_map_data(img_size: int, map_sizes: List[int], body: nn.Module) -> List[Dict]:
    """Generate a dictionary containing the layer index and number of output
    channels for each layer in `body` that generates feature maps used directly
    for object detection"""
    map_data = {}
    for map_size in map_sizes:
        map_data[map_size] = {
            "layer_idx": None,
            "num_channels": None,
        }

    inp = torch.randn(8, 3, img_size, img_size).cuda()
    for idx, layer in enumerate(body):
        inp = layer(inp)
        channels = inp.size(1)
        map_size = inp.size(2)
        if map_size in map_data:
            map_data[map_size]["layer_idx"] = idx
            map_data[map_size]["num_channels"] = channels
    
    for map_size, data in map_data.items():
        if data['layer_idx'] is None or data["num_channels"] is None:
            raise RuntimeError((f"Unable to find a layer with an output map size of {map_size}. "
                                "Consider adding additional layers to the model, changing "
                                "the input size or changing the desired map sizes."))
    
    return map_data


def get_conv_block(in_channels: int, out_channels: int, stride: int = 2,
                   padding: int = None, depthwise: bool = False) -> nn.Module:
    """Return a series of two convolutional layers with batch norm and ReLU"""
    mid_channels = out_channels // 2
    groups = mid_channels if depthwise else 1
    return nn.Sequential(
        *ConvBNRelu(in_channels, mid_channels, kernel_size=1, stride=1, groups=groups),
        *ConvBNRelu(mid_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=groups),
    )
