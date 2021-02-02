import torch.nn as nn


class ConvBNRelu(nn.Sequential):
    """A standard 2d convolutional layer followed by batch norm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 2, groups: int = 1,
                 padding: int = None):
        if padding is None: padding = kernel_size // 2
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        
        super().__init__(conv, bn, relu)
