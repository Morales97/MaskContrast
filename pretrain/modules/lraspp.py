# Torch's source code

from collections import OrderedDict
from typing import Any, Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter

import pdb


class LRASPPHead_with_saliency(nn.Module):
    '''
    Modified LR-ASPP head from torch's implementation
    Added a saliency prediction head (binary segmentation) for MaskContrast method
    '''
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

        self.low_classifier_saliency = nn.Conv2d(low_channels, 1, 1)
        self.high_classifier_saliency = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        out = self.low_classifier(low) + self.high_classifier(x)
        sal = self.low_classifier_saliency(low) + self.high_classifier_saliency(x)
        x = {'out': out, 'sal': sal}
        return x


def get_backbone_lraspp_mobilenetv3(pretrained=False):
    '''
    modify MobileNetV3 backbone so it can be attached to LR-ASPP head
    based on torch's implementation
    '''
    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained, dilated=True)
    
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  
    high_pos = stage_indices[-1]  
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return backbone, low_channels, high_channels


