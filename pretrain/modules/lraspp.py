# Torch's source code


from collections import OrderedDict
from typing import Any, Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter

import pdb

__all__ = ["LRASPP", "lraspp_mobilenet_v3_large"]


model_urls = {
    "lraspp_mobilenet_v3_large_coco": "https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
}


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result



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
        return out, sal


def get_backbone_lraspp_mobilenetv3(pretrained=True):
    '''
    modify MobileNetV3 backbone so it can be attached to LR-ASPP head
    based on torch's implementation
    '''
    import torchvision.mobilenetv3 as mobilenetv3
    backbone = mobilenetv3.mobilenet_v3_large(pretrained=pretrained, dilated=True)
    
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  
    high_pos = stage_indices[-1]  
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return backbone, low_channels, high_channels


