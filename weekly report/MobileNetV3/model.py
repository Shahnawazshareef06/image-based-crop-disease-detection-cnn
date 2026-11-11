"""MobileNetV3 model + optimizer for the TerraVista PlantVillage protocol."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from config import BACKBONE_LR, CLASSIFIER_LR, NUM_CLASSES, WEIGHT_DECAY


def build_mobilenetv3(variant: str, device) -> nn.Module:
    """ImageNet-pretrained MobileNetV3 with a 38-class head, all layers trainable."""
    if variant == "large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    else:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    for p in model.parameters():
        p.requires_grad = True
    return model.to(device)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """AdamW, two groups: new 38-class head at CLASSIFIER_LR, all pretrained layers at BACKBONE_LR.

    ResNet-50: head = model.fc.  MobileNetV3: head = model.classifier[3] (the only new layer).
    """
    head_params = list(model.classifier[3].parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
