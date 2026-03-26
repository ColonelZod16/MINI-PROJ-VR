"""
models.py
=========
All 3 required CNN architectures adapted for multi-label classification:
  - ResNet-50
  - EfficientNet-B0 / B2
  - MobileNetV3-Large

Each model:
  - Uses pretrained ImageNet weights
  - Replaces the final head with a multi-label classification head
  - Outputs raw logits (NO sigmoid) — sigmoid + BCE applied in loss
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,            ResNet50_Weights,
    efficientnet_b0,     EfficientNet_B0_Weights,
    efficientnet_b2,     EfficientNet_B2_Weights,
    mobilenet_v3_large,  MobileNet_V3_Large_Weights,
)


# ─────────────────────────────────────────────
# SHARED HEAD
# ─────────────────────────────────────────────
def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """
    Shared multi-label classification head.
    Outputs raw logits — apply sigmoid at inference.
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes),   # raw logits, no sigmoid
    )


# ─────────────────────────────────────────────
# 1. ResNet-50
# ─────────────────────────────────────────────
class ResNet50MultiLabel(nn.Module):
    """
    ResNet-50 pretrained on ImageNet.
    Final FC replaced with multi-label head.
    """
    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Keep everything except the final FC
        self.backbone   = nn.Sequential(*list(base.children())[:-1])  # ends at avgpool
        self.classifier = _make_head(2048, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(x)              # [B, 2048, 1, 1]
        feats  = feats.flatten(1)              # [B, 2048]
        return self.classifier(feats)          # [B, num_classes]


# ─────────────────────────────────────────────
# 2. EfficientNet-B0 / B2
# ─────────────────────────────────────────────
class EfficientNetMultiLabel(nn.Module):
    """
    EfficientNet-B0 or B2 pretrained on ImageNet.
    Classifier head replaced with multi-label head.

    variant : "b0"  →  in_features=1280
              "b2"  →  in_features=1408
    """
    VARIANTS = {
        "b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
        "b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408),
    }

    def __init__(self, num_classes: int = 5, variant: str = "b0", dropout: float = 0.4):
        super().__init__()
        assert variant in self.VARIANTS, f"variant must be 'b0' or 'b2'"
        build_fn, weights, in_features = self.VARIANTS[variant]

        base = build_fn(weights=weights)
        self.features   = base.features        # CNN backbone
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = _make_head(in_features, num_classes, dropout)
        self.variant    = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.features(x)              # [B, C, H, W]
        feats  = self.pool(feats).flatten(1)   # [B, C]
        return self.classifier(feats)          # [B, num_classes]


# ─────────────────────────────────────────────
# 3. MobileNetV3-Large
# ─────────────────────────────────────────────
class MobileNetV3MultiLabel(nn.Module):
    """
    MobileNetV3-Large pretrained on ImageNet.
    Classifier head replaced with multi-label head.
    Lightest model — fastest to train, good for quick baseline.
    """
    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # MobileNetV3 head: features → avgpool → flatten → classifier
        self.features   = base.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = _make_head(960, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.features(x)              # [B, 960, H, W]
        feats  = self.pool(feats).flatten(1)   # [B, 960]
        return self.classifier(feats)          # [B, num_classes]


# ─────────────────────────────────────────────
# FACTORY  — single entry point
# ─────────────────────────────────────────────
def build_model(
    arch:        str,
    num_classes: int   = 5,
    dropout:     float = 0.4,
    variant:     str   = "b0",   # only used when arch="efficientnet"
) -> nn.Module:
    """
    arch options:
        "resnet50"       → ResNet-50
        "efficientnet"   → EfficientNet-B0 or B2 (set variant="b0" or "b2")
        "mobilenetv3"    → MobileNetV3-Large
    """
    arch = arch.lower()
    if arch == "resnet50":
        model = ResNet50MultiLabel(num_classes=num_classes, dropout=dropout)
    elif arch == "efficientnet":
        model = EfficientNetMultiLabel(num_classes=num_classes, variant=variant, dropout=dropout)
    elif arch == "mobilenetv3":
        model = MobileNetV3MultiLabel(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose: resnet50 | efficientnet | mobilenetv3")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    name      = f"EfficientNet-{variant.upper()}" if arch == "efficientnet" else arch
    print(f"[Model] {name:<25}  total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M")
    return model
