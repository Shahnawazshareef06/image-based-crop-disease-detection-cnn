"""PyTorch Model Definition for Frozen ViT-Tiny on PlantVillage (38 Classes).

Implements:
1. Native PyTorch VisionTransformerTiny architecture (embed_dim=192, depth=12, heads=3).
2. TIMM compatibility loader for 'vit_tiny_patch16_224' with automatic fallback.
3. Backbone freezing protocol (eval mode, requires_grad=False for backbone,
   only model.head trainable with 7,334 parameters).
4. Head checkpoint save/load utility for compact weight persistence.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    """Splits 224x224x3 image into 16x16 non-overlapping patches (196 tokens)."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 192, 14, 14) -> (B, 196, 192)
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformerTiny(nn.Module):
    """Pure PyTorch implementation of ViT-Tiny (DeiT-Tiny) matching timm specification."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 38,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        d_ffn = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=d_ffn,
                dropout=drop_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # Truncated normal weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # Return [CLS] token representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(feat)


def build_vit_tiny(
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Builds and initializes ViT-Tiny with frozen backbone and trainable 38-class head."""
    device = torch.device(device)
    model: nn.Module

    try:
        import timm
        # Load standard ImageNet-1k pretrained ViT-Tiny
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception:
        # Fallback to pure PyTorch implementation
        model = VisionTransformerTiny(num_classes=num_classes)

    if freeze_backbone:
        # Freeze entire backbone
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze ONLY classification head
        for param in model.head.parameters():
            param.requires_grad = True

        # Keep backbone permanently in eval mode (disables dropout, fixes LayerNorm stats)
        model.eval()

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"[ViT-Tiny Setup] Total Parameters: {total_params:,} | "
        f"Trainable (Head): {trainable_params:,} | "
        f"Frozen (Backbone): {total_params - trainable_params:,}"
    )
    return model


def save_head_checkpoint(model: nn.Module, save_path: str | Path) -> None:
    """Saves only the 7,334 head parameters to disk (lightweight ~30 KB file)."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head_state = model.head.state_dict()
    torch.save(head_state, path)
    print(f"[Checkpoint] Saved trained head weights ({len(head_state)} tensors) to {path}")


def load_head_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Loads head parameters into a pre-initialized ViT-Tiny model."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    state = torch.load(path, map_location="cpu")
    model.head.load_state_dict(state)
    print(f"[Checkpoint] Successfully restored head weights from {path}")
    return model


if __name__ == "__main__":
    print("Testing ViT-Tiny Model Initialization...")
    m = build_vit_tiny(num_classes=38, pretrained=False, freeze_backbone=True, device="cpu")
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(dummy_input)
    print(f"Forward Pass Successful: Output shape = {out.shape} (Expected: [2, 38])")
