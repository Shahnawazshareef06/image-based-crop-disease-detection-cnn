"""Sanity check to verify ViT-Tiny model, dimensions, loss, optimizer, and augmentation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    AUGMENTATION_RECIPES,
    BACKBONE_LR,
    CLASSIFIER_LR,
    IMAGE_SIZE,
    NUM_CLASSES,
    WEIGHT_DECAY,
)
from model import build_vit_tiny, count_parameters, get_optimizer_param_groups

try:
    from augmentation_pipeline import augment_batch_gpu, validate_probability_config
except ImportError:
    from TerraVista_Reproducible_Team_Handoff.augmentation_pipeline import (
        augment_batch_gpu,
        validate_probability_config,
    )


def resolve_device() -> torch.device:
    """Detect working CUDA accelerator or fallback to CPU with documented notice."""
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"Notice: CUDA is reported by driver but unavailable at runtime ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def run_sanity_check() -> None:
    print("=======================================================")
    print("STEP 5: ViT-TINY MODEL AND PIPELINE SANITY CHECK")
    print("=======================================================")

    device = resolve_device()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Target Device: {device} ({dev_name})")

    # 1. Instantiate ViT-Tiny
    print("\n1. Instantiating ViT-Tiny (vit_tiny_patch16_224)...")
    model = build_vit_tiny(num_classes=NUM_CLASSES, pretrained=False)
    model.to(device)
    print("   Success: Model created and moved to target device.")

    # 2. Check model parameters
    print("\n2. Checking Parameter Counts & Trainable Scope...")
    params = count_parameters(model)
    print(f"   Total Parameters:       {params['total_parameters']:,}")
    print(f"   Trainable Parameters:   {params['trainable_parameters']:,}")
    print(f"   Backbone Parameters:    {params['backbone_parameters']:,}")
    print(f"   Classifier Parameters:  {params['classifier_parameters']:,}")

    assert params["total_parameters"] == params["trainable_parameters"], (
        "Error: All parameters must be trainable for full fine-tuning."
    )
    assert model.head.out_features == NUM_CLASSES, (
        f"Error: Expected {NUM_CLASSES} output classes, found {model.head.out_features}."
    )
    print("   Success: Full fine-tuning scope and 38-class head verified.")

    # 3. Optimizer parameter groups
    print("\n3. Verifying Optimizer Parameter Groups...")
    param_groups = get_optimizer_param_groups(model)
    print(f"   Number of parameter groups: {len(param_groups)}")
    for i, g in enumerate(param_groups):
        print(f"   Group {i} ({g['name']}): lr={g['lr']}, weight_decay={g['weight_decay']}, num_tensors={len(g['params'])}")

    assert len(param_groups) == 2, "Must have exactly 2 parameter groups."
    assert param_groups[0]["lr"] == BACKBONE_LR, f"Backbone LR should be {BACKBONE_LR}"
    assert param_groups[1]["lr"] == CLASSIFIER_LR, f"Classifier LR should be {CLASSIFIER_LR}"
    assert param_groups[0]["weight_decay"] == WEIGHT_DECAY
    assert param_groups[1]["weight_decay"] == WEIGHT_DECAY
    print("   Success: Backbone and Classifier separate LR groups verified.")

    # 4. Input & Output Dimensions
    print("\n4. Verifying Forward Pass & Tensor Shapes...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_targets = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    outputs = model(dummy_input)
    print(f"   Input shape:  {list(dummy_input.shape)}")
    print(f"   Output shape: {list(outputs.shape)}")
    assert outputs.shape == (batch_size, NUM_CLASSES), f"Shape mismatch: {outputs.shape}"
    print("   Success: Model correctly outputs (batch_size, 38) logits.")

    # 5. Loss calculation & backward pass
    print("\n5. Verifying Loss Calculation & Gradient Flow...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, dummy_targets)
    loss.backward()
    print(f"   CrossEntropyLoss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss must be positive"
    assert model.head.weight.grad is not None, "Classification head must have gradients"
    print("   Success: Gradients successfully backpropagated.")

    # 6. Verify Augmentation Recipes
    print("\n6. Verifying Top 5 Augmentation Recipes...")
    dummy_uint8 = torch.randint(0, 256, (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8, device=device)
    dummy_masks = torch.full((batch_size, 112, 112), 255, dtype=torch.uint8, device=device)

    for name, probs in AUGMENTATION_RECIPES.items():
        validate_probability_config(probs)
        need_mask = probs.get("p_background", 0.0) > 0.0
        aug_out = augment_batch_gpu(
            dummy_uint8, probs, device=device, background_alpha_uint8=dummy_masks if need_mask else None
        )
        assert aug_out.shape == (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        print(f"   Recipe '{name:25s}': Verified OK (Output shape: {list(aug_out.shape)})")

    print("\n=======================================================")
    print("ALL SANITY CHECKS COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    run_sanity_check()
