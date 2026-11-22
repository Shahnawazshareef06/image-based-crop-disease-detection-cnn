"""Quick CPU smoke test for the TerraVista handoff package.

Run: python verify_handoff.py
This does not download pretrained weights or train on PlantVillage.
"""
from __future__ import annotations

from terravista_bootstrap import apply_kaggle_triton_guard
apply_kaggle_triton_guard()

import hashlib
import json
from pathlib import Path

import torch

from augmentation_pipeline import AUGMENTATION_RECIPES, augment_batch_gpu, get_augmentation_recipe
from terravista_resnet50_settings import seed_everything, build_resnet50, make_optimizer
from terravista_dataset import load_train_val_frames, notebook_cache_fingerprint

BASE = Path(__file__).resolve().parent


def main():
    class_to_idx = json.loads((BASE / "class_to_idx.json").read_text(encoding="utf-8"))
    manifest = json.loads((BASE / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    split_path = BASE / "train_val_split.csv"
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == manifest["dataset"]["train_val_manifest_sha256"]
    train_df, val_df = load_train_val_frames(split_path, class_to_idx)
    assert len(train_df) == 37751 and len(val_df) == 7976 and len(class_to_idx) == 38
    assert notebook_cache_fingerprint(train_df, val_df) == manifest["dataset"]["notebook_cache_fingerprint_for_224px_train_val"]

    seed_everything(42)
    # Random initialization is intentional: this smoke test must not need internet access.
    model = build_resnet50(device="cpu", pretrained=False)
    assert model.fc.out_features == 38
    optimizer = make_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 5e-4]
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 64, 64))
    assert tuple(logits.shape) == (1, 38)

    image_batch = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    alpha_batch = torch.full((2, 112, 112), 255, dtype=torch.uint8)
    alpha_batch[:, 12:100, 12:100] = 0
    for recipe_name in AUGMENTATION_RECIPES:
        recipe = get_augmentation_recipe(recipe_name)
        output = augment_batch_gpu(
            image_batch,
            recipe,
            device="cpu",
            background_alpha_uint8=alpha_batch if recipe["p_background"] > 0 else None,
        )
        assert tuple(output.shape) == (2, 3, 224, 224)
        assert torch.isfinite(output).all()

    print("PASS: bundled split/hash/class mapping")
    print("PASS: ResNet-50 38-class head and AdamW parameter groups")
    print("PASS: all five augmentation recipe branches")
    print("Note: this is a CPU smoke test, not a full training run.")


if __name__ == "__main__":
    main()
