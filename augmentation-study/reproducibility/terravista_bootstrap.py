"""Optional Kaggle startup guard for the known PyTorch/Triton incompatibility.

Call apply_kaggle_triton_guard() before importing torch, torchvision,
augmentation_pipeline, or terravista_resnet50_settings.
"""
from __future__ import annotations

import os
import sys


def apply_kaggle_triton_guard() -> bool:
    """Hide Triton before torch import; return True when the guard was applied.

    Set TERRAVISTA_KEEP_TRITON=1 to opt out. This matches the original study
    notebook, which does not use torch.compile. A fresh Python process/kernel
    is required if torch has already been imported.
    """
    if os.environ.get("TERRAVISTA_KEEP_TRITON", "").strip() == "1":
        return False
    if "torch" in sys.modules:
        raise RuntimeError(
            "Triton guard must run before importing torch. Restart the Python/Kaggle "
            "kernel and call apply_kaggle_triton_guard() as the first project action."
        )
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            sys.modules.pop(name, None)
    sys.modules["triton"] = None
    return True
