"""Reproducible ResNet-50 settings for the TerraVista PlantVillage experiment.

The constants and helpers mirror the completed five-recipe comparison notebook.
Call seed_everything(seed) BEFORE creating the model, optimizer, or DataLoaders
for each (recipe, seed) run.
"""
from __future__ import annotations

import random
import platform
from typing import Optional, Union

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

# ---------------------------------------------------------------------------
# Dataset / preprocessing contract (must match the shared split manifest)
# ---------------------------------------------------------------------------
NUM_CLASSES = 38
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Matched five-recipe comparison protocol
# ---------------------------------------------------------------------------
SEEDS = (42, 123, 2026)
COMPARISON_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 4
PIN_MEMORY = True

# In the original notebook, workers/prefetch were reduced only when /dev/shm
# was constrained. Keep this helper so teammates get the same safeguards.
def resolved_loader_workers() -> tuple[int, int]:
    workers = DEFAULT_NUM_WORKERS
    prefetch = DEFAULT_PREFETCH_FACTOR
    try:
        import shutil
        free_mib = shutil.disk_usage("/dev/shm").free / (1024 ** 2)
        if free_mib < 256:
            workers, prefetch = 2, 2
        elif free_mib < 1024:
            prefetch = 2
    except OSError:
        pass
    return workers, prefetch

NUM_WORKERS, PREFETCH_FACTOR = resolved_loader_workers()

# ---------------------------------------------------------------------------
# Model and optimizer — do not alter during augmentation recipe comparisons
# ---------------------------------------------------------------------------
ARCHITECTURE = "ResNet-50"
PRETRAINED_WEIGHTS = "ResNet50_Weights.DEFAULT"
OPTIMIZER_NAME = "AdamW"
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 5e-4
WEIGHT_DECAY = 1e-4
LOSS_NAME = "CrossEntropyLoss"
USE_AMP = True
AMP_DTYPE = torch.float16
SCHEDULER_NAME = None  # The matched comparison notebook uses no LR scheduler.

# Notebook's deterministic cuDNN / matmul settings.
def configure_deterministic_backends() -> None:
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, Torch CPU and all CUDA RNGs for one experiment run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_deterministic_backends()
    return seed


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs inside each DataLoader worker, as in the study."""
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_generator(seed: int) -> torch.Generator:
    """Dedicated DataLoader generator; use a fresh one for each training run."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_dataloader_kwargs(*, seed: int, training: bool) -> dict:
    """Return DataLoader kwargs matching the notebook's batch/worker policy.

    Example:
        loader = DataLoader(dataset, **make_dataloader_kwargs(seed=42, training=True))
    """
    workers, prefetch = NUM_WORKERS, PREFETCH_FACTOR
    kwargs = {
        "batch_size": TRAIN_BATCH_SIZE if training else VALIDATION_BATCH_SIZE,
        "shuffle": bool(training),
        "num_workers": workers,
        "pin_memory": PIN_MEMORY,
        "worker_init_fn": seed_worker,
    }
    if training:
        kwargs["generator"] = make_data_generator(seed)
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch
    return kwargs


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def build_resnet50(
    device: Optional[Union[str, torch.device]] = None,
    *,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build ImageNet-pretrained ResNet-50 with a 38-class linear head.

    All layers are trainable (full fine-tuning), matching the comparison study.
    `pretrained=False` is intended only for offline smoke tests.
    """
    if int(num_classes) != NUM_CLASSES:
        raise ValueError(f"This experiment expects {NUM_CLASSES} classes, got {num_classes}.")
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model.to(resolve_device(device))


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """AdamW with separate backbone and classifier parameter groups."""
    if not hasattr(model, "fc"):
        raise ValueError("Expected a ResNet-50-like model with classifier named 'fc'.")
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    classifier_params = list(model.fc.parameters())
    if not backbone_params or not classifier_params:
        raise ValueError("Model backbone or classifier parameters are missing.")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": classifier_params, "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def make_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


def make_grad_scaler(device: Optional[Union[str, torch.device]] = None):
    resolved = resolve_device(device)
    enabled = bool(USE_AMP and resolved.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: Union[str, torch.device]):
    resolved = resolve_device(device)
    enabled = bool(USE_AMP and resolved.type == "cuda")
    return torch.autocast(
        device_type=resolved.type,
        dtype=AMP_DTYPE,
        enabled=enabled,
    )


def model_settings_dict() -> dict:
    return {
        "architecture": ARCHITECTURE,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "num_classes": NUM_CLASSES,
        "all_layers_trainable": True,
        "image_size": IMAGE_SIZE,
        "input_layout": "RGB uint8 NHWC before augmentation; float32 NCHW after augmentation",
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "optimizer": OPTIMIZER_NAME,
        "backbone_lr": BACKBONE_LR,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": LOSS_NAME,
        "scheduler": SCHEDULER_NAME,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "comparison_epochs": COMPARISON_EPOCHS,
        "seeds": list(SEEDS),
        "mixed_precision": USE_AMP,
        "amp_dtype": "float16",
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "high",
    }


def environment_report() -> dict:
    """Return versions/hardware for logging next to every experiment result."""
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
