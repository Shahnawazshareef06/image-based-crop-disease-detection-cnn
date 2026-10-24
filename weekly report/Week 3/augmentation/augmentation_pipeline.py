"""TerraVista PlantVillage PyTorch augmentation pipeline.

Use build_train_transform only for training examples. Use build_eval_transform
for validation/test examples. Background replacement is a heuristic and should
be enabled only after visually checking the foreground mask.
"""
import math
import random
from typing import Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMG_SIZE = 224
MAX_CUTOUT_AREA = 0.20
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKGROUND_PALETTE = [
    (166, 166, 158), (196, 181, 153), (151, 169, 178),
    (157, 169, 147), (210, 204, 188), (151, 139, 125),
]


def load_rgb_image(path: Union[str, "Path"]) -> Image.Image:
    """Open an image and return a detached RGB PIL image."""
    with Image.open(path) as image:
        return image.convert("RGB").copy()

def get_foreground_alpha(image: Image.Image) -> Optional[np.ndarray]:
    """Estimate the foreground/leaf mask as a float alpha image in [0, 1].

    GrabCut uses a central foreground seed and a thin sure-background border.
    Returns None if the estimated mask is implausibly small/large or GrabCut fails.
    Visually inspect the mask on several classes before relying on background replacement.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height < 24 or width < 24:
        return None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)

    # Mark only a thin border as definite background so leaves near the edge are less likely to be cut off.
    border = max(1, round(min(height, width) * 0.015))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Seed the central region as definite foreground; GrabCut refines its boundary.
    center = (width // 2, height // 2)
    axes = (max(5, int(width * 0.15)), max(5, int(height * 0.22)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, cv2.GC_FGD, thickness=-1)

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            bgr, mask, None, background_model, foreground_model,
            iterCount=3, mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None

    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    foreground_fraction = float(foreground.mean())
    if not 0.08 <= foreground_fraction <= 0.92:
        return None

    # Close tiny holes and softly feather the edge to reduce a hard composite seam.
    kernel = np.ones((3, 3), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.GaussianBlur(foreground.astype(np.float32), (0, 0), sigmaX=1.2)
    return np.clip(alpha, 0.0, 1.0)

def _make_background(height: int, width: int, mode: str = "random") -> np.ndarray:
    """Create a muted solid, gradient or softly textured synthetic RGB background."""
    allowed = {"random", "solid", "gradient", "texture"}
    if mode not in allowed:
        raise ValueError(f"mode must be one of {sorted(allowed)}")
    if mode == "random":
        mode = random.choice(["solid", "gradient", "texture"])

    palette = np.asarray(BACKGROUND_PALETTE, dtype=np.float32)
    if mode == "solid":
        colour = palette[random.randrange(len(palette))]
        background = np.empty((height, width, 3), dtype=np.float32)
        background[:] = colour
        # Small low-amplitude variation avoids a perfectly artificial flat colour.
        background += np.random.normal(0, 1.5, size=(height, width, 1)).astype(np.float32)

    elif mode == "gradient":
        colour_a = palette[random.randrange(len(palette))]
        colour_b = palette[random.randrange(len(palette))]
        if random.choice([True, False]):
            blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
            background = colour_a[None, None, :] * (1 - blend) + colour_b[None, None, :] * blend
            background = np.broadcast_to(background, (height, width, 3)).copy()
        else:
            blend = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
            background = colour_a[None, None, :] * (1 - blend) + colour_b[None, None, :] * blend
            background = np.broadcast_to(background, (height, width, 3)).copy()
        background += np.random.normal(0, 2.0, size=(height, width, 1)).astype(np.float32)

    else:  # texture
        base = palette[random.randrange(len(palette))]
        small_h, small_w = max(4, height // 18), max(4, width // 18)
        low_res = np.random.normal(
            loc=base, scale=12.0, size=(small_h, small_w, 3)
        ).astype(np.float32)
        background = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_CUBIC)
        background = cv2.GaussianBlur(background, (0, 0), sigmaX=3.0)

    return np.clip(background, 0, 255).astype(np.uint8)


def replace_background(
    image: Image.Image,
    mode: str = "random",
    alpha: Optional[np.ndarray] = None,
    return_mask: bool = False,
):
    """Replace the estimated background; keep the estimated foreground as-is.

    If segmentation is uncertain, the original image is returned unchanged.
    When return_mask=True, returns (image, alpha); alpha is None if segmentation failed.
    """
    image = image.convert("RGB")
    rgb = np.asarray(image)
    if alpha is None:
        alpha = get_foreground_alpha(image)
    if alpha is None:
        return (image.copy(), None) if return_mask else image.copy()

    background = _make_background(rgb.shape[0], rgb.shape[1], mode=mode)
    alpha_3d = alpha[:, :, None]
    composite = rgb.astype(np.float32) * alpha_3d + background.astype(np.float32) * (1.0 - alpha_3d)
    result = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), mode="RGB")
    return (result, alpha) if return_mask else result


class RandomBackgroundVariation:
    """Torchvision-compatible PIL transform that substitutes a background."""
    def __call__(self, image: Image.Image) -> Image.Image:
        return replace_background(image, mode="random")

def sample_cutout_box(
    height: int,
    width: int,
    min_area: float = 0.02,
    max_area: float = MAX_CUTOUT_AREA,
) -> tuple[int, int, int, int]:
    """Return (top, left, cut_h, cut_w) whose pixel area is <= max_area."""
    if not 0.0 < min_area <= max_area <= 0.20:
        raise ValueError("Require 0 < min_area <= max_area <= 0.20")
    target_area = random.uniform(min_area, max_area) * height * width
    aspect = random.uniform(0.5, 2.0)  # width / height
    cut_h = max(1, min(height, int(math.floor(math.sqrt(target_area / aspect)))))
    cut_w = max(1, min(width, int(math.floor(math.sqrt(target_area * aspect)))))
    top = random.randint(0, height - cut_h)
    left = random.randint(0, width - cut_w)
    return top, left, cut_h, cut_w


class RandomGaussianNoise:
    """Add mild Gaussian sensor noise to a [C,H,W] torch tensor in [0,1]."""
    def __init__(self, probability: float = 0.20, std: float = 0.02):
        self.probability = probability
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


class RandomCutout:
    """Hide one random rectangle; its area never exceeds max_area of the image."""
    def __init__(
        self,
        probability: float = 0.25,
        max_area: float = MAX_CUTOUT_AREA,
        min_area: float = 0.02,
    ):
        if not 0.0 < min_area <= max_area <= 0.20:
            raise ValueError("Require 0 < min_area <= max_area <= 0.20")
        self.probability = probability
        self.max_area = max_area
        self.min_area = min_area

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return tensor
        if tensor.ndim != 3:
            raise ValueError("RandomCutout expects a [C,H,W] image tensor")

        _, height, width = tensor.shape
        top, left, cut_h, cut_w = sample_cutout_box(
            height, width, min_area=self.min_area, max_area=self.max_area
        )
        output = tensor.clone()
        fill = output.mean(dim=(1, 2), keepdim=True)
        output[:, top:top + cut_h, left:left + cut_w] = fill
        return output

def build_train_transform(
    image_size: int = IMG_SIZE,
    strong: bool = False,
    background_probability: float = 0.20,
    cutout_max_area: float = MAX_CUTOUT_AREA,
    zoom_scale: Optional[tuple[float, float]] = None,
    normalize: bool = False,
    rotation_degrees: Optional[float] = None,
    brightness_range: Optional[tuple[float, float]] = None,
    contrast_range: Optional[tuple[float, float]] = None,
    saturation_range: Optional[tuple[float, float]] = None,
    hue_strength: Optional[float] = None,
    blur_probability: Optional[float] = None,
    blur_sigma: tuple[float, float] = (0.5, 1.2),
    noise_probability: Optional[float] = None,
    noise_std: Optional[float] = None,
    cutout_probability: Optional[float] = None,
) -> T.Compose:
    """Build a configurable PyTorch augmentation transform. Training images only.

    Exposed knobs make the settings testable as hyperparameters:
    - rotation_degrees and zoom_scale control geometric strength
    - brightness/contrast/saturation/hue control colour jitter
    - blur_probability, noise_probability/std control image imperfections
    - cutout_probability and cutout_max_area control occlusion
    - background_probability controls synthetic background replacement
    Set normalize=True when using ImageNet-pretrained ResNet-50 features.
    """
    if rotation_degrees is None:
        rotation_degrees = 40 if strong else 30
    if zoom_scale is None:
        zoom_scale = (0.75, 1.25) if strong else (0.85, 1.15)
    if brightness_range is None:
        brightness_range = (0.60, 1.40)
    if contrast_range is None:
        contrast_range = (0.75, 1.25) if strong else (0.80, 1.20)
    if saturation_range is None:
        saturation_range = (0.75, 1.25) if strong else (0.80, 1.20)
    if hue_strength is None:
        hue_strength = 0.03 if strong else 0.02
    if blur_probability is None:
        blur_probability = 0.30 if strong else 0.20
    if noise_probability is None:
        noise_probability = 0.30 if strong else 0.20
    if noise_std is None:
        noise_std = 0.03 if strong else 0.02
    if cutout_probability is None:
        cutout_probability = 0.35 if strong else 0.25

    if not 0.0 <= background_probability <= 1.0:
        raise ValueError("background_probability must be in [0, 1]")
    if not 0.0 < cutout_max_area <= MAX_CUTOUT_AREA:
        raise ValueError("cutout_max_area must be greater than 0 and at most 0.20")
    if not 0.0 < zoom_scale[0] <= zoom_scale[1]:
        raise ValueError("zoom_scale must be an increasing pair of positive values")
    if not 0.0 <= blur_probability <= 1.0:
        raise ValueError("blur_probability must be in [0, 1]")
    if not 0.0 <= noise_probability <= 1.0:
        raise ValueError("noise_probability must be in [0, 1]")
    if not 0.0 <= cutout_probability <= 1.0:
        raise ValueError("cutout_probability must be in [0, 1]")
    if not 0.0 <= hue_strength <= 0.5:
        raise ValueError("hue_strength must be in [0, 0.5]")

    steps = [
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(
            degrees=(-rotation_degrees, rotation_degrees),
            translate=(0.02, 0.02),
            scale=zoom_scale,
            interpolation=InterpolationMode.BILINEAR,
            fill=(127, 127, 127),
        ),
    ]

    if background_probability > 0:
        steps.append(T.RandomApply([RandomBackgroundVariation()], p=background_probability))

    steps.extend([
        T.ColorJitter(
            brightness=brightness_range,
            contrast=contrast_range,
            saturation=saturation_range,
            hue=hue_strength,
        ),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=blur_sigma)
        ], p=blur_probability),
        T.ToTensor(),
        RandomGaussianNoise(probability=noise_probability, std=noise_std),
        RandomCutout(probability=cutout_probability, max_area=cutout_max_area),
    ])

    if normalize:
        steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return T.Compose(steps)


def build_eval_transform(
    image_size: int = IMG_SIZE,
    normalize: bool = False,
) -> T.Compose:
    """Deterministic resize/tensor conversion for validation and test; no augmentation."""
    steps = [
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if normalize:
        steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(steps)
