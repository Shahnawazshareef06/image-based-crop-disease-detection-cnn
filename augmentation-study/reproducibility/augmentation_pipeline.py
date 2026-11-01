"""TerraVista PyTorch augmentation pipeline.

This module matches the augmentation implementation used by
TerraVista_Augmentation_Probability_Tuning_Final.ipynb.

Expected training input:
    images_uint8: uint8 RGB tensor shaped [batch, height, width, 3]
    config: mapping with p_flip, p_rotation, p_colour, p_blur, p_noise,
            p_cutout, and p_background probabilities (each in [0, 1])
    background_alpha_uint8: optional uint8 mask tensor [batch, mask_h, mask_w],
            required when p_background > 0; 255 preserves the leaf, 0 replaces background.

Returns ImageNet-normalized float tensor shaped [batch, 3, height, width].
Apply random augmentation to training images only. For validation, pass an all-zero
probability config and do not supply/rewrite validation backgrounds.
"""
from __future__ import annotations

import math
from typing import Mapping, Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

cv2.setNumThreads(1)

# Fixed augmentation strengths used by the probability-tuning notebook.
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROTATION_DEGREES = 15.0
BRIGHTNESS_RANGE = (0.60, 1.40)  # Includes darkening.
CONTRAST_RANGE = (0.80, 1.20)
SATURATION_RANGE = (0.80, 1.20)
HUE_STRENGTH = 0.02
SENSOR_NOISE_STD = 0.02
BLUR_KERNEL_SIZE = 5
CUTOUT_MIN_AREA = 0.02
CUTOUT_MAX_AREA = 0.15
BACKGROUND_MASK_SIZE = 112
BACKGROUND_MASK_CACHE_VERSION = "grabcut-112px-3iter-v1"
BACKGROUND_REPLACEMENT_PROBABILITY = 0.10
BACKGROUND_PALETTE = [

    (166, 166, 158), (196, 181, 153), (151, 169, 178),
    (157, 169, 147), (210, 204, 188), (151, 139, 125),
]
PROBABILITY_KEYS = (
    "p_flip", "p_rotation", "p_colour", "p_blur", "p_noise", "p_cutout", "p_background"
)


# Validation-ranked recipes from the matched five-recipe study.
# Scores are mean best validation macro-F1 over seeds 42, 123, and 2026.
# The no-augmentation baseline ranked first overall; uniform_p010 ranked first
# among augmented recipes.
AUGMENTATION_RECIPES = {
    "no_augmentation": {
        "p_flip": 0.0, "p_rotation": 0.0, "p_colour": 0.0,
        "p_blur": 0.0, "p_noise": 0.0, "p_cutout": 0.0,
        "p_background": 0.0,
    },
    "uniform_p010": {
        "p_flip": 0.10, "p_rotation": 0.10, "p_colour": 0.10,
        "p_blur": 0.10, "p_noise": 0.10, "p_cutout": 0.10,
        "p_background": 0.10,
    },
    "uniform_p020": {
        "p_flip": 0.20, "p_rotation": 0.20, "p_colour": 0.20,
        "p_blur": 0.20, "p_noise": 0.20, "p_cutout": 0.20,
        "p_background": 0.20,
    },
    "joint_refined": {
        "p_flip": 0.0, "p_rotation": 0.0, "p_colour": 0.25,
        "p_blur": 0.10, "p_noise": 0.25, "p_cutout": 0.25,
        "p_background": 0.10,
    },
    "uniform_p025": {
        "p_flip": 0.25, "p_rotation": 0.25, "p_colour": 0.25,
        "p_blur": 0.25, "p_noise": 0.25, "p_cutout": 0.25,
        "p_background": 0.25,
    },
}

# Higher value means higher mean validation macro-F1 in the matched
# five-recipe, 3-seed, 3-epoch comparison.
AUGMENTATION_RECIPE_VALIDATION_MACRO_F1 = {
    "no_augmentation": 0.984665,
    "uniform_p010": 0.984099,
    "uniform_p020": 0.983800,
    "joint_refined": 0.983067,
    "uniform_p025": 0.982333,
}

def get_augmentation_recipe(name: str) -> dict[str, float]:
    """Return a fresh copy of a named validation-ranked recipe."""
    if name not in AUGMENTATION_RECIPES:
        choices = ", ".join(AUGMENTATION_RECIPES)
        raise KeyError(f"Unknown recipe {name!r}. Available recipes: {choices}")
    config = dict(AUGMENTATION_RECIPES[name])
    validate_probability_config(config)
    return config



def zero_probability_config() -> dict[str, float]:
    """Return a no-random-augmentation config suitable for validation/baseline."""
    return {key: 0.0 for key in PROBABILITY_KEYS}


def single_family_config(family: str, probability: float) -> dict[str, float]:
    """Create a config enabling only one family at the requested probability."""
    family_to_key = {
        "flip": "p_flip",
        "rotation": "p_rotation",
        "colour": "p_colour",
        "blur": "p_blur",
        "sensor_noise": "p_noise",
        "cutout": "p_cutout",
        "background": "p_background",
    }
    if family not in family_to_key:
        raise ValueError(f"Unknown family {family!r}; choose one of {sorted(family_to_key)}")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    config = zero_probability_config()
    config[family_to_key[family]] = float(probability)
    return config


def validate_probability_config(config: Mapping[str, Any]) -> None:
    """Raise a useful error if a probability is missing/out of range."""
    for key in PROBABILITY_KEYS:
        value = float(config.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1], got {value}")


def estimate_foreground_alpha_from_rgb(rgb_uint8, output_size=BACKGROUND_MASK_SIZE):
    """Return alpha uint8 (255=preserve plant, 0=replace background) and mask-valid flag."""
    small = cv2.resize(rgb_uint8, (output_size, output_size), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    if h < 24 or w < 24:
        return np.full((output_size, output_size), 255, np.uint8), False
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    border = max(1, round(min(h, w) * 0.015))
    mask[:border, :] = cv2.GC_BGD; mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD; mask[:, -border:] = cv2.GC_BGD
    cv2.ellipse(mask, (w//2, h//2), (max(5, int(w*0.15)), max(5, int(h*0.22))), 0, 0, 360, cv2.GC_FGD, -1)
    bg_model = np.zeros((1, 65), np.float64); fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bg_model, fg_model, iterCount=3, mode=cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.full((output_size, output_size), 255, np.uint8), False
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    frac = float(fg.mean())
    if not 0.08 <= frac <= 0.92:
        return np.full((output_size, output_size), 255, np.uint8), False
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (0,0), sigmaX=1.0)
    return np.round(np.clip(alpha,0,1)*255).astype(np.uint8), True


def make_random_background_batch(batch_size,height,width,device):
    palette=torch.as_tensor(BACKGROUND_PALETTE,device=device,dtype=torch.float32)/255.0
    c1=palette[torch.randint(0,len(BACKGROUND_PALETTE),(batch_size,),device=device)]
    c2=palette[torch.randint(0,len(BACKGROUND_PALETTE),(batch_size,),device=device)]
    solid=c1[:,:,None,None].expand(-1,-1,height,width)
    vh=torch.linspace(0,1,height,device=device).view(1,1,height,1)
    vw=torch.linspace(0,1,width,device=device).view(1,1,1,width)
    vertical=c1[:,:,None,None]*(1-vh)+c2[:,:,None,None]*vh
    horizontal=c1[:,:,None,None]*(1-vw)+c2[:,:,None,None]*vw
    choose_v=torch.rand(batch_size,device=device)<0.5
    gradient=torch.where(choose_v[:,None,None,None],vertical.expand(-1,-1,height,width),horizontal.expand(-1,-1,height,width))
    low=F.interpolate(c1[:,:,None,None]+torch.randn((batch_size,3,max(4,height//18),max(4,width//18)),device=device)*0.04,size=(height,width),mode="bilinear",align_corners=False)
    texture=F.avg_pool2d(low,kernel_size=3,stride=1,padding=1)
    style=torch.randint(0,3,(batch_size,),device=device)
    return torch.where((style==0)[:,None,None,None],solid,torch.where((style==1)[:,None,None,None],gradient,texture)).clamp(0,1)



def rgb_to_hsv_batch(x):
    """Convert float RGB [B,3,H,W] in [0,1] to hue, saturation, value tensors."""
    maxc, maxidx = x.max(dim=1)
    minc = x.min(dim=1).values
    delta = maxc - minc
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    hue = torch.zeros_like(maxc)
    nz = delta > 1e-6
    mask = (maxidx == 0) & nz
    hue[mask] = ((g - b)[mask] / delta[mask]) % 6.0
    mask = (maxidx == 1) & nz
    hue[mask] = ((b - r)[mask] / delta[mask]) + 2.0
    mask = (maxidx == 2) & nz
    hue[mask] = ((r - g)[mask] / delta[mask]) + 4.0
    hue = (hue / 6.0) % 1.0
    saturation = torch.where(maxc > 1e-6, delta / maxc.clamp_min(1e-6), torch.zeros_like(maxc))
    return hue, saturation, maxc


def hsv_to_rgb_batch(h, s, v):
    """Convert hue/saturation/value tensors [B,H,W] to float RGB [B,3,H,W]."""
    h6 = (h * 6.0) % 6.0
    idx = torch.floor(h6).long() % 6
    f = h6 - torch.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    choices = torch.stack([
        torch.stack([v, t, p], dim=1),
        torch.stack([q, v, p], dim=1),
        torch.stack([p, v, t], dim=1),
        torch.stack([p, q, v], dim=1),
        torch.stack([t, p, v], dim=1),
        torch.stack([v, p, q], dim=1),
    ], dim=1)  # [B,6,3,H,W]
    gather_idx = idx[:, None, None, :, :].expand(-1, 1, 3, -1, -1)
    return choices.gather(1, gather_idx).squeeze(1).clamp(0.0, 1.0)

def augment_batch_gpu(images_uint8,config,*,device,background_alpha_uint8=None):
    """Apply independently sampled per-image transforms using config['p_*'] probabilities."""
    validate_probability_config(config)
    device=torch.device(device)
    if images_uint8.ndim!=4 or images_uint8.shape[-1]!=3: raise ValueError(f"Expected [B,H,W,3], got {tuple(images_uint8.shape)}")
    x=images_uint8.to(device=device,non_blocking=True).permute(0,3,1,2).contiguous().float().div_(255.0)
    b,_,h,w=x.shape

    # Background replacement is composited before geometry, using a train-only cached mask.
    if config.get("p_background",0.0)>0:
        if background_alpha_uint8 is None: raise ValueError("Background probability > 0 requires cached training masks.")
        alpha=background_alpha_uint8.to(device=device,non_blocking=True).float().unsqueeze(1).div_(255.0)
        alpha=F.interpolate(alpha,size=(h,w),mode="bilinear",align_corners=False).clamp_(0,1)
        bg=make_random_background_batch(b,h,w,device)
        composite=x*alpha+bg*(1-alpha)
        selected=(torch.rand(b,device=device)<config["p_background"])
        x=torch.where(selected[:,None,None,None],composite,x)

    # Flips and rotation probabilities are controlled independently.
    p_flip=float(config.get("p_flip",0.0))
    if p_flip>0:
        flip_h=torch.rand(b,device=device)<p_flip; flip_v=torch.rand(b,device=device)<p_flip
        x=torch.where(flip_h[:,None,None,None],x.flip(-1),x)
        x=torch.where(flip_v[:,None,None,None],x.flip(-2),x)
    p_rot=float(config.get("p_rotation",0.0))
    if p_rot>0:
        apply_rot=torch.rand(b,device=device)<p_rot
        angle=(torch.rand(b,device=device)*2-1)*math.radians(ROTATION_DEGREES)
        angle=torch.where(apply_rot,angle,torch.zeros_like(angle))
        co,si=torch.cos(angle),torch.sin(angle)
        theta=torch.zeros((b,2,3),device=device,dtype=x.dtype)
        theta[:,0,0]=co; theta[:,0,1]=-si; theta[:,1,0]=si; theta[:,1,1]=co
        grid=F.affine_grid(theta,x.size(),align_corners=False)
        x=F.grid_sample(x,grid,mode="bilinear",padding_mode="border",align_corners=False)

    p_colour=float(config.get("p_colour",0.0))
    if p_colour>0:
        apply=(torch.rand(b,device=device)<p_colour)[:,None,None,None]
        brightness=torch.empty((b,1,1,1),device=device).uniform_(*BRIGHTNESS_RANGE)
        changed=x*brightness
        gray=0.2989*changed[:,0:1]+0.5870*changed[:,1:2]+0.1140*changed[:,2:3]
        gray_mean=gray.mean((2,3),keepdim=True)
        contrast=torch.empty((b,1,1,1),device=device).uniform_(*CONTRAST_RANGE)
        changed=(changed-gray_mean)*contrast+gray_mean
        gray=0.2989*changed[:,0:1]+0.5870*changed[:,1:2]+0.1140*changed[:,2:3]
        saturation=torch.empty((b,1,1,1),device=device).uniform_(*SATURATION_RANGE)
        changed=gray+(changed-gray)*saturation
        # Small hue jitter is also part of colour jitter. Darkening is represented by
        # brightness factors below 1.0 in BRIGHTNESS_RANGE.
        hh, ss, vv = rgb_to_hsv_batch(changed.clamp(0,1))
        hue_shift=torch.empty((b,1,1),device=device).uniform_(-HUE_STRENGTH,HUE_STRENGTH)
        changed=hsv_to_rgb_batch((hh+hue_shift)%1.0,ss,vv)
        x=torch.where(apply,changed.clamp(0,1),x)

    p_blur=float(config.get("p_blur",0.0))
    if p_blur>0:
        k=torch.tensor([1,4,6,4,1],device=device,dtype=x.dtype); k=k/k.sum()
        kernel=(k[:,None]*k[None,:]).view(1,1,5,5).expand(3,1,5,5).contiguous()
        blurred=F.conv2d(x,kernel,padding=2,groups=3)
        apply=(torch.rand(b,device=device)<p_blur)[:,None,None,None]
        x=torch.where(apply,blurred,x)

    p_noise=float(config.get("p_noise",0.0))
    if p_noise>0:
        apply=(torch.rand(b,device=device)<p_noise)[:,None,None,None]
        noisy=(x+torch.randn_like(x)*SENSOR_NOISE_STD).clamp(0,1)
        x=torch.where(apply,noisy,x)

    p_cut=float(config.get("p_cutout",0.0))
    if p_cut>0:
        chosen=torch.rand(b,device=device)<p_cut
        area=torch.empty(b,device=device).uniform_(CUTOUT_MIN_AREA,CUTOUT_MAX_AREA)
        ratio=torch.empty(b,device=device).uniform_(math.log(0.5),math.log(2.0)).exp()
        ph=torch.sqrt(area*h*w*ratio).floor().long().clamp_(1,h)
        pw=torch.sqrt(area*h*w/ratio).floor().long().clamp_(1,w)
        top=(torch.rand(b,device=device)*(h-ph+1)).floor().long()
        left=(torch.rand(b,device=device)*(w-pw+1)).floor().long()
        yy=torch.arange(h,device=device).view(1,h,1); xx=torch.arange(w,device=device).view(1,1,w)
        mask=chosen[:,None,None]&(yy>=top[:,None,None])&(yy<(top+ph)[:,None,None])&(xx>=left[:,None,None])&(xx<(left+pw)[:,None,None])
        fill=x.mean((2,3),keepdim=True)
        x=torch.where(mask[:,None,:,:],fill,x)

    mean=torch.as_tensor(IMAGENET_MEAN,device=device,dtype=x.dtype).view(1,3,1,1)
    std=torch.as_tensor(IMAGENET_STD,device=device,dtype=x.dtype).view(1,3,1,1)
    return (x-mean)/std
