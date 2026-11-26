
import os
import sys
import io
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# Prevent OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import (
    resnet50,
    mobilenet_v3_large,
)

INPUT_SIZE = 224
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TERRAVISTA_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


class GradCAM:
    """Computes Grad-CAM activation maps for a specified convolutional layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, outp):
        self.activations = outp.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class_idx):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[0, target_class_idx]
        score.backward()

        # Channel-wise weight alpha: GAP of gradients
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_act = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(weighted_act)

        # Min-max normalization
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Bilinear upsample to input resolution (224x224)
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), logits.detach().cpu().numpy()


def load_fine_tuned_models(res_ckpt_path, mb_ckpt_path):
    """
    Loads ResNet-50 and MobileNetV3 Large using their EXACT fine-tuned checkpoint weights.
    """
    print(f"Loading Fine-Tuned ResNet-50 checkpoint from:\n -> {res_ckpt_path}")
    if not os.path.exists(res_ckpt_path):
        raise FileNotFoundError(f"ResNet-50 checkpoint not found at: {res_ckpt_path}")
    
    res_model = resnet50(weights=None)
    res_model.fc = nn.Linear(2048, NUM_CLASSES)
    res_raw = torch.load(res_ckpt_path, map_location='cpu')
    res_state = res_raw['model_state_dict'] if 'model_state_dict' in res_raw else res_raw
    clean_res_state = {k.replace('module.', ''): v for k, v in res_state.items()}
    res_model.load_state_dict(clean_res_state)
    res_model = res_model.to(DEVICE).eval()
    print(" -> ResNet-50 fine-tuned weights successfully loaded into memory!")

    print(f"Loading Fine-Tuned MobileNetV3 Large checkpoint from:\n -> {mb_ckpt_path}")
    if not os.path.exists(mb_ckpt_path):
        raise FileNotFoundError(f"MobileNetV3 checkpoint not found at: {mb_ckpt_path}")

    mb_model = mobilenet_v3_large(weights=None)
    mb_model.classifier[3] = nn.Linear(mb_model.classifier[3].in_features, NUM_CLASSES)
    mb_raw = torch.load(mb_ckpt_path, map_location='cpu')
    mb_state = mb_raw['model_state_dict'] if (isinstance(mb_raw, dict) and 'model_state_dict' in mb_raw) else mb_raw
    clean_mb_state = {k.replace('module.', ''): v for k, v in mb_state.items()}
    mb_model.load_state_dict(clean_mb_state)
    mb_model = mb_model.to(DEVICE).eval()
    print(" -> MobileNetV3 Large fine-tuned weights successfully loaded into memory!")

    return res_model, mb_model


def assess_attention(cam, img_pil, is_field=False):
    """
    Quantifies whether Grad-CAM peak attention falls on the genuine disease lesion
    or drifts to non-causal background (soil, sky, hands, clean backdrop).
    """
    img_arr = np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE)), dtype=float)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    gray = np.mean(img_arr, axis=2)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)

    if not is_field:
        # Laboratory background: neutral gray/black backdrop
        bg_mask = (color_diff < 18) | (gray < 30) | (gray > 225)
        is_lesion = ~bg_mask
    else:
        # In-the-wild field: background comprises soil, solar glare, and sky
        soil = (r > g + 15) & (r > b + 15)
        glare = (r > 200) & (g > 200) & (b > 200)
        healthy_green = (g > r * 1.15) & (g > b * 1.15)
        bg_mask = soil | glare | (gray < 30)
        is_lesion = (color_diff > 18) & ~bg_mask & ~healthy_green

    high_cam = cam >= 0.50
    total_high_cam = high_cam.sum() + 1e-7

    lesion_ratio = float((high_cam & is_lesion).sum() / total_high_cam)
    bg_ratio = float((high_cam & bg_mask).sum() / total_high_cam)

    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    peak_on_lesion = bool(is_lesion[py, px])
    peak_on_bg = bool(bg_mask[py, px])

    if not is_field:
        if lesion_ratio >= 0.70 and peak_on_lesion:
            verdict = "Lesion-Focused"
        elif peak_on_bg or lesion_ratio < 0.35:
            verdict = "Background-Focused"
        else:
            verdict = "Mixed / Context"
    else:
        if lesion_ratio >= 0.50 and peak_on_lesion:
            verdict = "Lesion-Focused"
        elif peak_on_bg or lesion_ratio < 0.25:
            verdict = "Background-Focused"
        else:
            verdict = "Mixed / Non-Lesion"

    return verdict, lesion_ratio


def build_cohort(pv_zip_path, plantdoc_dir):
    """
    Constructs the 45 strictly unique images across the 3 cohorts:
    - 15 Group 1 (Correct, PlantVillage)
    - 15 Group 2 (Weakest classes, PlantVillage)
    - 15 Group 3 (Real-world PlantDoc field images)
    """
    cohort = []

    with zipfile.ZipFile(pv_zip_path, 'r') as z:
        all_pv_files = [n for n in z.namelist() if not n.endswith('/')]

    # Group 1: 15 distinct classes (Correct)
    g1_classes = [
        ('apple___apple_scab', 'Apple___Apple_scab'),
        ('apple___cedar_apple_rust', 'Apple___Cedar_apple_rust'),
        ('cherry_(including_sour)___healthy', 'Cherry_(including_sour)___healthy'),
        ('corn_(maize)___healthy', 'Corn_(maize)___healthy'),
        ('grape___black_rot', 'Grape___Black_rot'),
        ('grape___esca_(black_measles)', 'Grape___Esca_(Black_Measles)'),
        ('peach___bacterial_spot', 'Peach___Bacterial_spot'),
        ('pepper,_bell___bacterial_spot', 'Pepper,_bell___Bacterial_spot'),
        ('potato___early_blight', 'Potato___Early_blight'),
        ('potato___late_blight', 'Potato___Late_blight'),
        ('soybean___healthy', 'Soybean___healthy'),
        ('strawberry___healthy', 'Strawberry___healthy'),
        ('tomato___bacterial_spot', 'Tomato___Bacterial_spot'),
        ('tomato___septoria_leaf_spot', 'Tomato___Septoria_leaf_spot'),
        ('tomato___target_spot', 'Tomato___Target_Spot')
    ]
    for c_prefix, tv_name in g1_classes:
        matching = [n for n in all_pv_files if n.startswith(c_prefix + '/')]
        cohort.append({
            'cohort': 'Group 1: Correct',
            'dataset': 'PlantVillage',
            'true_class': tv_name,
            'source_path': matching[0],
            'is_plantdoc': False
        })

    # Group 2: 15 distinct images from weakest error classes
    g2_classes = [
        ('corn_(maize)___cercospora_leaf_spot gray_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 3),
        ('corn_(maize)___northern_leaf_blight', 'Corn_(maize)___Northern_Leaf_Blight', 3),
        ('corn_(maize)___common_rust_', 'Corn_(maize)___Common_rust_', 3),
        ('tomato___early_blight', 'Tomato___Early_blight', 3),
        ('tomato___target_spot', 'Tomato___Target_Spot', 3),
    ]
    for c_prefix, tv_name, count in g2_classes:
        matching = [n for n in all_pv_files if n.startswith(c_prefix + '/')]
        for f in matching[1:1+count]:
            cohort.append({
                'cohort': 'Group 2: Weakest Classes',
                'dataset': 'PlantVillage',
                'true_class': tv_name,
                'source_path': f,
                'is_plantdoc': False
            })

    # Group 3: 15 distinct images from PlantDoc field test images
    plantdoc_files = sorted([f for f in os.listdir(plantdoc_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:15]
    for fname in plantdoc_files:
        parts = fname.split('__')
        tv_name = parts[0]
        cohort.append({
            'cohort': 'Group 3: PlantDoc Field Test',
            'dataset': 'PlantDoc (Field Context)',
            'true_class': tv_name,
            'source_path': os.path.join(plantdoc_dir, fname),
            'is_plantdoc': True
        })

    all_paths = [x['source_path'] for x in cohort]
    assert len(cohort) == 45, f"Expected 45 images, got {len(cohort)}"
    assert len(set(all_paths)) == 45, f"Duplicate detected! Only {len(set(all_paths))} unique paths."
    print(f" -> Successfully built cohort: 45 strictly unique images verified (0 duplicates)!")
    return cohort


def render_group_panel(group_items, group_title, save_path):
    """
    Renders a 15-sample publication-grade visualization panel:
    Column 1: Original Image
    Column 2: ResNet-50 Grad-CAM Overlay
    Column 3: MobileNetV3 Grad-CAM Overlay
    """
    n = len(group_items)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 2.2))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, item in enumerate(group_items):
        img_pil = item['img_pil']
        res_cam = item['res_cam']
        mb_cam = item['mb_cam']

        # 1. Original Image
        axes[i, 0].imshow(img_pil)
        c_short = item['true_class'].replace('___', ' - ').replace('_', ' ')
        src_label = "PlantDoc Field" if item['is_plantdoc'] else "PlantVillage"
        axes[i, 0].set_title(f"#{i+1}: {c_short}\n[{src_label}]", fontsize=9, fontweight='bold', pad=3)
        axes[i, 0].axis('off')

        # 2. ResNet-50 Grad-CAM Overlay
        heatmap_res = cm.jet(res_cam)[:, :, :3]
        overlay_res = 0.5 * np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE))) / 255.0 + 0.5 * heatmap_res
        overlay_res = np.clip(overlay_res, 0, 1)
        axes[i, 1].imshow(overlay_res)
        color_res = '#08519c' if item['res_verdict'] == 'Lesion-Focused' else '#b2182b'
        axes[i, 1].set_title(
            f"ResNet-50: {item['res_verdict']} ({item['res_ratio']*100:.1f}%)\n"
            f"Pred: {item['res_pred_name'].split('___')[-1].replace('_', ' ')}",
            fontsize=8.5, color=color_res, fontweight='bold', pad=3
        )
        axes[i, 1].axis('off')

        # 3. MobileNetV3 Grad-CAM Overlay
        heatmap_mb = cm.jet(mb_cam)[:, :, :3]
        overlay_mb = 0.5 * np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE))) / 255.0 + 0.5 * heatmap_mb
        overlay_mb = np.clip(overlay_mb, 0, 1)
        axes[i, 2].imshow(overlay_mb)
        color_mb = '#08519c' if item['mb_verdict'] == 'Lesion-Focused' else '#b2182b'
        axes[i, 2].set_title(
            f"MobileNetV3: {item['mb_verdict']} ({item['mb_ratio']*100:.1f}%)\n"
            f"Pred: {item['mb_pred_name'].split('___')[-1].replace('_', ' ')}",
            fontsize=8.5, color=color_mb, fontweight='bold', pad=3
        )
        axes[i, 2].axis('off')

    plt.suptitle(group_title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" -> Saved {os.path.basename(save_path)}")


def render_summary_chart(results_df, save_path):
    """
    Renders comparative bar chart of Lesion vs. Background focus across the 3 cohorts.
    """
    groups = ["Group 1: Correct", "Group 2: Weakest Classes", "Group 3: PlantDoc Field Test"]
    short_names = ["Cohort 1\n(Correct Lab)", "Cohort 2\n(Weakest Lab)", "Cohort 3\n(PlantDoc Field)"]

    res_lesion_pct = []
    mb_lesion_pct = []

    for g in groups:
        sub = results_df[results_df['Cohort'] == g]
        res_lesion_pct.append((sub['ResNet_Verdict'] == 'Lesion-Focused').mean() * 100)
        mb_lesion_pct.append((sub['MobileNet_Verdict'] == 'Lesion-Focused').mean() * 100)

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, res_lesion_pct, width, label='ResNet-50 (Fine-Tuned)', color='#d73027', alpha=0.9)
    plt.bar(x + width/2, mb_lesion_pct, width, label='MobileNetV3 Large (Fine-Tuned)', color='#4575b4', alpha=0.9)

    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.6, label='Majority Focus Threshold (50%)')
    plt.ylabel('% of Images with Lesion-Focused Attention', fontsize=11, fontweight='bold')
    plt.title('Grad-CAM Attention Focus: Lesion vs. Background Across Cohorts\n(Evaluated with Exact Fine-Tuned Model Checkpoints)', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, short_names, fontsize=10, fontweight='bold')
    plt.ylim(0, 105)

    for i in range(len(groups)):
        plt.text(i - width/2, res_lesion_pct[i] + 2, f"{res_lesion_pct[i]:.0f}%", ha='center', fontweight='bold', fontsize=10)
        plt.text(i + width/2, mb_lesion_pct[i] + 2, f"{mb_lesion_pct[i]:.0f}%", ha='center', fontweight='bold', fontsize=10)

    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved {os.path.basename(save_path)}")


