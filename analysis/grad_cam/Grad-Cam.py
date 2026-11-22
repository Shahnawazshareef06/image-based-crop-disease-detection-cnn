

import os
import sys
import io
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# Set environment variable to prevent OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import (
    resnet50,
    mobilenet_v3_large,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
)

# Configuration & Paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = r"C:\Users\harsh\OneDrive\Desktop\MML Project\FineTuned_GradCAM_Analysis"
IMAGE_ZIP_PATH = r"C:\Users\harsh\Downloads\plantvillage_color_deduplicated.zip"
PROJECT_ZIP_PATH = r"C:\Users\harsh\OneDrive\Desktop\MML Project\image-based-crop-disease-detection-cnn-main.zip"
FC_CKPT_ZIP = r"C:\Users\harsh\OneDrive\Desktop\MML Project\repo\image-based-crop-disease-detection-cnn-main\weekly report\Week 4\frozen_p010_results.zip"

NUM_CLASSES = 38
INPUT_SIZE = 224

# Standard PyTorch normalization
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class GradCAM:
    """Computes Grad-CAM activations for a target layer in a CNN model."""
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

        # Global average pooling of gradients -> weights alpha
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_act = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(weighted_act)

        # Min-max normalization
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input dimensions
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), logits.detach().cpu().numpy()


def load_models():
    """Builds and loads ResNet-50 and MobileNetV3 Large with 38-class diagnostic heads."""
    print("Loading ResNet-50 model...")
    res_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    res_model.fc = nn.Linear(2048, NUM_CLASSES)

    # Load trained classifier head from checkpoint if available
    if os.path.exists(FC_CKPT_ZIP):
        try:
            with zipfile.ZipFile(FC_CKPT_ZIP, 'r') as z:
                ckpt_data = z.read('frozen_resnet50_runs/uniform_p010_seed42/best_fc_checkpoint.pt')
                ckpt = torch.load(io.BytesIO(ckpt_data), map_location='cpu')
                res_model.fc.load_state_dict(ckpt['fc_state_dict'])
                print(" -> Successfully loaded trained ResNet-50 38-class classifier head!")
        except Exception as e:
            print(f" -> Notice: using initialized linear head ({e})")

    res_model = res_model.to(DEVICE).eval()

    print("Loading MobileNetV3 Large model...")
    mb_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    mb_model.classifier[3] = nn.Linear(mb_model.classifier[3].in_features, NUM_CLASSES)
    mb_model = mb_model.to(DEVICE).eval()

    return res_model, mb_model


def get_fixed_cohort():
    """
    Selects 45 fixed images across 3 diagnostic groups:
    - Group 1: 15 Correct Predictions
    - Group 2: 15 Weakest Classes / Errors (Corn Rust, Gray Leaf Spot, Blights, Target Spot)
    - Group 3: 15 Field / Natural Context images
    """
    cohort = [
        # ==========================================
        # GROUP 1: CORRECT PREDICTIONS (15 images)
        # ==========================================
        {"group": "Group 1: Correct", "true_class": "Tomato___healthy", "subpath": "Tomato___healthy/0001460f-650a-41b5-ac4b-93e0076a5249___GH_HL Leaf 259.1.JPG"},
        {"group": "Group 1: Correct", "true_class": "Tomato___healthy", "subpath": "Tomato___healthy/00bce06e-9608-498 concrete..."},
        {"group": "Group 1: Correct", "true_class": "Corn_(maize)___healthy", "subpath": "Corn_(maize)___healthy/00031d81-ab1c-4b5d-9b16-56a84976451e___R.S_HL 7956.JPG"},
        {"group": "Group 1: Correct", "true_class": "Apple___Apple_scab", "subpath": "Apple___Apple_scab/00075aa8-d81a-4e84-a43d-51a6574f88b5___FREC_Scab 3335.JPG"},
        {"group": "Group 1: Correct", "true_class": "Apple___Black_rot", "subpath": "Apple___Black_rot/00e909aa-e3ae-4558-9961-336bb7247ba3___JR_FrScab.S 8593.JPG"},
        {"group": "Group 1: Correct", "true_class": "Potato___healthy", "subpath": "Potato___healthy/00fc8eac-fbef-4c7f-8948-80138c2ac710___RS_HL 1864.JPG"},
        {"group": "Group 1: Correct", "true_class": "Potato___Late_blight", "subpath": "Potato___Late_blight/006955edd848.JPG"},
        {"group": "Group 1: Correct", "true_class": "Grape___Black_rot", "subpath": "Grape___Black_rot/00e47087-0b16-43b6-9936-e8d122bf773a___FAM_B.Rot 3169.JPG"},
        {"group": "Group 1: Correct", "true_class": "Grape___Esca_(Black_Measles)", "subpath": "Grape___Esca_(Black_Measles)/01162551-b800-4df8-ac46-64d17d6ac5bf___FAM_B.Msls 1153.JPG"},
        {"group": "Group 1: Correct", "true_class": "Peach___Bacterial_spot", "subpath": "Peach___Bacterial_spot/00f96894-3990-48e0-bb12-9276d1e44f1e___Rutg._Bact.S 1344.JPG"},
        {"group": "Group 1: Correct", "true_class": "Pepper,_bell___Bacterial_spot", "subpath": "Pepper,_bell___Bacterial_spot/00f2e69a-1e56-4165-b4d3-8d64226d9d8d___JR_B.Spot 3132.JPG"},
        {"group": "Group 1: Correct", "true_class": "Soybean___healthy", "subpath": "Soybean___healthy/00161616-e60e-4e01-8946-b883416ab485___RS_HL 4413.JPG"},
        {"group": "Group 1: Correct", "true_class": "Strawberry___healthy", "subpath": "Strawberry___healthy/00e84b86-04ab-48d8-9c59-d88681533ad0___RS_HL 2095.JPG"},
        {"group": "Group 1: Correct", "true_class": "Tomato___Bacterial_spot", "subpath": "Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___UF.GRC_BS_Lab Leaf 0498.JPG"},
        {"group": "Group 1: Correct", "true_class": "Tomato___Target_Spot", "subpath": "Tomato___Target_Spot/00222384-93eb-46b5-b79e-e09265103a89___Com.G_TgS_FL 8153.JPG"},

        # ==========================================
        # GROUP 2: WEAKEST CLASSES & ERRORS (15 images)
        # ==========================================
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Common_rust_", "subpath": "Corn_(maize)___Common_rust_/002163b7-7d88-4e89-a212-073c683b5f36___RS_Rust 1665.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Common_rust_", "subpath": "Corn_(maize)___Common_rust_/00105312-3da4-42c1-93c4-2d07b82c6dd1___RS_Rust 1640.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Common_rust_", "subpath": "Corn_(maize)___Common_rust_/RS_Rust 1564.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "subpath": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/cfbb5e1c-2812-4467-9748-ee171462c331___RS_GLSp 4371.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "subpath": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/00120658-4560-4b68-b80c-03fb0cfcb3b5___RS_GLSp 4374.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Northern_Leaf_Blight", "subpath": "Corn_(maize)___Northern_Leaf_Blight/0073577d-8153-4dc9-982c-473d09a250fa___RS_NLB 3848.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Corn_(maize)___Northern_Leaf_Blight", "subpath": "Corn_(maize)___Northern_Leaf_Blight/0003157b-83c3-43fe-a887-f834a3eb8752___RS_NLB 3632.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Early_blight", "subpath": "Tomato___Early_blight/b58d7a66-097f-4167-abae-92f865407d90___RS_Erly.B 7845.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Early_blight", "subpath": "Tomato___Early_blight/b24f8f98-4bfe-4605-85f5-738fc09e58fe___RS_Erly.B 6406.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Early_blight", "subpath": "Tomato___Early_blight/f8374b63-016a-4b76-9ca8-df348cb44564___RS_Erly.B 7688.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Target_Spot", "subpath": "Tomato___Target_Spot/e475a5f4-8166-4b18-9f29-8a7ccb541bbb___Com.G_TgS_FL 9878.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Target_Spot", "subpath": "Tomato___Target_Spot/05991e8b-e178-450f-8da9-d7c10bd2cda9___Com.G_TgS_FL 9731.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Spider_mites Two-spotted_spider_mite", "subpath": "Tomato___Spider_mites Two-spotted_spider_mite/0a7714a4-530b-49e0-821f-be8d33c588ac___Com.G_SpM_FL 1775.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Spider_mites Two-spotted_spider_mite", "subpath": "Tomato___Spider_mites Two-spotted_spider_mite/0004be52-4752-43ce-b391-7f9754f76239___Com.G_SpM_FL 8945.JPG"},
        {"group": "Group 2: Weakest Classes", "true_class": "Tomato___Bacterial_spot", "subpath": "Tomato___Bacterial_spot/8833a1e2-425e-4902-9883-2727ad858d23___UF.GRC_BS_Lab Leaf 9103.JPG"},

        # ==========================================
        # GROUP 3: FIELD / NATURAL CONTEXT (15 images)
        # ==========================================
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Late_blight", "subpath": "Tomato___Late_blight/00ce4248-89c8-454e-add6-83441cf5e688___GHLB2 Leaf 9037.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Late_blight", "subpath": "Tomato___Late_blight/002f20ab-f052-4a0b-a010-863a3c267270___GHLB2 Leaf 90.1.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Septoria_leaf_spot", "subpath": "Tomato___Septoria_leaf_spot/000c79c8-724d-48fd-a0c3-f09c69d80d09___JR_Sept.L.S 8436.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Septoria_leaf_spot", "subpath": "Tomato___Septoria_leaf_spot/002e2124-b1ec-4e00-843a-23ff7a7d4cf4___JR_Sept.L.S 8597.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "subpath": "Tomato___Tomato_Yellow_Leaf_Curl_Virus/00a74797-2ff9-4efd-b94f-4a30e8c603fc___YLCV_GCREC 2552.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "subpath": "Tomato___Tomato_Yellow_Leaf_Curl_Virus/00f13d80-5a9f-4315-9430-c3d5e2365261___YLCV_GCREC 2786.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Tomato___Tomato_mosaic_virus", "subpath": "Tomato___Tomato_mosaic_virus/000ec694-0152-4751-8669-e587ad045a19___GH_HL Leaf 366.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Squash___Powdery_mildew", "subpath": "Squash___Powdery_mildew/0034a755-e9d6-4443-85bb-264d1502fa62___UM_Powd.M 0517.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Squash___Powdery_mildew", "subpath": "Squash___Powdery_mildew/00e8c187-ec12-4c60-84a2-19e49a1fbe7e___UM_Powd.M 0407.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Corn_(maize)___healthy", "subpath": "Corn_(maize)___healthy/011f0a5d-b0a7-4c74-a690-3ae8e68db30d___R.S_HL 8094.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Apple___Cedar_apple_rust", "subpath": "Apple___Cedar_apple_rust/005e8188-66a9-4673-9a3b-28564a2754fe___FREC_C.Rust 3747.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Apple___healthy", "subpath": "Apple___healthy/00f4ba19-e00e-4583-a5a8-3cb34177b966___RS_HL 7935.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Cherry_(including_sour)___Powdery_mildew", "subpath": "Cherry_(including_sour)___Powdery_mildew/00c73229-2364-4e94-8149-d7b6eb4313f8___FREC_Pwd.M 4887.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Cherry_(including_sour)___healthy", "subpath": "Cherry_(including_sour)___healthy/00e5f037-f8cc-4d3f-b3a1-77884587a8b6___JR_HL 9766.JPG"},
        {"group": "Group 3: Field / Natural Context", "true_class": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "subpath": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/0007f3fc-4b36-4074-b5a5-c3c6f44e8c3f___FAM_L.Blight 1460.JPG"}
    ]
    return cohort


def load_image_from_zip(img_zip, subpath, index_map):
    """Loads and returns PIL RGB image from zip using flexible path lookup."""
    clean = subpath.replace('\\', '/').strip().lower()
    base = os.path.basename(clean)

    # 1. Exact match
    if clean in index_map:
        return Image.open(io.BytesIO(img_zip.read(index_map[clean]))).convert('RGB')
    
    # 2. Base filename match
    for k, v in index_map.items():
        if k.endswith('/' + base) or k == base:
            return Image.open(io.BytesIO(img_zip.read(v))).convert('RGB')

    # 3. Class folder partial search
    cls_prefix = clean.split('/')[0] if '/' in clean else ""
    for k, v in index_map.items():
        if cls_prefix and k.startswith(cls_prefix):
            return Image.open(io.BytesIO(img_zip.read(v))).convert('RGB')

    return None


def assess_attention(cam, img_pil):
    """
    Quantifies whether Grad-CAM peak attention falls on the lesion/leaf or background.
    Returns:
    - verdict: 'Lesion-Focused' | 'Background-Focused' | 'Mixed'
    - attribution_ratio: float (proportion of energy on foreground leaf)
    """
    img_arr = np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE)))
    
    # Simple foreground leaf segmentation:
    # Leaf tissue in PlantVillage has green/yellow/brown pigmentation.
    # Background is neutral gray/black (R ≈ G ≈ B, or low saturation).
    r, g, b = img_arr[:,:,0].astype(float), img_arr[:,:,1].astype(float), img_arr[:,:,2].astype(float)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    is_leaf = (color_diff > 18) & ((g > 40) | (r > 40))

    # Mask of high CAM attention (>= 50% of maximum heat)
    high_cam = cam >= 0.50
    total_high_cam = high_cam.sum()

    if total_high_cam == 0:
        return "Mixed", 0.50

    # Proportion of high attention located on the leaf / lesion tissue
    leaf_cam_overlap = (high_cam & is_leaf).sum()
    attribution_ratio = float(leaf_cam_overlap / total_high_cam)

    # Find coordinates of peak CAM activation
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    peak_on_leaf = is_leaf[peak_y, peak_x]

    if attribution_ratio >= 0.70 and peak_on_leaf:
        verdict = "Lesion-Focused"
    elif attribution_ratio < 0.40 or not peak_on_leaf:
        verdict = "Background-Focused"
    else:
        verdict = "Mixed"

    return verdict, round(attribution_ratio, 3)


def overlay_cam_on_image(img_pil, cam, alpha=0.55):
    """Overlays heatmap on original image using Jet colormap."""
    img_resized = img_pil.resize((INPUT_SIZE, INPUT_SIZE))
    heatmap = cm.jet(cam)[:, :, :3]  # Drop alpha channel
    heatmap = np.uint8(255 * heatmap)
    
    img_np = np.array(img_resized)
    blend = np.uint8(img_np * (1 - alpha) + heatmap * alpha)
    return blend


def render_group_panel(results_list, group_title, save_path):
    """Generates a high-resolution 15-sample visual grid for a diagnostic group."""
    n_samples = len(results_list)
    cols = 3  # Columns per sample: [Original, ResNet-50 CAM, MobileNetV3 CAM]
    rows = min(n_samples, 5)  # Render up to 5 representative samples per visual panel

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(rows):
        res = results_list[i]
        orig_img = res['orig_img']
        res_blend = res['res_blend']
        mb_blend = res['mb_blend']

        # Format clean label
        c_name = res['true_class'].split('___')[-1].replace('_', ' ')
        crop = res['true_class'].split('___')[0]

        # Column 1: Original Image
        axes[i, 0].imshow(orig_img.resize((INPUT_SIZE, INPUT_SIZE)))
        axes[i, 0].set_title(f"Sample {i+1}: {crop}\nTrue: {c_name}", fontsize=9, fontweight='bold', pad=4)
        axes[i, 0].axis('off')

        # Column 2: ResNet-50 CAM
        res_v = res['res_verdict']
        res_col = '#2ca02c' if res_v == 'Lesion-Focused' else ('#d62728' if res_v == 'Background-Focused' else '#ff7f0e')
        axes[i, 1].imshow(res_blend)
        axes[i, 1].set_title(f"ResNet-50 CAM\nVerdict: {res_v} ({res['res_ratio']:.0%})", fontsize=9, color=res_col, fontweight='bold', pad=4)
        axes[i, 1].axis('off')

        # Column 3: MobileNetV3 CAM
        mb_v = res['mb_verdict']
        mb_col = '#2ca02c' if mb_v == 'Lesion-Focused' else ('#d62728' if mb_v == 'Background-Focused' else '#ff7f0e')
        axes[i, 2].imshow(mb_blend)
        axes[i, 2].set_title(f"MobileNetV3 CAM\nVerdict: {mb_v} ({res['mb_ratio']:.0%})", fontsize=9, color=mb_col, fontweight='bold', pad=4)
        axes[i, 2].axis('off')

    plt.suptitle(f"Grad-CAM Heatmap Analysis: {group_title}", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f" -> Saved visual panel: {os.path.basename(save_path)}")


def plot_summary_barchart(df_results, save_path):
    """Plots comparative breakdown of Lesion vs Background attention across groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    groups = df_results['group'].unique()
    categories = ['Lesion-Focused', 'Mixed', 'Background-Focused']
    colors = ['#2ca02c', '#ff7f0e', '#d62728']

    # ResNet-50 counts
    res_counts = df_results.groupby(['group', 'res_verdict']).size().unstack(fill_value=0)
    for c in categories:
        if c not in res_counts.columns: res_counts[c] = 0
    res_counts = res_counts[categories]

    # MobileNetV3 counts
    mb_counts = df_results.groupby(['group', 'mb_verdict']).size().unstack(fill_value=0)
    for c in categories:
        if c not in mb_counts.columns: mb_counts[c] = 0
    mb_counts = mb_counts[categories]

    # Bar chart 1: ResNet-50
    res_counts.plot(kind='bar', stacked=True, ax=ax1, color=colors, alpha=0.9, width=0.55)
    ax1.set_title("ResNet-50: Attention by Group", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Number of Images", fontsize=11, fontweight='bold')
    ax1.set_xticklabels([g.split(':')[0] for g in groups], rotation=0, fontsize=10)
    ax1.set_xlabel("")
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_ylim(0, 18)

    # Bar chart 2: MobileNetV3
    mb_counts.plot(kind='bar', stacked=True, ax=ax2, color=colors, alpha=0.9, width=0.55)
    ax2.set_title("MobileNetV3 Large: Attention by Group", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Number of Images", fontsize=11, fontweight='bold')
    ax2.set_xticklabels([g.split(':')[0] for g in groups], rotation=0, fontsize=10)
    ax2.set_xlabel("")
    ax2.legend(loc='upper right', frameon=True)
    ax2.set_ylim(0, 18)

    plt.suptitle("Grad-CAM Verdict Summary: Lesion vs. Background Attention", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved summary chart: {os.path.basename(save_path)}")


