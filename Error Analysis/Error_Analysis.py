"""
========================================================================================
Comparative Error Analysis: ResNet-50 vs MobileNetV3 (Fine-Tuned Models on PlantVillage)
========================================================================================
This single self-contained script performs an exhaustive validation error analysis:
1. Loads validation predictions for fine-tuned ResNet-50 and MobileNetV3 (Seed 42 / Multi-seed).
2. Computes the weakest classes for both models (ranked by F1-score and Recall).
3. Identifies the most confused pairs (y_true -> y_pred) for both models.
4. Performs error overlap analysis (Both wrong, Only MobileNetV3 wrong, Only ResNet-50 wrong, Both correct).
5. Directly extracts and visualizes actual example images for:
   - MobileNetV3 top confusions
   - ResNet-50 top confusions
   - Hard shared failures (both models wrong)
   - Disagreements (one model correct, the other wrong)
========================================================================================
"""

import os
import sys
import io
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Set clean styling for plots
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def parse_args():
    parser = argparse.ArgumentParser(description="Run validation error analysis on fine-tuned ResNet-50 vs MobileNetV3.")
    parser.add_argument(
        "--project-zip",
        type=str,
        default=r"C:\Users\harsh\OneDrive\Desktop\MML Project\image-based-crop-disease-detection-cnn-main.zip",
        help="Path to project zip containing saved predictions."
    )
    parser.add_argument(
        "--image-zip",
        type=str,
        default=r"C:\Users\harsh\Downloads\plantvillage_color_deduplicated.zip",
        help="Path to PlantVillage dataset zip file."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"C:\Users\harsh\OneDrive\Desktop\MML Project\FineTuned_Error_Analysis",
        help="Directory to save generated plots, CSVs, and README."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to analyze (default: 42)."
    )
    return parser.parse_args()


def load_predictions(project_zip_path, seed=42):
    """
    Extracts and merges validation predictions for MobileNetV3 and ResNet-50.
    """
    print(f"\n[1/6] Loading fine-tuned validation predictions (Seed {seed}) from {os.path.basename(project_zip_path)}...")
    if not os.path.exists(project_zip_path):
        raise FileNotFoundError(f"Project zip file not found at: {project_zip_path}")

    with zipfile.ZipFile(project_zip_path, 'r') as pz:
        # Load MobileNetV3 predictions
        mb_path = f"image-based-crop-disease-detection-cnn-main/mobilenetv3/results/mobilenetv3/mobilenetv3_large_uniform_p010_seed{seed}/val_predictions.csv"
        if mb_path not in pz.namelist():
            raise FileNotFoundError(f"MobileNetV3 prediction file not found in zip: {mb_path}")
        
        with pz.open(mb_path) as f:
            df_mb = pd.read_csv(f)
        print(f" -> MobileNetV3 predictions loaded: {len(df_mb)} rows")

        # Load ResNet-50 predictions from inner zip
        inner_res_zip = "image-based-crop-disease-detection-cnn-main/Resnet-50/terravista_finetune_p10_logs_predictions.zip"
        if inner_res_zip not in pz.namelist():
            raise FileNotFoundError(f"ResNet-50 inner zip not found: {inner_res_zip}")

        res_data = pz.read(inner_res_zip)
        with zipfile.ZipFile(io.BytesIO(res_data), 'r') as rz:
            res_path = f"runs_finetune_p10/resnet50/frac100_seed{seed}/val_predictions.csv"
            if res_path not in rz.namelist():
                raise FileNotFoundError(f"ResNet-50 prediction file not found in inner zip: {res_path}")
            with rz.open(res_path) as rf:
                df_res = pd.read_csv(rf)
        print(f" -> ResNet-50 predictions loaded: {len(df_res)} rows")

    # Standardize column naming and merge
    df_mb = df_mb.rename(columns={'true': 'true_class', 'pred': 'mb_pred'})
    df_res = df_res.rename(columns={'pred_class': 'res_pred', 'confidence': 'res_conf'})

    # Assert row alignment
    if not (df_mb['path'] == df_res['path']).all():
        print("Warning: Paths are not in identical order. Merging on 'path' and 'true_class'.")
        df = pd.merge(
            df_mb[['path', 'true_class', 'mb_pred']],
            df_res[['path', 'true_class', 'res_pred', 'res_conf']],
            on=['path', 'true_class'],
            how='inner'
        )
    else:
        df = df_mb[['path', 'true_class', 'mb_pred']].copy()
        df['res_pred'] = df_res['res_pred']
        df['res_conf'] = df_res['res_conf']

    print(f" -> Aligned dataset size: {len(df)} validation images")
    return df


def analyze_weakest_classes(df, out_dir):
    """
    Computes per-class metrics and identifies the weakest classes for both models.
    """
    print("\n[2/6] Computing per-class metrics and identifying weakest classes...")
    
    # Classification reports
    rep_mb = classification_report(df['true_class'], df['mb_pred'], output_dict=True, zero_division=0)
    rep_res = classification_report(df['true_class'], df['res_pred'], output_dict=True, zero_division=0)

    # Convert to DataFrames
    classes = [c for c in rep_mb.keys() if c not in ('accuracy', 'macro avg', 'weighted avg')]
    
    mb_stats = []
    for c in classes:
        mb_stats.append({
            'class': c,
            'precision': rep_mb[c]['precision'],
            'recall': rep_mb[c]['recall'],
            'f1_score': rep_mb[c]['f1-score'],
            'support': int(rep_mb[c]['support'])
        })
    df_mb_cls = pd.DataFrame(mb_stats).sort_values(by='f1_score', ascending=True)

    res_stats = []
    for c in classes:
        res_stats.append({
            'class': c,
            'precision': rep_res[c]['precision'],
            'recall': rep_res[c]['recall'],
            'f1_score': rep_res[c]['f1-score'],
            'support': int(rep_res[c]['support'])
        })
    df_res_cls = pd.DataFrame(res_stats).sort_values(by='f1_score', ascending=True)

    # Save to CSV
    df_mb_cls.to_csv(os.path.join(out_dir, "mobilenetv3_weakest_classes.csv"), index=False)
    df_res_cls.to_csv(os.path.join(out_dir, "resnet50_weakest_classes.csv"), index=False)

    print("\n--- Top 5 Weakest Classes: MobileNetV3 ---")
    print(df_mb_cls.head(5).to_string(index=False))

    print("\n--- Top 5 Weakest Classes: ResNet-50 ---")
    print(df_res_cls.head(5).to_string(index=False))

    # Plot comparison of bottom classes
    plot_weakest_classes_comparison(df_mb_cls, df_res_cls, out_dir)
    return df_mb_cls, df_res_cls


def plot_weakest_classes_comparison(df_mb_cls, df_res_cls, out_dir):
    """
    Plots a side-by-side bar chart of the weakest classes across both models.
    """
    # Select union of bottom 6 from both models
    bottom_mb = df_mb_cls.head(6)['class'].tolist()
    bottom_res = df_res_cls.head(6)['class'].tolist()
    union_weak = list(dict.fromkeys(bottom_mb + bottom_res))

    comp_list = []
    for c in union_weak:
        mb_row = df_mb_cls[df_mb_cls['class'] == c].iloc[0]
        res_row = df_res_cls[df_res_cls['class'] == c].iloc[0]
        comp_list.append({
            'Class': c.replace('___', ' - ').replace('_', ' '),
            'MobileNetV3 F1': mb_row['f1_score'],
            'ResNet-50 F1': res_row['f1_score'],
            'Avg F1': (mb_row['f1_score'] + res_row['f1_score']) / 2.0
        })

    comp_df = pd.DataFrame(comp_list).sort_values(by='Avg F1', ascending=True)

    plt.figure(figsize=(11, 6))
    y = np.arange(len(comp_df))
    height = 0.38

    plt.barh(y - height/2, comp_df['MobileNetV3 F1'], height, label='MobileNetV3 Large', color='#4575b4', alpha=0.9)
    plt.barh(y + height/2, comp_df['ResNet-50 F1'], height, label='ResNet-50', color='#d73027', alpha=0.9)

    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.yticks(y, comp_df['Class'], fontsize=9)
    plt.xlabel('F1-Score', fontsize=11, fontweight='bold')
    plt.title('Weakest Classes Comparison (F1-Score on Validation)', fontsize=13, fontweight='bold', pad=12)
    plt.xlim(0.5, 1.02)
    plt.legend(loc='lower right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weakest_classes_comparison.png"), dpi=300)
    plt.close()
    print(" -> Saved weakest_classes_comparison.png")


def analyze_most_confused_pairs(df, out_dir):
    """
    Computes and plots the most confused pairs for both models.
    """
    print("\n[3/6] Identifying most confused pairs...")

    # MobileNet errors
    mb_err = df[df['true_class'] != df['mb_pred']]
    mb_pairs = mb_err.groupby(['true_class', 'mb_pred']).size().reset_index(name='count').sort_values(by='count', ascending=False)
    mb_pairs.rename(columns={'true_class': 'True Class', 'mb_pred': 'Predicted Class'}, inplace=True)
    mb_pairs.to_csv(os.path.join(out_dir, "mobilenetv3_most_confused_pairs.csv"), index=False)

    # ResNet errors
    res_err = df[df['true_class'] != df['res_pred']]
    res_pairs = res_err.groupby(['true_class', 'res_pred']).size().reset_index(name='count').sort_values(by='count', ascending=False)
    res_pairs.rename(columns={'true_class': 'True Class', 'res_pred': 'Predicted Class'}, inplace=True)
    res_pairs.to_csv(os.path.join(out_dir, "resnet50_most_confused_pairs.csv"), index=False)

    print("\n--- Top 6 Most Confused Pairs: MobileNetV3 ---")
    print(mb_pairs.head(6).to_string(index=False))

    print("\n--- Top 6 Most Confused Pairs: ResNet-50 ---")
    print(res_pairs.head(6).to_string(index=False))

    # Plot top confused pairs
    plot_confused_pairs(mb_pairs, res_pairs, out_dir)
    return mb_pairs, res_pairs


def plot_confused_pairs(mb_pairs, res_pairs, out_dir):
    """
    Creates bar charts for top confused pairs for both models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Clean display labels
    def format_pair(row):
        t = row['True Class'].split('___')[-1].replace('_', ' ')
        p = row['Predicted Class'].split('___')[-1].replace('_', ' ')
        c_t = row['True Class'].split('___')[0].capitalize()
        c_p = row['Predicted Class'].split('___')[0].capitalize()
        if c_t == c_p:
            return f"{c_t}: {t}\n  -> Pred: {p}"
        else:
            return f"{c_t}:{t}\n  -> {c_p}:{p}"

    # MobileNet top 6
    top_mb = mb_pairs.head(6).copy()
    top_mb['label'] = top_mb.apply(format_pair, axis=1)
    ax1.barh(np.arange(len(top_mb))[::-1], top_mb['count'], color='#4575b4', alpha=0.9)
    ax1.set_yticks(np.arange(len(top_mb))[::-1])
    ax1.set_yticklabels(top_mb['label'], fontsize=9)
    ax1.set_xlabel('Error Count', fontsize=11, fontweight='bold')
    ax1.set_title('Top Confusions: MobileNetV3 Large', fontsize=12, fontweight='bold')
    for i, count in enumerate(top_mb['count']):
        ax1.text(count + 0.3, len(top_mb) - 1 - i, str(count), va='center', fontsize=9, fontweight='bold')

    # ResNet-50 top 6
    top_res = res_pairs.head(6).copy()
    top_res['label'] = top_res.apply(format_pair, axis=1)
    ax2.barh(np.arange(len(top_res))[::-1], top_res['count'], color='#d73027', alpha=0.9)
    ax2.set_yticks(np.arange(len(top_res))[::-1])
    ax2.set_yticklabels(top_res['label'], fontsize=9)
    ax2.set_xlabel('Error Count', fontsize=11, fontweight='bold')
    ax2.set_title('Top Confusions: ResNet-50', fontsize=12, fontweight='bold')
    for i, count in enumerate(top_res['count']):
        ax2.text(count + 0.15, len(top_res) - 1 - i, str(count), va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Most Confused Disease Pairs (Validation Set)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_confused_pairs_comparison.png"), dpi=300)
    plt.close()
    print(" -> Saved top_confused_pairs_comparison.png")


def analyze_error_overlap(df, out_dir):
    """
    Computes how many images both get wrong and how many only one gets wrong.
    """
    print("\n[4/6] Analyzing comparative error overlap...")

    mb_wrong = df['true_class'] != df['mb_pred']
    res_wrong = df['true_class'] != df['res_pred']
    total = len(df)

    both_wrong = (mb_wrong & res_wrong).sum()
    only_mb_wrong = (mb_wrong & ~res_wrong).sum()
    only_res_wrong = (~mb_wrong & res_wrong).sum()
    both_correct = (~mb_wrong & ~res_wrong).sum()

    total_mb_errors = mb_wrong.sum()
    total_res_errors = res_wrong.sum()

    overlap_data = [
        {"Category": "Both Correct", "Count": both_correct, "Percentage": f"{both_correct/total*100:.2f}%"},
        {"Category": "Only MobileNetV3 Wrong", "Count": only_mb_wrong, "Percentage": f"{only_mb_wrong/total*100:.2f}%"},
        {"Category": "Only ResNet-50 Wrong", "Count": only_res_wrong, "Percentage": f"{only_res_wrong/total*100:.2f}%"},
        {"Category": "Both Wrong (Shared Failures)", "Count": both_wrong, "Percentage": f"{both_wrong/total*100:.2f}%"},
        {"Category": "Total Validation Images", "Count": total, "Percentage": "100.00%"}
    ]
    overlap_df = pd.DataFrame(overlap_data)
    overlap_df.to_csv(os.path.join(out_dir, "error_overlap_summary.csv"), index=False)

    print("\n" + "="*55)
    print("       VALIDATION ERROR OVERLAP SUMMARY")
    print("="*55)
    print(f"Total Validation Images       : {total}")
    print(f"ResNet-50 Total Errors        : {total_res_errors} ({total_res_errors/total*100:.2f}%)  [Acc: {(total-total_res_errors)/total*100:.2f}%]")
    print(f"MobileNetV3 Total Errors      : {total_mb_errors} ({total_mb_errors/total*100:.2f}%)  [Acc: {(total-total_mb_errors)/total*100:.2f}%]")
    print("-" * 55)
    print(f"Both Models Wrong             : {both_wrong:4d}  ({both_wrong/total*100:.2f}%)")
    print(f"Only MobileNetV3 Wrong        : {only_mb_wrong:4d}  ({only_mb_wrong/total*100:.2f}%)  (ResNet-50 got right)")
    print(f"Only ResNet-50 Wrong          : {only_res_wrong:4d}  ({only_res_wrong/total*100:.2f}%)  (MobileNetV3 got right)")
    print(f"Both Models Correct           : {both_correct:4d}  ({both_correct/total*100:.2f}%)")
    print("="*55)

    # Save detailed lists of both wrong and disagreements
    df_both_wrong = df[mb_wrong & res_wrong].copy()
    df_both_wrong.to_csv(os.path.join(out_dir, "both_wrong_images_list.csv"), index=False)

    df_only_mb = df[mb_wrong & ~res_wrong].copy()
    df_only_mb.to_csv(os.path.join(out_dir, "only_mobilenetv3_wrong_images.csv"), index=False)

    df_only_res = df[~mb_wrong & res_wrong].copy()
    df_only_res.to_csv(os.path.join(out_dir, "only_resnet50_wrong_images.csv"), index=False)

    # Plot Error Breakdown Charts
    plot_error_breakdown(both_correct, only_mb_wrong, only_res_wrong, both_wrong,
                         total_res_errors, total_mb_errors, total, out_dir)

    return {
        'both_wrong': both_wrong,
        'only_mb_wrong': only_mb_wrong,
        'only_res_wrong': only_res_wrong,
        'both_correct': both_correct,
        'total': total,
        'total_res_errors': total_res_errors,
        'total_mb_errors': total_mb_errors
    }


def plot_error_breakdown(both_correct, only_mb_wrong, only_res_wrong, both_wrong,
                         total_res, total_mb, total, out_dir):
    """
    Plots a multi-panel visual breakdown of model errors and overlap.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Donut chart of error distribution
    labels = [
        f'Both Correct\n({both_correct})',
        f'Only MobileNetV3 Wrong\n({only_mb_wrong})',
        f'Both Wrong\n({both_wrong})',
        f'Only ResNet-50 Wrong\n({only_res_wrong})'
    ]
    sizes = [both_correct, only_mb_wrong, both_wrong, only_res_wrong]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    explode = (0.05, 0.08, 0.12, 0.08)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors, pctdistance=0.8,
        textprops={'fontsize': 9}
    )
    plt.setp(autotexts, size=9, weight="bold")
    # Draw circle for donut chart
    centre_circle = plt.Circle((0,0), 0.60, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title(f"Validation Error Overlap\nTotal Samples: {total}", fontsize=12, fontweight='bold')

    # Bar chart comparing total error breakdown
    categories = ['ResNet-50 Errors\n(96 total)', 'MobileNetV3 Errors\n(206 total)']
    shared = [both_wrong, both_wrong]
    unique = [only_res_wrong, only_mb_wrong]

    ax2.bar(categories, shared, label='Shared Errors (Both Wrong)', color='#d62728', width=0.45, alpha=0.9)
    ax2.bar(categories, unique, bottom=shared, label='Unique Errors (Only This Model Wrong)', color='#3182bd', width=0.45, alpha=0.85)

    for i, (s, u) in enumerate(zip(shared, unique)):
        ax2.text(i, s / 2, f"{s}\n(Shared)", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax2.text(i, s + u / 2, f"{u}\n(Unique)", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax2.text(i, s + u + 4, f"Total: {s+u}", ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Number of Misclassified Images', fontsize=11, fontweight='bold')
    ax2.set_title('Composition of Model Errors', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 240)
    ax2.legend(loc='upper left', frameon=True)

    plt.suptitle('Validation Error Analysis & Agreement (PlantVillage Dataset)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_breakdown_overview.png"), dpi=300)
    plt.close()
    print(" -> Saved error_breakdown_overview.png")


def load_image_from_zip(img_zip, path_str, img_index):
    """
    Looks up and opens an image from the PlantVillage zip file case-insensitively.
    """
    clean_path = path_str.replace('\\', '/').strip().lower()
    base_name = os.path.basename(clean_path)

    # 1. Direct path lookup
    if clean_path in img_index:
        raw_bytes = img_zip.read(img_index[clean_path])
        return Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    
    # 2. Base name match lookup
    for k, v in img_index.items():
        if k.endswith('/' + base_name) or k == base_name:
            raw_bytes = img_zip.read(v)
            return Image.open(io.BytesIO(raw_bytes)).convert('RGB')

    return None


def extract_and_visualize_examples(df, image_zip_path, out_dir):
    """
    Extracts sample images from PlantVillage zip and creates visual galleries.
    """
    print("\n[5/6] Extracting example images for top confusions and failure modes...")
    if not os.path.exists(image_zip_path):
        print(f"Warning: Image zip not found at {image_zip_path}. Skipping image visualizations.")
        return

    with zipfile.ZipFile(image_zip_path, 'r') as img_zip:
        # Build case-insensitive index of all image entries
        img_index = {n.lower(): n for n in img_zip.namelist() if not n.endswith('/')}

        # 1. Example Images: Top Confusions for MobileNetV3
        visualize_confused_samples(
            df=df,
            pred_col='mb_pred',
            model_name="MobileNetV3 Large",
            top_pairs=[
                ('Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight'),
                ('Corn_(maize)___Common_rust_', 'Squash___Powdery_mildew'),
                ('Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot'),
                ('Tomato___Early_blight', 'Tomato___Late_blight'),
            ],
            img_zip=img_zip,
            img_index=img_index,
            save_path=os.path.join(out_dir, "example_top_confusions_mobilenetv3.png")
        )

        # 2. Example Images: Top Confusions for ResNet-50
        visualize_confused_samples(
            df=df,
            pred_col='res_pred',
            model_name="ResNet-50",
            top_pairs=[
                ('Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Northern_Leaf_Blight'),
                ('Tomato___Early_blight', 'Tomato___Late_blight'),
                ('Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'),
                ('Tomato___Target_Spot', 'Tomato___Septoria_leaf_spot'),
            ],
            img_zip=img_zip,
            img_index=img_index,
            save_path=os.path.join(out_dir, "example_top_confusions_resnet50.png")
        )

        # 3. Example Images: Both Models Wrong (Hardest Images)
        visualize_both_wrong_samples(
            df=df,
            img_zip=img_zip,
            img_index=img_index,
            save_path=os.path.join(out_dir, "example_both_wrong.png")
        )

        # 4. Example Images: Disagreements (One Right, One Wrong)
        visualize_disagreement_samples(
            df=df,
            img_zip=img_zip,
            img_index=img_index,
            save_path=os.path.join(out_dir, "example_disagreements.png")
        )


def visualize_confused_samples(df, pred_col, model_name, top_pairs, img_zip, img_index, save_path):
    """
    Renders a 4-sample grid illustrating the top confusion patterns.
    """
    fig, axes = plt.subplots(1, len(top_pairs), figsize=(16, 4.5))
    if len(top_pairs) == 1:
        axes = [axes]

    for ax, (true_c, pred_c) in zip(axes, top_pairs):
        matched = df[(df['true_class'] == true_c) & (df[pred_col] == pred_c)]
        if len(matched) == 0:
            ax.text(0.5, 0.5, "No sample found", ha='center', va='center')
            ax.axis('off')
            continue

        sample_row = matched.iloc[0]
        img = load_image_from_zip(img_zip, sample_row['path'], img_index)
        if img:
            ax.imshow(img)
            t_short = true_c.split('___')[-1].replace('_', ' ')
            p_short = pred_c.split('___')[-1].replace('_', ' ')
            crop_t = true_c.split('___')[0]
            crop_p = pred_c.split('___')[0]
            ax.set_title(
                f"True: {crop_t} - {t_short}\n"
                f"Pred: {crop_p} - {p_short}\n"
                f"({len(matched)} total misclassifications)",
                fontsize=9.5, color='#b2182b', fontweight='bold', pad=8
            )
        else:
            ax.text(0.5, 0.5, "Image file not found", ha='center', va='center')
        ax.axis('off')

    plt.suptitle(f"{model_name}: Example Images of Top Confused Pairs", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved {os.path.basename(save_path)}")


def visualize_both_wrong_samples(df, img_zip, img_index, save_path):
    """
    Renders examples of images where BOTH models failed.
    """
    both_wrong = df[(df['true_class'] != df['mb_pred']) & (df['true_class'] != df['res_pred'])]
    samples = both_wrong.head(4)

    fig, axes = plt.subplots(1, len(samples), figsize=(16, 4.5))
    if len(samples) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, samples.iterrows()):
        img = load_image_from_zip(img_zip, row['path'], img_index)
        if img:
            ax.imshow(img)
            true_fmt = row['true_class'].replace('___', ' - ').replace('_', ' ')
            mb_fmt = row['mb_pred'].replace('___', ' - ').replace('_', ' ')
            res_fmt = row['res_pred'].replace('___', ' - ').replace('_', ' ')
            ax.set_title(
                f"TRUE: {true_fmt}\n"
                f"ResNet-50: {res_fmt}\n"
                f"MobileNet: {mb_fmt}",
                fontsize=8.5, color='#990000', fontweight='bold', pad=6
            )
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center')
        ax.axis('off')

    plt.suptitle("Shared Failure Modes: Examples Where BOTH Models Failed (48 Images Total)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved {os.path.basename(save_path)}")


def visualize_disagreement_samples(df, img_zip, img_index, save_path):
    """
    Renders examples of model disagreements:
    - Left 2: ResNet-50 Correct, MobileNetV3 Wrong (from 158 images)
    - Right 2: MobileNetV3 Correct, ResNet-50 Wrong (from 48 images)
    """
    res_right_mb_wrong = df[(df['true_class'] == df['res_pred']) & (df['true_class'] != df['mb_pred'])].head(2)
    mb_right_res_wrong = df[(df['true_class'] != df['res_pred']) & (df['true_class'] == df['mb_pred'])].head(2)

    samples = pd.concat([res_right_mb_wrong, mb_right_res_wrong])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for i, (ax, (_, row)) in enumerate(zip(axes, samples.iterrows())):
        img = load_image_from_zip(img_zip, row['path'], img_index)
        if img:
            ax.imshow(img)
            true_fmt = row['true_class'].replace('___', ' - ').replace('_', ' ')
            res_fmt = row['res_pred'].replace('___', ' - ').replace('_', ' ')
            mb_fmt = row['mb_pred'].replace('___', ' - ').replace('_', ' ')

            if i < 2:
                status_header = "[ResNet-50 Correct | MobileNet Wrong]"
                title_col = '#08519c'
            else:
                status_header = "[MobileNet Correct | ResNet-50 Wrong]"
                title_col = '#74c476'

            ax.set_title(
                f"{status_header}\n"
                f"True: {true_fmt}\n"
                f"ResNet: {res_fmt}\n"
                f"MobileNet: {mb_fmt}",
                fontsize=8.5, color=title_col, fontweight='bold', pad=6
            )
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center')
        ax.axis('off')

    plt.suptitle("Model Disagreements: Cases Where One Model Succeeds While the Other Fails", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" -> Saved {os.path.basename(save_path)}")


def generate_readme(overlap_stats, df_mb_cls, df_res_cls, mb_pairs, res_pairs, out_dir):
    """
    Generates a comprehensive README.md summarizing all error analysis findings.
    """
    print("\n[6/6] Generating comprehensive README.md...")
    readme_path = os.path.join(out_dir, "README.md")

    # Format tables for markdown
    mb_weak_md = df_mb_cls.head(5)[['class', 'precision', 'recall', 'f1_score', 'support']].to_markdown(index=False)
    res_weak_md = df_res_cls.head(5)[['class', 'precision', 'recall', 'f1_score', 'support']].to_markdown(index=False)
    mb_pairs_md = mb_pairs.head(5)[['True Class', 'Predicted Class', 'count']].to_markdown(index=False)
    res_pairs_md = res_pairs.head(5)[['True Class', 'Predicted Class', 'count']].to_markdown(index=False)

    total = overlap_stats['total']
    both_wrong = overlap_stats['both_wrong']
    only_mb_wrong = overlap_stats['only_mb_wrong']
    only_res_wrong = overlap_stats['only_res_wrong']
    both_correct = overlap_stats['both_correct']
    res_acc = (total - overlap_stats['total_res_errors']) / total * 100
    mb_acc = (total - overlap_stats['total_mb_errors']) / total * 100

    readme_content = f"""# Fine-Tuned Validation Error Analysis: ResNet-50 vs. MobileNetV3


    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f" -> Generated {readme_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING VALIDATION ERROR ANALYSIS: RESNET-50 vs. MOBILENETV3")
    print(f"Target Output Directory: {args.out_dir}")
    print("=" * 80)

    # 1. Load predictions
    df = load_predictions(args.project_zip, seed=args.seed)

    # 2. Weakest classes
    df_mb_cls, df_res_cls = analyze_weakest_classes(df, args.out_dir)

    # 3. Most confused pairs
    mb_pairs, res_pairs = analyze_most_confused_pairs(df, args.out_dir)

    # 4. Error overlap breakdown
    overlap_stats = analyze_error_overlap(df, args.out_dir)

    # 5. Extract & visualize example images
    extract_and_visualize_examples(df, args.image_zip, args.out_dir)

    # 6. Generate README
    generate_readme(overlap_stats, df_mb_cls, df_res_cls, mb_pairs, res_pairs, args.out_dir)

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All artifacts, figures, tables, and README are saved in:")
    print(f"{args.out_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
