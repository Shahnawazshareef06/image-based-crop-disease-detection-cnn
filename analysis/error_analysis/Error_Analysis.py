
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
from sklearn.metrics import classification_report

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def safe_to_csv(df, filepath, **kwargs):
    """
    Safely writes DataFrame to CSV. If the file is open in Excel or another program,
    catches PermissionError gracefully without crashing the pipeline.
    """
    try:
        df.to_csv(filepath, **kwargs)
        print(f" -> Saved {os.path.basename(filepath)}")
    except PermissionError:
        print(f" -> Note: {os.path.basename(filepath)} is currently open in Excel/viewer. Existing verified contents retained.")
    except Exception as e:
        print(f" -> Warning saving {os.path.basename(filepath)}: {e}")


def safe_write_text(filepath, content):
   
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f" -> Generated {os.path.basename(filepath)}")
    except PermissionError:
        print(f" -> Note: {os.path.basename(filepath)} is open in another viewer. Existing verified contents retained.")
    except Exception as e:
        print(f" -> Warning generating {os.path.basename(filepath)}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed validation error analysis: ResNet-50 vs MobileNetV3.")
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
        default=r"C:\Users\harsh\OneDrive\Desktop\iiith\Error_Analysis",
        help="Directory to save generated outputs."
    )
    return parser.parse_args()


def load_all_seeds(project_zip_path, seeds=[42, 123, 2026]):
    """
    Extracts and aligns validation predictions for both models across all specified seeds.
    """
    print(f"\n[1/6] Loading predictions across all three seeds {seeds}...")
    if not os.path.exists(project_zip_path):
        raise FileNotFoundError(f"Project zip not found: {project_zip_path}")

    seed_data = {}
    with zipfile.ZipFile(project_zip_path, 'r') as pz:
        inner_res_zip = "image-based-crop-disease-detection-cnn-main/Resnet-50/terravista_finetune_p10_logs_predictions.zip"
        if inner_res_zip not in pz.namelist():
            raise FileNotFoundError(f"ResNet-50 inner zip not found: {inner_res_zip}")

        res_bytes = pz.read(inner_res_zip)
        with zipfile.ZipFile(io.BytesIO(res_bytes), 'r') as rz:
            for s in seeds:
                mb_path = f"image-based-crop-disease-detection-cnn-main/mobilenetv3/results/mobilenetv3/mobilenetv3_large_uniform_p010_seed{s}/val_predictions.csv"
                res_path = f"runs_finetune_p10/resnet50/frac100_seed{s}/val_predictions.csv"

                with pz.open(mb_path) as mf:
                    df_mb = pd.read_csv(mf).rename(columns={'true': 'true_class', 'pred': 'mb_pred'})
                with rz.open(res_path) as rf:
                    df_res = pd.read_csv(rf).rename(columns={'pred_class': 'res_pred', 'confidence': 'res_conf'})

                if not (df_mb['path'] == df_res['path']).all():
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

                seed_data[s] = df
                print(f" -> Seed {s:4d}: Loaded {len(df)} aligned validation rows")

    return seed_data


def analyze_multi_seed_overlap(seed_data, out_dir):
    """
    Computes per-seed and aggregated (Mean ± SD) error overlap statistics.
    """
    print("\n[2/6] Analyzing error overlap across all three seeds...")
    records = []
    total_imgs = 7976

    for s, df in seed_data.items():
        mb_err = df['true_class'] != df['mb_pred']
        res_err = df['true_class'] != df['res_pred']

        both_corr = int((~mb_err & ~res_err).sum())
        only_mb = int((mb_err & ~res_err).sum())
        only_res = int((~mb_err & res_err).sum())
        both_err = int((mb_err & res_err).sum())
        res_tot = int(res_err.sum())
        mb_tot = int(mb_err.sum())

        records.append({
            'Seed': s,
            'ResNet_Errors': res_tot,
            'ResNet_Acc': (total_imgs - res_tot) / total_imgs * 100,
            'MobileNet_Errors': mb_tot,
            'MobileNet_Acc': (total_imgs - mb_tot) / total_imgs * 100,
            'Both_Correct': both_corr,
            'Both_Correct_Pct': both_corr / total_imgs * 100,
            'Only_MobileNet_Wrong': only_mb,
            'Only_MobileNet_Wrong_Pct': only_mb / total_imgs * 100,
            'Only_ResNet_Wrong': only_res,
            'Only_ResNet_Wrong_Pct': only_res / total_imgs * 100,
            'Both_Wrong': both_err,
            'Both_Wrong_Pct': both_err / total_imgs * 100,
            'Oracle_Ensemble_Acc': (total_imgs - both_err) / total_imgs * 100
        })

    overlap_df = pd.DataFrame(records)

    # Compute 3-seed Mean and SD
    mean_row = {'Seed': 'Mean (3 Seeds)'}
    sd_row = {'Seed': 'SD (±)'}
    for col in overlap_df.columns:
        if col != 'Seed':
            mean_row[col] = float(overlap_df[col].mean())
            sd_row[col] = float(overlap_df[col].std())

    summary_df = pd.concat([overlap_df, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
    summary_path = os.path.join(out_dir, "multi_seed_error_overlap_summary.csv")
    safe_to_csv(summary_df, summary_path, index=False)

    print("\n" + "=" * 75)
    print("           MULTI-SEED VALIDATION ERROR OVERLAP SUMMARY")
    print("=" * 75)
    for idx, r in overlap_df.iterrows():
        s_val = int(r['Seed'])
        res_err = int(r['ResNet_Errors'])
        mb_err = int(r['MobileNet_Errors'])
        bw = int(r['Both_Wrong'])
        print(f"Seed {s_val:4d} | ResNet Acc: {r['ResNet_Acc']:.2f}% ({res_err:3d} err) | MB Acc: {r['MobileNet_Acc']:.2f}% ({mb_err:3d} err) | Both Wrong: {bw:2d} ({r['Both_Wrong_Pct']:.2f}%)")
    print("-" * 75)
    print(f"3-Seed Mean: ResNet Acc: {mean_row['ResNet_Acc']:.2f}±{sd_row['ResNet_Acc']:.2f}% | MB Acc: {mean_row['MobileNet_Acc']:.2f}±{sd_row['MobileNet_Acc']:.2f}% | Both Wrong: {mean_row['Both_Wrong']:.1f}±{sd_row['Both_Wrong']:.1f} ({mean_row['Both_Wrong_Pct']:.2f}%)")
    print(f"Theoretical Oracle Ensemble Acc: {mean_row['Oracle_Ensemble_Acc']:.2f}±{sd_row['Oracle_Ensemble_Acc']:.2f}%")
    print("=" * 75)

    # Identify persistent shared failures (Both models wrong in ALL 3 seeds)
    seed_keys = list(seed_data.keys())
    s1, s2, s3 = seed_keys[0], seed_keys[1], seed_keys[2]
    bw1 = set(seed_data[s1][(seed_data[s1]['true_class'] != seed_data[s1]['mb_pred']) & (seed_data[s1]['true_class'] != seed_data[s1]['res_pred'])]['path'])
    bw2 = set(seed_data[s2][(seed_data[s2]['true_class'] != seed_data[s2]['mb_pred']) & (seed_data[s2]['true_class'] != seed_data[s2]['res_pred'])]['path'])
    bw3 = set(seed_data[s3][(seed_data[s3]['true_class'] != seed_data[s3]['mb_pred']) & (seed_data[s3]['true_class'] != seed_data[s3]['res_pred'])]['path'])

    persistent_paths = sorted(list(bw1.intersection(bw2).intersection(bw3)))
    print(f"\n -> Found {len(persistent_paths)} PERSISTENT shared failures across all 3 seeds!")

    persistent_df = seed_data[s1][seed_data[s1]['path'].isin(persistent_paths)][['path', 'true_class', 'res_pred', 'mb_pred']].copy()
    safe_to_csv(persistent_df, os.path.join(out_dir, "persistent_both_wrong_images.csv"), index=False)

    return overlap_df, summary_df, persistent_df


def analyze_multi_seed_weakest_classes(seed_data, out_dir):
    """
    Computes per-class F1 across all 3 seeds, outputting:
    1. multi_seed_weakest_classes.csv (All classes ranked by combined mean F1)
    2. resnet50_weakest_classes_multiseed.csv (Ranked specifically for ResNet-50)
    3. mobilenetv3_weakest_classes_multiseed.csv (Ranked specifically for MobileNetV3)
    """
    print("\n[3/6] Computing weakest classes across all three seeds...")
    classes = None
    res_f1_by_seed = {s: {} for s in seed_data}
    mb_f1_by_seed = {s: {} for s in seed_data}
    support_dict = {}

    for s, df in seed_data.items():
        rep_res = classification_report(df['true_class'], df['res_pred'], output_dict=True, zero_division=0)
        rep_mb = classification_report(df['true_class'], df['mb_pred'], output_dict=True, zero_division=0)

        if classes is None:
            classes = sorted([c for c in rep_res.keys() if c not in ('accuracy', 'macro avg', 'weighted avg')])

        for c in classes:
            res_f1_by_seed[s][c] = rep_res[c]['f1-score']
            mb_f1_by_seed[s][c] = rep_mb[c]['f1-score']
            support_dict[c] = int(rep_res[c]['support'])

    rows = []
    for c in classes:
        r_f1s = [res_f1_by_seed[s][c] for s in seed_data]
        m_f1s = [mb_f1_by_seed[s][c] for s in seed_data]

        rows.append({
            'Class': c,
            'Clean_Name': c.replace('___', ' - ').replace('_', ' '),
            'Support': support_dict[c],
            'ResNet_Seed42_F1': r_f1s[0],
            'ResNet_Seed123_F1': r_f1s[1],
            'ResNet_Seed2026_F1': r_f1s[2],
            'ResNet_Mean_F1': float(np.mean(r_f1s)),
            'ResNet_SD_F1': float(np.std(r_f1s)),
            'MobileNet_Seed42_F1': m_f1s[0],
            'MobileNet_Seed123_F1': m_f1s[1],
            'MobileNet_Seed2026_F1': m_f1s[2],
            'MobileNet_Mean_F1': float(np.mean(m_f1s)),
            'MobileNet_SD_F1': float(np.std(m_f1s)),
            'Combined_Mean_F1': float(np.mean(r_f1s + m_f1s))
        })

    # 1. Combined
    weakest_df = pd.DataFrame(rows).sort_values(by='Combined_Mean_F1', ascending=True)
    safe_to_csv(weakest_df, os.path.join(out_dir, "multi_seed_weakest_classes.csv"), index=False)

    # 2. ResNet-50 specific
    res_weak_df = weakest_df[['Class', 'Clean_Name', 'Support',
                              'ResNet_Seed42_F1', 'ResNet_Seed123_F1', 'ResNet_Seed2026_F1',
                              'ResNet_Mean_F1', 'ResNet_SD_F1']].sort_values(by='ResNet_Mean_F1', ascending=True)
    safe_to_csv(res_weak_df, os.path.join(out_dir, "resnet50_weakest_classes_multiseed.csv"), index=False)

    # 3. MobileNetV3 specific
    mb_weak_df = weakest_df[['Class', 'Clean_Name', 'Support',
                             'MobileNet_Seed42_F1', 'MobileNet_Seed123_F1', 'MobileNet_Seed2026_F1',
                             'MobileNet_Mean_F1', 'MobileNet_SD_F1']].sort_values(by='MobileNet_Mean_F1', ascending=True)
    safe_to_csv(mb_weak_df, os.path.join(out_dir, "mobilenetv3_weakest_classes_multiseed.csv"), index=False)

    print("\n--- Top 5 Consistently Weakest Classes for ResNet-50 (3-Seed Mean F1) ---")
    print(res_weak_df[['Clean_Name', 'ResNet_Mean_F1', 'ResNet_SD_F1', 'Support']].head(5).to_string(index=False))

    print("\n--- Top 5 Consistently Weakest Classes for MobileNetV3 (3-Seed Mean F1) ---")
    print(mb_weak_df[['Clean_Name', 'MobileNet_Mean_F1', 'MobileNet_SD_F1', 'Support']].head(5).to_string(index=False))

    return weakest_df, res_weak_df, mb_weak_df


def format_pair_label(true_c, pred_c):
    """
    Creates concise, human-readable labels for confusion pairs, clearly distinguishing
    within-crop vs cross-crop confusion.
    """
    crop_t = true_c.split('___')[0].replace('_', ' ')
    dis_t = true_c.split('___')[1].replace('_', ' ')
    
    if '___' in pred_c:
        crop_p = pred_c.split('___')[0].replace('_', ' ')
        dis_p = pred_c.split('___')[1].replace('_', ' ')
    else:
        crop_p = ''
        dis_p = pred_c.replace('_', ' ')
        
    if crop_t.strip().lower() == crop_p.strip().lower():
        return f"{crop_t}: {dis_t} → {dis_p}"
    else:
        return f"{crop_t} ({dis_t}) → {crop_p} ({dis_p})"


def analyze_multi_seed_confused_pairs(seed_data, out_dir):
    """
    Computes most confused disease pairs (True -> Pred) across Seeds 42, 123, 2026
    for both ResNet-50 and MobileNetV3.
    """
    print("\n[4/6] Analyzing most confused pairs across all three seeds...")
    seeds = list(seed_data.keys())

    # MobileNetV3 confused pairs
    mb_records = {}
    for s in seeds:
        df = seed_data[s]
        err = df[df['true_class'] != df['mb_pred']]
        counts = err.groupby(['true_class', 'mb_pred']).size()
        for (tc, pc), cnt in counts.items():
            pair_key = (tc, pc)
            if pair_key not in mb_records:
                mb_records[pair_key] = {
                    'True_Class': tc,
                    'Predicted_Class': pc,
                    'Clean_Label': format_pair_label(tc, pc),
                    'Seed_42_Count': 0,
                    'Seed_123_Count': 0,
                    'Seed_2026_Count': 0
                }
            mb_records[pair_key][f'Seed_{s}_Count'] = int(cnt)

    mb_list = list(mb_records.values())
    for r in mb_list:
        r['Total_Count'] = r['Seed_42_Count'] + r['Seed_123_Count'] + r['Seed_2026_Count']
        r['Mean_Count'] = float(r['Total_Count'] / 3.0)

    mb_pairs_df = pd.DataFrame(mb_list).sort_values(by='Total_Count', ascending=False)
    safe_to_csv(mb_pairs_df, os.path.join(out_dir, "mobilenetv3_most_confused_pairs_multiseed.csv"), index=False)

    # ResNet-50 confused pairs
    res_records = {}
    for s in seeds:
        df = seed_data[s]
        err = df[df['true_class'] != df['res_pred']]
        counts = err.groupby(['true_class', 'res_pred']).size()
        for (tc, pc), cnt in counts.items():
            pair_key = (tc, pc)
            if pair_key not in res_records:
                res_records[pair_key] = {
                    'True_Class': tc,
                    'Predicted_Class': pc,
                    'Clean_Label': format_pair_label(tc, pc),
                    'Seed_42_Count': 0,
                    'Seed_123_Count': 0,
                    'Seed_2026_Count': 0
                }
            res_records[pair_key][f'Seed_{s}_Count'] = int(cnt)

    res_list = list(res_records.values())
    for r in res_list:
        r['Total_Count'] = r['Seed_42_Count'] + r['Seed_123_Count'] + r['Seed_2026_Count']
        r['Mean_Count'] = float(r['Total_Count'] / 3.0)

    res_pairs_df = pd.DataFrame(res_list).sort_values(by='Total_Count', ascending=False)
    safe_to_csv(res_pairs_df, os.path.join(out_dir, "resnet50_most_confused_pairs_multiseed.csv"), index=False)

    print("\n--- Top 5 Most Confused Pairs for ResNet-50 (3 Seeds Total) ---")
    print(res_pairs_df[['Clean_Label', 'Total_Count', 'Mean_Count', 'Seed_42_Count', 'Seed_123_Count', 'Seed_2026_Count']].head(5).to_string(index=False))

    print("\n--- Top 5 Most Confused Pairs for MobileNetV3 (3 Seeds Total) ---")
    print(mb_pairs_df[['Clean_Label', 'Total_Count', 'Mean_Count', 'Seed_42_Count', 'Seed_123_Count', 'Seed_2026_Count']].head(5).to_string(index=False))

    return mb_pairs_df, res_pairs_df


def render_confused_pairs_plot(mb_pairs_df, res_pairs_df, out_dir):
    """
    Renders publication-quality horizontal bar chart comparing top confused pairs across both architectures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2))

    # MobileNetV3
    top_mb = mb_pairs_df.head(7).iloc[::-1]
    y1 = np.arange(len(top_mb))
    ax1.barh(y1, top_mb['Total_Count'], color='#4575b4', alpha=0.9, height=0.62)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(top_mb['Clean_Label'], fontsize=9.5, fontweight='bold')
    ax1.set_xlabel('Total Errors Across 3 Seeds (Sum)', fontsize=11, fontweight='bold')
    ax1.set_title('MobileNetV3: Top Confused Disease Pairs\n(Aggregated Across Seeds 42, 123, 2026)', fontsize=12, fontweight='bold')
    max_mb = top_mb['Total_Count'].max()
    for i, (tot, mean_c) in enumerate(zip(top_mb['Total_Count'], top_mb['Mean_Count'])):
        ax1.text(tot + 0.8, i, f"{tot} (avg {mean_c:.1f}/seed)", va='center', fontsize=9, fontweight='bold', color='#1a365d')
    ax1.set_xlim(0, max_mb * 1.25)

    # ResNet-50
    top_res = res_pairs_df.head(7).iloc[::-1]
    y2 = np.arange(len(top_res))
    ax2.barh(y2, top_res['Total_Count'], color='#d73027', alpha=0.9, height=0.62)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(top_res['Clean_Label'], fontsize=9.5, fontweight='bold')
    ax2.set_xlabel('Total Errors Across 3 Seeds (Sum)', fontsize=11, fontweight='bold')
    ax2.set_title('ResNet-50: Top Confused Disease Pairs\n(Aggregated Across Seeds 42, 123, 2026)', fontsize=12, fontweight='bold')
    max_res = top_res['Total_Count'].max()
    for i, (tot, mean_c) in enumerate(zip(top_res['Total_Count'], top_res['Mean_Count'])):
        ax2.text(tot + 0.5, i, f"{tot} (avg {mean_c:.1f}/seed)", va='center', fontsize=9, fontweight='bold', color='#7b1113')
    ax2.set_xlim(0, max_res * 1.30)

    plt.suptitle('Most Confused Disease Classification Pairs: MobileNetV3 vs. ResNet-50 (Multi-Seed Aggregation)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "multi_seed_top_confused_pairs.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f" -> Saved {os.path.basename(plot_path)}")


def render_multi_seed_overview(overlap_df, weakest_df, out_dir):
    """
    Renders 2-panel figure:
    Panel A: Error breakdown comparison across Seeds 42, 123, 2026.
    Panel B: Weakest classes comparison across models with error bars.
    """
    print("\n[5/6] Rendering consolidated multi-seed overview figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Stacked bar chart of error breakdown across seeds
    seeds_str = [f"Seed {s}" for s in overlap_df['Seed']]
    x = np.arange(len(seeds_str))
    width = 0.45

    shared = overlap_df['Both_Wrong']
    only_res = overlap_df['Only_ResNet_Wrong']
    only_mb = overlap_df['Only_MobileNet_Wrong']

    ax1.bar(x, shared, width, label='Shared Errors (Both Models Failed)', color='#d73027', alpha=0.9)
    ax1.bar(x, only_res, width, bottom=shared, label='Only ResNet-50 Failed', color='#fc8d59', alpha=0.85)
    ax1.bar(x, only_mb, width, bottom=shared + only_res, label='Only MobileNetV3 Failed', color='#4575b4', alpha=0.85)

    for i in range(len(seeds_str)):
        tot = shared.iloc[i] + only_res.iloc[i] + only_mb.iloc[i]
        ax1.text(i, shared.iloc[i] / 2, f"{shared.iloc[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax1.text(i, tot + 5, f"Total Errs: {tot}\n(Both Wrong: {shared.iloc[i]})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds_str, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Misclassified Validation Images', fontsize=11, fontweight='bold')
    ax1.set_title('Validation Error Composition Across Seeds\n(Total Validation Size: 7,976)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 320)
    ax1.legend(loc='upper right', frameon=True)

    # Panel B: Top 6 Weakest Classes (Side-by-side bar chart with error bars)
    top_weak = weakest_df.head(6)
    y = np.arange(len(top_weak))
    h = 0.38

    ax2.barh(y - h/2, top_weak['MobileNet_Mean_F1'], h, xerr=top_weak['MobileNet_SD_F1'],
             capsize=4, label='MobileNetV3 Large (Mean ± SD)', color='#4575b4', alpha=0.9)
    ax2.barh(y + h/2, top_weak['ResNet_Mean_F1'], h, xerr=top_weak['ResNet_SD_F1'],
             capsize=4, label='ResNet-50 (Mean ± SD)', color='#d73027', alpha=0.9)

    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(top_weak['Clean_Name'], fontsize=9.5, fontweight='bold')
    ax2.set_xlabel('Macro F1-Score (3-Seed Average)', fontsize=11, fontweight='bold')
    ax2.set_title('Consistently Weakest Classes Across Seeds\n(Ranked by 3-Seed Combined Mean F1)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0.55, 1.02)
    ax2.legend(loc='lower right', frameon=True)

    plt.suptitle('Multi-Seed Comparative Error Analysis: ResNet-50 vs. MobileNetV3 Large (Seeds 42, 123, 2026)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "multi_seed_error_analysis_overview.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f" -> Saved {os.path.basename(plot_path)}")


def render_persistent_failure_examples(persistent_df, image_zip_path, out_dir):
    """
    Renders sample images of persistent failures that both models failed across all 3 seeds.
    """
    print("\n[6/6] Extracting visual samples of persistent shared failures...")
    if not os.path.exists(image_zip_path):
        print("Warning: Image zip not found. Skipping example visualization.")
        return

    with zipfile.ZipFile(image_zip_path, 'r') as img_zip:
        index_map = {os.path.basename(n).lower(): n for n in img_zip.namelist() if not n.endswith('/')}

        samples = persistent_df.head(4)
        fig, axes = plt.subplots(1, len(samples), figsize=(16, 4.2))
        if len(samples) == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, samples.iterrows()):
            base = os.path.basename(row['path']).lower()
            if base in index_map:
                img_data = img_zip.read(index_map[base])
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                ax.imshow(img)
                t_str = row['true_class'].replace('___', ' - ').replace('_', ' ')
                res_str = row['res_pred'].split('___')[-1].replace('_', ' ')
                mb_str = row['mb_pred'].split('___')[-1].replace('_', ' ')
                ax.set_title(
                    f"TRUE: {t_str}\n"
                    f"ResNet Pred: {res_str}\n"
                    f"MobileNet Pred: {mb_str}",
                    fontsize=8.5, color='#990000', fontweight='bold', pad=5
                )
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center')
            ax.axis('off')

        plt.suptitle("Persistent Shared Failures: Images Misclassified by BOTH Models Across ALL 3 Seeds", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "example_persistent_failures.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f" -> Saved {os.path.basename(save_path)}")


