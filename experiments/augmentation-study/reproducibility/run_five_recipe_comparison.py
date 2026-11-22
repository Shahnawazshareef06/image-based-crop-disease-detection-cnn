"""Run the five TerraVista recipes with the matched three-seed protocol.

Example:
 python run_five_recipe_comparison.py --color-dir /path/PlantVillage/color
"""
from __future__ import annotations

from terravista_bootstrap import apply_kaggle_triton_guard
apply_kaggle_triton_guard()

import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np

from augmentation_pipeline import AUGMENTATION_RECIPES
from terravista_dataset import make_plantvillage_datasets
from terravista_training_example import run_one_recipe_seed
from terravista_resnet50_settings import SEEDS, model_settings_dict, environment_report, resolve_device


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--color-dir", required=True, help="PlantVillage 'color' directory")
    parser.add_argument("--split-csv", default=str(base / "train_val_split.csv"))
    parser.add_argument("--output-dir", default="./terravista_runs")
    parser.add_argument("--cache-dir", default="./terravista_cache")
    parser.add_argument("--reuse-cache-dir", nargs="*", default=[], help="Read-only prior-version cache folder(s), e.g. a Kaggle attached Output/cache directory")
    parser.add_argument("--recipes", nargs="+", default=list(AUGMENTATION_RECIPES), choices=list(AUGMENTATION_RECIPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--allow-different-split", action="store_true", help="Allow a non-bundled split after explicit review; counts/classes are still validated")
    args = parser.parse_args()

    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError("The matched TerraVista comparison is GPU-based. Select a CUDA GPU; run verify_handoff.py for CPU smoke tests.")

    class_map_path = base / "class_to_idx.json"
    class_to_idx = json.loads(class_map_path.read_text(encoding="utf-8"))
    manifest = json.loads((base / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    class_map_sha = hashlib.sha256(class_map_path.read_bytes()).hexdigest()
    if class_map_sha != manifest["dataset"]["class_mapping_sha256"]:
        raise RuntimeError("Bundled class_to_idx.json does not match reproducibility_manifest.json.")
    actual_split_sha = hashlib.sha256(Path(args.split_csv).read_bytes()).hexdigest()
    expected_split_sha = manifest["dataset"]["train_val_manifest_sha256"]
    if actual_split_sha != expected_split_sha and not args.allow_different_split:
        raise RuntimeError(
            "The split CSV SHA-256 does not match the bundled reproducibility manifest. "
            "Use the bundled train_val_split.csv, or pass --allow-different-split only "
            "after reviewing the alternate split."
        )
    if actual_split_sha != expected_split_sha:
        print("WARNING: using an alternate split manifest; scores will not exactly reproduce the recorded experiment.")
    need_masks = any(AUGMENTATION_RECIPES[name]["p_background"] > 0 for name in args.recipes)
    train_ds, val_ds, train_df, val_df, mask_path, image_cache_path, cache_fingerprint = make_plantvillage_datasets(
        args.split_csv, args.color_dir, class_to_idx,
        need_background_masks=need_masks, cache_dir=args.cache_dir,
        reuse_cache_dirs=args.reuse_cache_dir,
    )
    print(f"Train={len(train_ds):,}; validation={len(val_ds):,}; classes={len(class_to_idx)}")
    print("Test images: not resolved, loaded, or evaluated.")
    print("Decoded train+validation image cache:", image_cache_path)
    print("Image-cache fingerprint:", cache_fingerprint)
    if mask_path:
        print("Training-only alpha-mask cache:", mask_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    run_metadata_base = {"split_sha256": actual_split_sha, "train_rows": len(train_ds), "validation_rows": len(val_ds)}
    for recipe_name in args.recipes:
        for seed in args.seeds:
            run_dir = output_dir / f"{recipe_name}_seed{int(seed)}"
            summary_file = run_dir / "run_summary.json"
            checkpoint_file = run_dir / "best_checkpoint.pt"
            history_file = run_dir / "history.csv"
            if summary_file.is_file() and checkpoint_file.is_file() and history_file.is_file():
                try:
                    previous = json.loads(summary_file.read_text(encoding="utf-8"))
                    same_run = (
                        previous.get("recipe_name") == recipe_name
                        and int(previous.get("seed", -1)) == int(seed)
                        and previous.get("run_metadata", {}).get("split_sha256") == actual_split_sha
                        and previous.get("augmentation_probabilities") == AUGMENTATION_RECIPES[recipe_name]
                        and previous.get("model_settings") == model_settings_dict()
                        and previous.get("environment") == environment_report()
                    )
                    if same_run:
                        print(f"Skipping completed recipe/seed from matching saved artifacts: {recipe_name}, seed={seed}")
                        results.append(previous)
                        rows = [{k: r[k] for k in ("recipe_name", "seed", "best_epoch", "best_validation_macro_f1")} for r in results]
                        with (output_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                            writer.writeheader(); writer.writerows(rows)
                        continue
                except Exception:
                    print(f"Existing saved artifacts for {recipe_name}, seed={seed} could not be verified; rerunning this pair.")

            result = run_one_recipe_seed(
                train_ds, val_ds, recipe_name=recipe_name, seed=seed,
                class_to_idx=class_to_idx, output_dir=output_dir,
                run_metadata={**run_metadata_base, "image_cache_fingerprint": cache_fingerprint, "image_cache_path": str(image_cache_path), "alpha_mask_cache_path": str(mask_path) if mask_path else None},
            )
            results.append(result)
            # Save after every completed seed so progress survives interruption.
            rows = [{k: r[k] for k in (
                "recipe_name", "seed", "best_epoch", "best_validation_macro_f1"
            )} for r in results]
            with (output_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

    recipe_summary = {}
    for recipe_name in args.recipes:
        rows = [r for r in results if r["recipe_name"] == recipe_name]
        scores = [float(r["best_validation_macro_f1"]) for r in rows]
        recipe_summary[recipe_name] = {
            "mean_best_validation_macro_f1": float(np.mean(scores)),
            "sample_std_best_validation_macro_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "seed_scores": {str(r["seed"]): r["best_validation_macro_f1"] for r in rows},
            "probabilities": AUGMENTATION_RECIPES[recipe_name],
        }
    ranked_recipes = sorted(
        recipe_summary.items(),
        key=lambda item: item[1]["mean_best_validation_macro_f1"],
        reverse=True,
    )
    augmented_ranked = [item for item in ranked_recipes if item[0] != "no_augmentation"]
    summary = {
        "selection_metric": "mean best validation macro-F1 over epochs, averaged across seeds",
        "ranking_descending": [
            {"rank": index + 1, "recipe_name": name, **values}
            for index, (name, values) in enumerate(ranked_recipes)
        ],
        "best_overall_recipe": ranked_recipes[0][0] if ranked_recipes else None,
        "best_augmented_recipe": augmented_ranked[0][0] if augmented_ranked else None,
        "seeds": args.seeds,
        "split_sha256": actual_split_sha,
        "train_rows": len(train_ds),
        "validation_rows": len(val_ds),
        "test_data_used": False,
        "model_settings": model_settings_dict(),
        "environment": environment_report(),
        "image_cache_fingerprint": cache_fingerprint,
        "recipes": recipe_summary,
    }
    (output_dir / "recipe_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", output_dir / "seed_results.csv")
    print("Saved:", output_dir / "recipe_comparison_summary.json")


if __name__ == "__main__":
    main()
