# MobileNetV3 Handover (PlantVillage, 38 classes)

Same protocol as the ResNet-50 / ViT-Tiny runs: AdamW (backbone 1e-5, head 5e-4, wd 1e-4),
CrossEntropyLoss, 3 epochs, batch 32, seeds 42/123/2026, FP16, no LR scheduler.
Model: torchvision `mobilenet_v3_large`, ImageNet weights, all layers trainable, new head `classifier[3]`.

| File | Role |
|---|---|
| `model.py` | `build_mobilenetv3()`, `make_optimizer()` (head = `classifier[3]`) |
| `config.py` | shared protocol constants (LRs, seeds, epochs, batch sizes, AMP) |
| `train.py` | training / validation / best-checkpoint script |
| `dataset.py` | cached PlantVillage dataset + background masks |
| `engine.py` | `train_one_epoch`, `evaluate_validation` (shared handoff code) |
| `augmentation_pipeline.py`, `terravista_bootstrap.py` | shared handoff code |
| `configurations/config.json` | full hyperparameters |
| `configurations/mobilenetv3_best_augmentation_config.json` | uniform_p010 recipe |
| `results/mean_std_results.csv` | per-seed + mean/std validation macro-F1 |

Validation results (3 seeds, mean macro-F1 +/- std): `no_augmentation` 0.9743 +/- 0.0014; `uniform_p010` 0.9733 +/- 0.0051. The `uniform_p010` recipe was selected on ResNet-50 and reused here without re-selection; on MobileNetV3 it does not beat no augmentation (difference within seed noise).

Run (GPU needed; first run builds the image cache):

    pip install -q -r requirements.txt
    python train.py --color-dir <PlantVillage/color> --cache-dir ./cache --output-dir results/mobilenetv3
    python make_results_csv.py results/mobilenetv3
