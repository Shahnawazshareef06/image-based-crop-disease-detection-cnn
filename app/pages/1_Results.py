import csv
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="TerraVista · Results", page_icon="🌿")
st.title("Model results")
st.write("Held-out evaluation across three training seeds (42, 123 and 2026). "
         "The prediction page uses seed 42.")
names = {"mobilenetv3_large": "MobileNetV3-Large", "resnet50": "ResNet-50"}
with (ROOT / "experiments/test_evaluation/test_results_summary.csv").open() as f:
    rows = [{"Model": names[r["model"]], "Accuracy": f"{float(r['test_accuracy_mean']):.2%}",
             "Macro-F1": f"{float(r['test_macro_f1_mean']):.4f}",
             "95% CI for macro-F1": f"{float(r['ci95_low_seed_mean']):.4f} – {float(r['ci95_high_seed_mean']):.4f}"}
            for r in csv.DictReader(f)]
st.subheader("PlantVillage lab test")
st.caption("7,978 images across all 38 classes")
st.dataframe(rows, hide_index=True, width="stretch")
st.subheader("Lab and field comparison")
st.caption("Same 27 classes after duplicate screening. Macro-F1 gives each class equal weight.")
with (ROOT / "experiments/plantdoc-field-test/duplicate_screened_metrics.csv").open() as f:
    rows = [{"Model": names[r["model"]], "Dataset": "PlantDoc field" if r["domain"] == "plantdoc_test" else "PlantVillage lab",
             "Images": int(r["images"]), "Macro-F1": f"{float(r['macro_f1_mean']):.4f}"}
            for r in csv.DictReader(f)]
st.dataframe(rows, hide_index=True, width="stretch")
