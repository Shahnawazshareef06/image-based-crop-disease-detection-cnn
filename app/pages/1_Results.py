import csv
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"mobilenetv3_large": "MobileNetV3-Large", "resnet50": "ResNet-50"}


def read_rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


st.markdown('<div class="tv-brand">TerraVista</div>', unsafe_allow_html=True)
st.title("Model results")
st.write("Compare performance on controlled lab photos and field images. Scores are averaged across seeds 42, 123 and 2026.")

lab, field = st.tabs(["PlantVillage lab test", "Lab vs field"])
with lab:
    rows = read_rows("experiments/test_evaluation/test_results_summary.csv")
    st.subheader("Held-out lab performance")
    st.caption("7,978 images · 38 classes")
    for card, row in zip(st.columns(len(rows)), rows):
        card.metric(NAMES[row["model"]] + " · macro-F1", f"{float(row['test_macro_f1_mean']):.4f}", border=True)
    table = [{"Model": NAMES[r["model"]], "Accuracy": f"{float(r['test_accuracy_mean']):.2%}",
              "Macro-F1": f"{float(r['test_macro_f1_mean']):.4f}",
              "95% interval for macro-F1": f"{float(r['ci95_low_seed_mean']):.4f} – {float(r['ci95_high_seed_mean']):.4f}"}
             for r in rows]
    st.dataframe(table, hide_index=True, width="stretch")
with field:
    rows = read_rows("experiments/plantdoc-field-test/duplicate_screened_metrics.csv")
    st.subheader("The same 27 classes in two settings")
    st.caption("Duplicate-screened cohorts · Macro-F1 ranges from 0 to 1")
    for card, (model, name) in zip(st.columns(len(NAMES)), NAMES.items()):
        with card.container(border=True):
            st.markdown(f"**{name}**")
            group = sorted((r for r in rows if r["model"] == model), key=lambda r: r["domain"] == "plantdoc_test")
            for row in group:
                value = float(row["macro_f1_mean"])
                dataset = "PlantDoc field" if row["domain"] == "plantdoc_test" else "PlantVillage lab"
                st.progress(value, text=f"{dataset} · {value:.4f} · {int(row['images']):,} images")
    st.write("Both models scored lower on field photos. Use field performance when interpreting a grower's upload.")
    with st.expander("View comparison table"):
        st.dataframe([{"Model": NAMES[r["model"]], "Dataset": r["domain"].split("_")[0],
                       "Images": int(r["images"]), "Macro-F1": float(r["macro_f1_mean"])} for r in rows],
                     hide_index=True, width="stretch")
st.caption("Macro-F1 gives every class equal weight. The analysis page uses the seed-42 checkpoint for your selected model.")
