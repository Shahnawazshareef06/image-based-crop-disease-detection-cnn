import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import streamlit as st
from PIL import Image, ImageChops

from predict import MODEL_NAMES, THRESHOLD, ROOT, load_model, predict, prepare

EXAMPLE = Path(__file__).resolve().parents[1] / "assets/cherry.jpg"


@st.cache_resource(max_entries=1, show_spinner=False)
def selected_model(name):
    return load_model(name)


def reset_example():
    st.session_state.pop("use_example", None)
    st.session_state.pop("last_analysis", None)


st.markdown('<div class="tv-brand">TerraVista</div>', unsafe_allow_html=True)
st.title("Leaf analysis")
st.write("Check a single leaf against 38 crop and condition classes, then explore the regions behind the prediction.")

with st.container(key="intake"):
    controls, reference = st.columns([1.45, 1], gap="medium")
with controls:
    with st.container(border=True):
        st.subheader("Analyse a photo")
        upload = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"],
                                  max_upload_size=12, on_change=reset_example,
                                  help="Use a sharp close-up in even light. Maximum 12 MB and 24 megapixels.")
        name = st.selectbox("Prediction model", list(MODEL_NAMES), format_func=MODEL_NAMES.get,
                            help="MobileNetV3-Large is the smaller CPU model. ResNet-50 scored higher on the project’s held-out test.")
        st.caption("Analysis starts automatically when you choose a photo.")
with reference:
    if upload is None:
        img = base64.b64encode(EXAMPLE.read_bytes()).decode("ascii")
        st.markdown(f'<figure class="tv-specimen"><img src="data:image/jpeg;base64,{img}" alt="A single cherry leaf photographed against a plain background"><figcaption>Sample photo <strong>Cherry leaf</strong></figcaption></figure>', unsafe_allow_html=True)
        if st.button("Use this sample" if not st.session_state.get("use_example") else "Clear sample", width="stretch"):
            st.session_state.pop("last_analysis", None)
            st.session_state.use_example = not st.session_state.get("use_example", False)
            st.rerun()
        st.caption("Photo from the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset).")


source = upload if upload is not None else EXAMPLE if st.session_state.get("use_example") else None
if source is not None:
    try:
        photo = prepare(source)
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("Choose a readable JPEG, PNG or WebP photo with at most 24 megapixels.")
        st.stop()
    if upload is not None:
        with reference:
            with st.container(border=True):
                st.image(photo, caption="Your photo · model input", width=224)
    analysis_key = (name, upload.file_id if upload is not None else "sample-cherry")
    previous = st.session_state.get("last_analysis")
    try:
        if previous is None or previous[0] != analysis_key:
            with st.spinner(f"Analysing with {MODEL_NAMES[name]}… The first run may take longer while the model downloads."):
                result, overlay = predict(selected_model(name), photo)
            st.session_state.last_analysis = (analysis_key, result, overlay)
        else:
            _, result, overlay = previous
    except HTTPError as exc:
        override = os.getenv(name.upper() + "_URL")
        source = "Checkpoint URL" if override else "Kaggle"
        st.error(f"{source} returned HTTP {exc.code} while fetching the model.")
        if not override:
            has_credentials = bool(os.getenv("KAGGLE_API_TOKEN") or
                                   (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")))
            if not has_credentials:
                st.caption("Add your Kaggle credentials as top-level keys in Streamlit Secrets.")
            elif exc.code == 401:
                st.caption("Kaggle credentials were configured. Check that the token is valid and active.")
            elif exc.code in (403, 404):
                st.caption("Kaggle credentials were configured. Try downloading version 1 of "
                           "shahnawaz006/terravista-ckpts using the same Kaggle account.")
        st.stop()
    except (OSError, ValueError, RuntimeError) as exc:
        st.error("Could not load the verified checkpoint. Check the Kaggle access or local cache described in the README.")
        st.caption(f"Error type: {type(exc).__name__}")
        st.stop()
    crop, condition = result["class"].split("___", 1)
    label = f"{crop} · {condition}".replace("_", " ").strip()
    unsure = result["decision"] == "unsure"
    st.subheader("Analysis result")
    with st.container(border=True):
        decision, score = st.columns([1.4, 1])
        with decision:
            if unsure:
                st.warning("Unsure", icon=":material/help:")
                st.write("Try another close-up in even light with a plain background.")
            else:
                st.markdown("**Predicted class**")
                st.subheader(label)
        with score:
            st.metric("Calibrated confidence", f"{result['confidence']:.2%}")
            st.progress(result["confidence"])
            st.caption(f"Decision threshold · {THRESHOLD:.0%}")
    if not unsure:
        care = json.loads((ROOT / "data/class_care.json").read_text(encoding="utf-8"))
        st.subheader("Plant care" if condition == "healthy" else "Care if confirmed")
        st.write(care[result["class"]])

    with st.expander("Model evidence and heatmap", expanded=not unsure):
        st.write("Highest-scoring class", label)
        left, right = st.columns(2)
        left.image(photo, caption="Model input · 224 × 224", width="stretch")
        if ImageChops.difference(photo, overlay).getbbox():
            right.image(overlay, caption="Grad-CAM · orange shows stronger support", width="stretch")
        else:
            right.info("Grad-CAM found no visible spatial contrast for this class.")
        st.caption(f"{MODEL_NAMES[name]} · Seed 42 · Temperature {result['temperature']:.4f}")
        st.write("The heatmap can highlight background as well as leaf tissue. Confidence was calibrated on lab photos. Field images and unrelated objects can receive high scores.")

