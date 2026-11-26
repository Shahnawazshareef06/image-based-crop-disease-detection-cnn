import os
from urllib.error import HTTPError

import streamlit as st
from PIL import Image

from predict import MODEL_NAMES, THRESHOLD, load_model, predict, prepare

st.set_page_config(page_title="TerraVista · Leaf analysis", page_icon="🌿")
st.title("Leaf analysis")
st.write("Upload a close-up of one leaf to check it against 38 PlantVillage classes.")


@st.cache_resource(max_entries=1, show_spinner=False)
def selected_model(name):
    return load_model(name)


name = st.selectbox("Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
upload = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"], max_upload_size=12)
if upload:
    try:
        photo = prepare(upload)
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("Choose a readable JPEG, PNG or WebP photo under 24 megapixels.")
        st.stop()
    try:
        with st.spinner(f"Preparing {MODEL_NAMES[name]} and analysing the leaf…"):
            result, overlay = predict(selected_model(name), photo)
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
    if result["decision"] == "unsure":
        st.warning("Unsure")
        st.write("Highest-scoring class", label)
    else:
        st.subheader(label)
    st.metric("Calibrated confidence", f"{result['confidence']:.2%}")
    st.caption(f"Unsure below {THRESHOLD:.0%} · Seed 42 · Temperature {result['temperature']:.4f}")
    left, right = st.columns(2)
    left.image(photo, caption="Model input", width="stretch")
    right.image(overlay, caption="Grad-CAM for the highest-scoring class", width="stretch")
    st.caption("Orange highlights regions that contributed most strongly to the class score. "
               "Confidence was calibrated on lab images. Field photos can receive high scores even when the prediction is wrong.")
