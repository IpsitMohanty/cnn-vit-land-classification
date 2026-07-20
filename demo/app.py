"""
Streamlit demo for the satellite land classification CNN.

Runs inference through the ONNX export (demo/cnn_model.onnx) via
onnxruntime -- no torch/torchvision in this file, since Streamlit Cloud's
free tier has a tight memory ceiling and the torch CPU wheel alone is
~200MB. demo/export_onnx.py regenerates cnn_model.onnx from
cnn_state_dict.pth (the source of truth) if the checkpoint changes; see
that script's verification of ONNX-vs-torch output parity.

The Gradio/torch version of this demo lives in app_gradio.py.
"""
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(__file__)
ONNX_PATH = os.path.join(HERE, "cnn_model.onnx")
EXAMPLES_DIR = os.path.join(HERE, "examples")
CLASS_NAMES = ["non-agricultural", "agricultural"]

CAVEAT = (
    "⚠️ **Read before trusting a prediction:** this model was trained on a single "
    "distribution of 64×64 Sentinel-2 satellite tiles, evenly split between "
    "agricultural and non-agricultural land, and scores ~99% accuracy *on that "
    "benchmark*. It has not seen aerial photos, drone imagery, ground-level photos, "
    "or satellite tiles from sensors/resolutions/regions outside its training data. "
    "Predictions on anything else — including the upload box below — are not "
    "reliable. The bundled examples are real held-out tiles from the training "
    "distribution; they are the fair test of this model, not a demonstration of "
    "general-purpose satellite image understanding."
)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

EXAMPLE_PATHS = {}
for _cls in ("agri", "non_agri"):
    _cls_dir = os.path.join(EXAMPLES_DIR, _cls)
    if os.path.isdir(_cls_dir):
        for _fname in sorted(os.listdir(_cls_dir)):
            EXAMPLE_PATHS[_fname] = os.path.join(_cls_dir, _fname)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((64, 64), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr[np.newaxis, ...].astype(np.float32)


def predict(image: Image.Image):
    if image is None:
        return None
    x = preprocess(image)
    logits = _session.run(None, {"input": x})[0].squeeze(0)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def main():
    import streamlit as st

    st.set_page_config(page_title="Satellite Land Classification", page_icon="\U0001f6f0️")
    st.title("Satellite Land Classification (CNN, ONNX)")
    st.markdown(
        "Classifies a 64×64 satellite tile as **agricultural** or **non-agricultural** land. "
        "Pick one of the example tiles below, or upload your own."
    )
    st.warning(CAVEAT)

    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
        st.session_state.selected_name = None

    st.subheader("Example tiles (held out from training)")
    cols = st.columns(len(EXAMPLE_PATHS))
    for col, (name, path) in zip(cols, EXAMPLE_PATHS.items()):
        with col:
            st.image(path, caption=name, width="stretch")
            if st.button("Use this tile", key=f"example_{name}"):
                st.session_state.selected_image = Image.open(path)
                st.session_state.selected_name = name

    st.subheader("Or upload your own")
    uploaded = st.file_uploader("Satellite tile", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state.selected_image = Image.open(uploaded)
        st.session_state.selected_name = uploaded.name

    image = st.session_state.selected_image
    if image is None:
        st.info("Pick an example tile or upload an image to get a prediction.")
        return

    st.subheader("Prediction")
    st.image(image, caption=st.session_state.selected_name, width=200)
    result = predict(image)
    predicted = max(result, key=result.get)
    st.markdown(f"**{predicted}**")
    for label, confidence in sorted(result.items(), key=lambda kv: -kv[1]):
        st.progress(confidence, text=f"{label}: {confidence:.1%}")


if __name__ == "__main__":
    main()
