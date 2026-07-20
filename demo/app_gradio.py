"""
Gradio demo for the satellite land classification CNN (PyTorch).

Kept alongside app.py (the Streamlit/onnxruntime version this repo
actually deploys) as a zero-cost alternative for anyone who clones the
repo and wants a torch-based UI to run locally. Uses the small CNN
checkpoint from notebooks/05-pytorch-cnn-classifier.ipynb — NOT the
CNN-ViT hybrid, which saves as a 1.5-1.9GB file.
"""
import os

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from model_def import build_model

DEVICE = "cpu"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "cnn_state_dict.pth")
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


model = build_model()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    if image is None:
        return None
    image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


example_paths = []
examples_dir = os.path.join(os.path.dirname(__file__), "examples")
for cls in ("agri", "non_agri"):
    cls_dir = os.path.join(examples_dir, cls)
    if os.path.isdir(cls_dir):
        example_paths.extend(os.path.join(cls_dir, f) for f in sorted(os.listdir(cls_dir)))

with gr.Blocks(title="Satellite Land Classification") as demo:
    gr.Markdown("# Satellite Land Classification (CNN, PyTorch)")
    gr.Markdown(
        "Classifies a 64×64 satellite tile as **agricultural** or **non-agricultural** land. "
        "Click one of the example tiles below, or upload your own."
    )
    gr.Markdown(CAVEAT)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Satellite tile")
            gr.Examples(examples=example_paths, inputs=image_input, label="Example tiles (held out from training)")
        with gr.Column():
            output_label = gr.Label(num_top_classes=2, label="Prediction")

    image_input.change(fn=predict, inputs=image_input, outputs=output_label)

if __name__ == "__main__":
    demo.launch()
