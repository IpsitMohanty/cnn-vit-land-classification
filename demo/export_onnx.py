"""
Export cnn_state_dict.pth to ONNX for the Streamlit app's inference path
(onnxruntime, no torch). Run this after regenerating the checkpoint:

    python demo/export_onnx.py

The .pth remains the source of truth; cnn_model.onnx is a derived,
committed artifact so the Streamlit app doesn't need torch installed.
"""
import os

import torch

from model_def import build_model

DEVICE = "cpu"
HERE = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(HERE, "cnn_state_dict.pth")
ONNX_PATH = os.path.join(HERE, "cnn_model.onnx")


def main():
    model = build_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    dummy_input = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {ONNX_PATH}")


if __name__ == "__main__":
    main()
