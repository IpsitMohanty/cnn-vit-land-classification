"""
CNN architecture shared by the ONNX export script and the Gradio (torch)
demo. Not used by the Streamlit app, which runs inference through the
exported ONNX graph via onnxruntime instead of loading this module.
"""
import torch.nn as nn


def build_model() -> nn.Module:
    """Same architecture as notebooks/05-pytorch-cnn-classifier.ipynb."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
        nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256),
        nn.Conv2d(256, 512, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(512),
        nn.Conv2d(512, 1024, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.4),
        nn.Linear(2048, 2),
    )
