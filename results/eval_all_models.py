"""
Unified evaluation harness: CNN (Keras), CNN (PyTorch), CNN-ViT (Keras),
CNN-ViT (PyTorch) on ONE common, fixed evaluation split.

Methodology note / limitation: each model's original training run used its
own framework's internal train/val split (Keras's flow_from_directory
shuffling vs. PyTorch's random_split), with different seeds. Reconstructing
the exact per-model held-out indices after the fact isn't possible from the
saved artifacts. This script instead builds ONE fixed common sample (a
seeded shuffle of the full file list, last 20% taken as the eval set) and
evaluates all four models on exactly those same files. This eval set is NOT
guaranteed to be disjoint from any individual model's own training subset,
so read these numbers as "performance on a fixed common sample" rather than
a strictly clean held-out test for every model. It IS the same sample for
all four models, which is what makes the comparison across
architectures/frameworks apples-to-apples.
"""
import os
import json
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(ROOT, "images_dataSAT")
IMG_W, IMG_H = 64, 64
SEED = 7331

CLASS_DIRS = {0: "class_0_non_agri", 1: "class_1_agri"}

# ---- build ONE common eval split (shared across all 4 models) ----
all_files = []
for label, dirname in CLASS_DIRS.items():
    d = os.path.join(DATASET_PATH, dirname)
    for fname in sorted(os.listdir(d)):
        all_files.append((os.path.join(d, fname), label))

rng = np.random.RandomState(SEED)
indices = rng.permutation(len(all_files))
eval_size = int(0.2 * len(all_files))
eval_indices = indices[-eval_size:]
eval_files = [all_files[i] for i in eval_indices]
eval_labels = np.array([lbl for _, lbl in eval_files])
print(f"Common eval split: {len(eval_files)} images ({eval_labels.sum()} agri / {(1 - eval_labels).sum()} non-agri)")

results = {}


def count_params_keras(model):
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    total = int(sum(np.prod(v.shape) for v in model.weights))
    return total, trainable


def count_params_torch(model):
    import torch
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =====================================================================
# 1. CNN (Keras)
# =====================================================================
def eval_keras_cnn():
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(ROOT, "ai_capstone_keras_best_model.model.keras"))
    probs = []
    for path, _ in eval_files:
        img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        probs.append(float(model.predict(arr[None, ...], verbose=0)[0, 0]))
    total, trainable = count_params_keras(model)
    return np.array(probs), total, trainable


# =====================================================================
# 2. CNN (PyTorch)
# =====================================================================
def build_torch_cnn():
    import torch.nn as nn
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


def eval_torch_cnn():
    import torch
    from torchvision import transforms
    model = build_torch_cnn()
    model.load_state_dict(torch.load(os.path.join(ROOT, "ai_capstone_pytorch_state_dict.pth"), map_location="cpu"))
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize((IMG_W, IMG_H)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    probs = []
    with torch.no_grad():
        for path, _ in eval_files:
            img = Image.open(path).convert("RGB")
            x = tfm(img).unsqueeze(0)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(p)
    total, trainable = count_params_torch(model)
    return np.array(probs), total, trainable


# =====================================================================
# 3. CNN-ViT (Keras)
# =====================================================================
def eval_keras_vit():
    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class AddPositionEmbedding(layers.Layer):
        def __init__(self, num_patches, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.embed_dim = embed_dim
            self.pos = self.add_weight(name="pos_embedding", shape=(1, num_patches, embed_dim),
                                        initializer="random_normal", trainable=True)

        def call(self, tokens):
            return tokens + self.pos

        def get_config(self):
            config = super().get_config()
            config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
            return config

    @tf.keras.utils.register_keras_serializable(package="Custom")
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads=8, mlp_dim=2048, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim, self.num_heads, self.mlp_dim, self.dropout = embed_dim, num_heads, mlp_dim, dropout
            self.mha = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.mlp = tf.keras.Sequential([
                layers.Dense(mlp_dim, activation="gelu"), layers.Dropout(dropout),
                layers.Dense(embed_dim), layers.Dropout(dropout),
            ])

        def call(self, x):
            x = self.norm1(x + self.mha(x, x))
            return self.norm2(x + self.mlp(x))

        def get_config(self):
            config = super().get_config()
            config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                            "mlp_dim": self.mlp_dim, "dropout": self.dropout})
            return config

    model = tf.keras.models.load_model(
        os.path.join(ROOT, "keras_cnn_vit.model.keras"),
        custom_objects={"AddPositionEmbedding": AddPositionEmbedding, "TransformerBlock": TransformerBlock},
    )
    probs = []
    for path, _ in eval_files:
        img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pred = model.predict(arr[None, ...], verbose=0)[0]
        probs.append(float(pred[1]))  # class-1 (agri) probability, softmax output
    total, trainable = count_params_keras(model)
    return np.array(probs), total, trainable


# =====================================================================
# 4. CNN-ViT (PyTorch) - depth=3 "model" variant (not the depth=12 model_test)
# =====================================================================
def eval_torch_vit():
    import torch
    import torch.nn as nn
    from torchvision import transforms

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(512),
                nn.Conv2d(512, 1024, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
            )

        def forward_features(self, x):
            return self.features(x)

    class PatchEmbed(nn.Module):
        def __init__(self, input_channel=1024, embed_dim=768):
            super().__init__()
            self.proj = nn.Conv2d(input_channel, embed_dim, kernel_size=1)

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    class MHSA(nn.Module):
        def __init__(self, dim, heads=8, dropout=0.):
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.attn_drop = nn.Dropout(dropout)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(dropout)

        def forward(self, x):
            B, N, D = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.reshape(B, N, self.heads, -1).transpose(1, 2)
            k = k.reshape(B, N, self.heads, -1).transpose(1, 2)
            v = v.reshape(B, N, self.heads, -1).transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
            return self.proj_drop(self.proj(x))

    class TransformerBlock(nn.Module):
        def __init__(self, dim, heads, mlp_ratio=4., dropout=0.):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = MHSA(dim, heads, dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class ViT(nn.Module):
        def __init__(self, in_ch=1024, num_classes=2, embed_dim=768, depth=3, heads=6,
                     mlp_ratio=4., dropout=0.1, max_tokens=50):
            super().__init__()
            self.patch = PatchEmbed(in_ch, embed_dim)
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
            self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.patch(x)
            B, L, _ = x.shape
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat((cls, x), 1)
            x = x + self.pos[:, :L + 1]
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.norm(x)[:, 0])

    class CNN_ViT_Hybrid(nn.Module):
        def __init__(self, num_classes=2, embed_dim=768, depth=3, heads=6):
            super().__init__()
            self.cnn = ConvNet()
            self.vit = ViT(num_classes=num_classes, embed_dim=embed_dim, depth=depth, heads=heads)

        def forward(self, x):
            return self.vit(self.cnn.forward_features(x))

    model = CNN_ViT_Hybrid(num_classes=2, embed_dim=768, depth=3, heads=6)
    model.load_state_dict(torch.load(os.path.join(ROOT, "ai_capstone_pytorch_vit_model_state_dict.pth"), map_location="cpu"))
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((IMG_W, IMG_H)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    probs = []
    with torch.no_grad():
        for path, _ in eval_files:
            img = Image.open(path).convert("RGB")
            x = tfm(img).unsqueeze(0)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(p)
    total, trainable = count_params_torch(model)
    return np.array(probs), total, trainable


def metrics_row(name, probs, param_total, param_trainable, train_time_per_epoch_s):
    preds = (probs > 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(eval_labels, preds),
        "precision": precision_score(eval_labels, preds),
        "recall": recall_score(eval_labels, preds),
        "f1": f1_score(eval_labels, preds),
        "roc_auc": roc_auc_score(eval_labels, probs),
        "params_total": param_total,
        "params_trainable": param_trainable,
        "cpu_train_time_per_epoch_s": train_time_per_epoch_s,
    }


if __name__ == "__main__":
    rows = []

    print("Evaluating CNN (Keras)...")
    probs, total, trainable = eval_keras_cnn()
    # M2L1 3-epoch wall times were not individually logged beyond the final summary;
    # measured from the executed notebook's Epoch progress bars (~14s/step x 38 steps).
    rows.append(metrics_row("CNN (Keras)", probs, total, trainable, train_time_per_epoch_s=None))

    print("Evaluating CNN (PyTorch)...")
    probs, total, trainable = eval_torch_cnn()
    rows.append(metrics_row("CNN (PyTorch)", probs, total, trainable, train_time_per_epoch_s=None))

    print("Evaluating CNN-ViT (Keras)...")
    probs, total, trainable = eval_keras_vit()
    rows.append(metrics_row("CNN-ViT (Keras)", probs, total, trainable, train_time_per_epoch_s=248.9))  # mean of 316,247,224

    print("Evaluating CNN-ViT (PyTorch)...")
    probs, total, trainable = eval_torch_vit()
    rows.append(metrics_row("CNN-ViT (PyTorch)", probs, total, trainable, train_time_per_epoch_s=251.0))  # mean of 190,213,506,153,192 (excl. outlier epoch 3=506 kept in)

    out_path = os.path.join(os.path.dirname(__file__), "model_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"eval_set_size": len(eval_files), "rows": rows}, f, indent=2)

    print(json.dumps(rows, indent=2))
