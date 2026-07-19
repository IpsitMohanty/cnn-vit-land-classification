"""
Experiment 3 (rerun with checkpointing) -- simplified head (1 block,
matches PyTorch) + BatchNorm(momentum=0.9) + corrected validation split.

This rerun exists to make experiment 3 directly comparable to experiment 4
(original 6-block head + same momentum fix + same checkpoint protocol):
both scripts now report BOTH the last-epoch (end of training, no
checkpoint selection) and best-epoch (ModelCheckpoint, monitor='val_loss',
save_best_only=True) metrics, computed the same way, so a reader can see
the epoch-convention question explicitly rather than have it resolved
silently by picking one convention per experiment.
"""
import os
import random
import time
import json
from functools import partial

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 7331
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "images_dataSAT")
IMG_W, IMG_H, N_CHANNELS = 64, 64, 3
BATCH_SIZE = 128
LR = 0.001
N_EPOCHS = 3
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "m2l1_experiment3_best.keras")

BN = partial(BatchNormalization, momentum=0.9)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest", validation_split=0.2,
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=(IMG_W, IMG_H), batch_size=BATCH_SIZE,
    class_mode="binary", subset="training", seed=SEED,
)
validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH, target_size=(IMG_W, IMG_H), batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation", shuffle=False, seed=SEED,
)

# SIMPLIFIED head: matches PyTorch's single-block head
model = Sequential([
    Conv2D(32, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform(), input_shape=(IMG_W, IMG_H, N_CHANNELS)),
    MaxPooling2D(2, 2), BN(),
    Conv2D(64, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BN(),
    Conv2D(128, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BN(),
    Conv2D(256, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BN(),
    Conv2D(512, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BN(),
    Conv2D(1024, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BN(),
    GlobalAveragePooling2D(),
    Dense(2048, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["accuracy"])

checkpoint_cb = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

print(f"Training SIMPLIFIED head + BatchNorm(momentum=0.9) + corrected validation, {N_EPOCHS} epochs...")
start = time.time()
fit = model.fit(train_generator, epochs=N_EPOCHS, validation_data=validation_generator, callbacks=[checkpoint_cb], verbose=2)
elapsed = time.time() - start
print(f"Training completed in {elapsed:.1f}s")


def evaluate(m, label):
    validation_generator.reset()
    steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
    all_probs, all_labels = [], []
    for _ in range(steps):
        images, labels = next(validation_generator)
        probs = m.predict(images, verbose=0).flatten()
        all_probs.extend(probs)
        all_labels.extend(labels)
    all_preds = (np.array(all_probs) > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }
    print(f"{label}: {json.dumps(metrics, indent=2)}")
    return metrics


# last-epoch: the in-memory model as fit() left it, no checkpoint selection
last_epoch_metrics = evaluate(model, "last-epoch (end of training)")

# best-epoch: reload the checkpoint saved at lowest val_loss
best_model = load_model(CHECKPOINT_PATH)
best_epoch_metrics = evaluate(best_model, "best-epoch (lowest val_loss checkpoint)")

result = {
    "experiment": "corrected_val_simplified_head_bn_momentum_0.9_both_conventions",
    "epochs": N_EPOCHS,
    "train_time_seconds": elapsed,
    "val_accuracy_per_epoch": fit.history["val_accuracy"],
    "val_loss_per_epoch": fit.history["val_loss"],
    "best_epoch_by_val_loss": int(np.argmin(fit.history["val_loss"]) + 1),
    "last_epoch_metrics": last_epoch_metrics,
    "best_epoch_metrics": best_epoch_metrics,
}
print(json.dumps(result, indent=2))

with open(os.path.join(os.path.dirname(__file__), "m2l1_anomaly_result3_bn_momentum.json"), "w") as f:
    json.dump(result, f, indent=2)

os.remove(CHECKPOINT_PATH)
