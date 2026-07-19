"""
Confirming experiment for the M2L1 Keras-vs-PyTorch CNN accuracy anomaly.

Original result: Keras CNN (notebooks/04) scored 70.1% val accuracy after 3
epochs; PyTorch CNN (notebooks/05), same architecture family, same epoch
budget, scored 98.6%. Two candidate root causes were identified by reading
the two notebooks' data pipelines and model definitions side by side:

  (A) Validation-augmentation leak: notebooks/04 builds ONE
      ImageDataGenerator with heavy augmentation (rotation=40,
      width/height_shift=0.2, shear=0.2, zoom=0.2, horizontal_flip) and a
      validation_split=0.2, then calls flow_from_directory(subset="training")
      and flow_from_directory(subset="validation") on that SAME generator.
      Both subsets inherit the same augmentation policy, so the reported
      "validation accuracy" is measured on heavily distorted images.
      notebooks/05 (PyTorch) uses a separate, augmentation-free val_transform
      for its validation split.

  (B) Classifier head depth/regularization asymmetry: the Keras head is 6
      stacked (Dense -> BatchNorm -> Dropout(0.4)) blocks growing
      64->128->256->512->1024->2048 before the output layer (2.88M params
      across 7 dense layers). The PyTorch head is a single
      (Dense(2048) -> BatchNorm -> Dropout(0.4)) block before the output
      (2.11M params across 2 dense layers). Comparable total parameter
      count, but 6x the sequential dropout gates to converge through in
      the same 3-epoch budget.

This script isolates (A) by retraining the ORIGINAL 6-block head with a
corrected validation pipeline (separate, augmentation-free generator for
validation, same split), same 3 epochs, same seed, same hyperparameters as
notebooks/04. If validation accuracy recovers close to the PyTorch number,
(A) is the dominant cause. If it stays low, (B) dominates.
"""
import os
import random
import time
import json

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import HeUniform
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

# ---- corrected data pipeline: augmentation on TRAIN only ----
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
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

# ---- ORIGINAL 6-block head architecture, unchanged from notebooks/04 ----
model = Sequential([
    Conv2D(32, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform(), input_shape=(IMG_W, IMG_H, N_CHANNELS)),
    MaxPooling2D(2, 2), BatchNormalization(),
    Conv2D(64, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BatchNormalization(),
    Conv2D(128, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BatchNormalization(),
    Conv2D(256, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BatchNormalization(),
    Conv2D(512, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BatchNormalization(),
    Conv2D(1024, (5, 5), activation="relu", padding="same", strides=(1, 1), kernel_initializer=HeUniform()),
    MaxPooling2D(2, 2), BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(64, activation="relu", kernel_initializer=HeUniform()), BatchNormalization(), Dropout(0.4),
    Dense(128, activation="relu", kernel_initializer=HeUniform()), BatchNormalization(), Dropout(0.4),
    Dense(256, activation="relu", kernel_initializer=HeUniform()), BatchNormalization(), Dropout(0.4),
    Dense(512, activation="relu", kernel_initializer=HeUniform()), BatchNormalization(), Dropout(0.4),
    Dense(1024, activation="relu", kernel_initializer=HeUniform()), BatchNormalization(), Dropout(0.4),
    Dense(2048, activation="relu", kernel_initializer=HeUniform()), BatchNormalization(), Dropout(0.4),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["accuracy"])

print(f"Training ORIGINAL 6-block-head architecture with CORRECTED (unaugmented) validation split, {N_EPOCHS} epochs...")
start = time.time()
fit = model.fit(train_generator, epochs=N_EPOCHS, validation_data=validation_generator, verbose=2)
elapsed = time.time() - start
print(f"Training completed in {elapsed:.1f}s")

# ---- clean evaluation with full metrics, matching the eval harness used elsewhere ----
validation_generator.reset()
steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
all_probs, all_labels = [], []
for _ in range(steps):
    images, labels = next(validation_generator)
    probs = model.predict(images, verbose=0).flatten()
    all_probs.extend(probs)
    all_labels.extend(labels)
all_preds = (np.array(all_probs) > 0.5).astype(int)

result = {
    "experiment": "corrected_validation_pipeline_original_head",
    "epochs": N_EPOCHS,
    "train_time_seconds": elapsed,
    "val_accuracy_per_epoch": fit.history["val_accuracy"],
    "final_accuracy": accuracy_score(all_labels, all_preds),
    "final_precision": precision_score(all_labels, all_preds),
    "final_recall": recall_score(all_labels, all_preds),
    "final_f1": f1_score(all_labels, all_preds),
    "final_roc_auc": roc_auc_score(all_labels, all_probs),
}
print(json.dumps(result, indent=2))

with open(os.path.join(os.path.dirname(__file__), "m2l1_anomaly_result.json"), "w") as f:
    json.dump(result, f, indent=2)
