"""
Fourth confirming experiment -- cleanly separates "does BatchNorm momentum
alone explain the collapse" from "does head depth independently matter."

Prior three experiments, ROC-AUC and val_loss:
  1. Original head (6 blocks), momentum=0.99 (default): AUC 0.514 (chance), loss 109->7.2->9.5
  2. Simplified head (1 block), momentum=0.99 (default): AUC 0.754,          loss 4.8->2.1->3.8
  3. Simplified head (1 block), momentum=0.9  (fixed):   AUC 0.9998,        loss 0.076->0.032->0.139

Experiments 1 vs 2 changed head depth only (both at default momentum) and
went from chance-level collapse to partial function. Experiment 3 then
fixed momentum on top of the simplified head and reached near-parity with
PyTorch. What's untested: the ORIGINAL 6-block head with momentum FIXED.
This is the cleanest possible isolation:
  - If this reaches ~98% too, BatchNorm momentum is confirmed as the
    COMPLETE explanation, and head depth was never an independent cause.
  - If this plateaus well below ~98%, head depth has a real, independent
    negative effect beyond amplifying momentum immaturity.

This version adds a ModelCheckpoint (monitor='val_loss', mode='min',
save_best_only=True) -- matching how notebooks/04 and notebooks/05 handle
epoch-to-epoch instability in a short training budget -- and evaluates the
RELOADED BEST checkpoint with full sklearn metrics, rather than reporting
whatever the last epoch happens to land on (which may not be the best
epoch and would make the "accuracy" number and the "precision/recall/F1"
numbers reflect two different models if taken from different points in
the training log).
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
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "m2l1_experiment4_best.keras")

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

# ORIGINAL 6-block head, but with the momentum fix applied
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
    Dense(64, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(128, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(256, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(512, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(1024, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(2048, activation="relu", kernel_initializer=HeUniform()), BN(), Dropout(0.4),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["accuracy"])

checkpoint_cb = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

print(f"Training ORIGINAL 6-block head + BatchNorm(momentum=0.9) + corrected validation, {N_EPOCHS} epochs...")
start = time.time()
fit = model.fit(train_generator, epochs=N_EPOCHS, validation_data=validation_generator, callbacks=[checkpoint_cb], verbose=2)
elapsed = time.time() - start
print(f"Training completed in {elapsed:.1f}s")

# Reload the BEST checkpoint (lowest val_loss) and evaluate it with full metrics,
# so accuracy/precision/recall/F1/ROC-AUC all describe the same model snapshot.
best_model = load_model(CHECKPOINT_PATH)
validation_generator.reset()
steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
all_probs, all_labels = [], []
for _ in range(steps):
    images, labels = next(validation_generator)
    probs = best_model.predict(images, verbose=0).flatten()
    all_probs.extend(probs)
    all_labels.extend(labels)
all_preds = (np.array(all_probs) > 0.5).astype(int)

result = {
    "experiment": "corrected_val_original_head_bn_momentum_0.9_best_checkpoint",
    "epochs": N_EPOCHS,
    "train_time_seconds": elapsed,
    "val_accuracy_per_epoch": fit.history["val_accuracy"],
    "val_loss_per_epoch": fit.history["val_loss"],
    "best_epoch_by_val_loss": int(np.argmin(fit.history["val_loss"]) + 1),
    "final_accuracy": accuracy_score(all_labels, all_preds),
    "final_precision": precision_score(all_labels, all_preds),
    "final_recall": recall_score(all_labels, all_preds),
    "final_f1": f1_score(all_labels, all_preds),
    "final_roc_auc": roc_auc_score(all_labels, all_probs),
}
print(json.dumps(result, indent=2))

with open(os.path.join(os.path.dirname(__file__), "m2l1_anomaly_result4_original_head_bn_fix.json"), "w") as f:
    json.dump(result, f, indent=2)

os.remove(CHECKPOINT_PATH)
