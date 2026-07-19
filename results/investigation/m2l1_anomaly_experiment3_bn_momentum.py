"""
Third confirming experiment for the M2L1 anomaly -- the decisive one.

Experiment 1 (corrected validation pipeline only): val ROC-AUC 0.514 (chance).
Experiment 2 (+ simplified head matching PyTorch): val ROC-AUC 0.754 (better,
still far below PyTorch's ~1.0). Both experiments still use Keras
BatchNormalization's DEFAULT momentum=0.99.

Keras and PyTorch's BatchNorm "momentum" parameter use OPPOSITE conventions:
  - Keras:   new_running_stat = momentum * old + (1 - momentum) * batch_stat
             default momentum=0.99 -> only 1% adaptation toward batch stats per step
  - PyTorch: new_running_stat = (1 - momentum) * old + momentum * batch_stat
             default momentum=0.1  -> 10% adaptation toward batch stats per step

That's a 10x difference in how fast running statistics (used at inference/
validation time) converge. Over 3 epochs x 38 steps = 114 updates:
  - Keras default (0.99):  running stat has moved 1-0.99^114 = 68% of the way
    from initialization to the true data statistic -- NOT converged.
  - PyTorch default (0.1):  running stat has moved 1-0.9^114 = 99.9998% of the
    way -- fully converged.

Both the CNN backbone (6 BatchNorm2d layers, identical in both frameworks'
architectures) and the dense head inherit this default. This experiment sets
Keras's BatchNormalization(momentum=0.9) everywhere (the equivalent
adaptation rate to PyTorch's default) on top of experiment 2's simplified
head and corrected validation pipeline. If this closes the gap to PyTorch's
~98.6%, BatchNorm momentum is confirmed as the complete, dominant mechanism.
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

# BatchNorm with the PyTorch-equivalent adaptation rate (momentum=0.9 in
# Keras's "weight on old value" convention == PyTorch's momentum=0.1 default)
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

print(f"Training SIMPLIFIED head + corrected validation + BatchNorm(momentum=0.9), {N_EPOCHS} epochs...")
start = time.time()
fit = model.fit(train_generator, epochs=N_EPOCHS, validation_data=validation_generator, verbose=2)
elapsed = time.time() - start
print(f"Training completed in {elapsed:.1f}s")

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
    "experiment": "corrected_val_simplified_head_bn_momentum_0.9",
    "epochs": N_EPOCHS,
    "train_time_seconds": elapsed,
    "val_accuracy_per_epoch": fit.history["val_accuracy"],
    "val_loss_per_epoch": fit.history["val_loss"],
    "final_accuracy": accuracy_score(all_labels, all_preds),
    "final_precision": precision_score(all_labels, all_preds),
    "final_recall": recall_score(all_labels, all_preds),
    "final_f1": f1_score(all_labels, all_preds),
    "final_roc_auc": roc_auc_score(all_labels, all_probs),
}
print(json.dumps(result, indent=2))

with open(os.path.join(os.path.dirname(__file__), "m2l1_anomaly_result3_bn_momentum.json"), "w") as f:
    json.dump(result, f, indent=2)
