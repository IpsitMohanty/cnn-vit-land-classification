# Four-model comparison

All four models evaluated on one common, fixed sample: a seeded shuffle of the full 6,000-image dataset, last 20% held out (1,200 images: 596 agricultural / 604 non-agricultural). See [`eval_all_models.py`](eval_all_models.py) for the exact methodology and its limitation note (this common split is not guaranteed disjoint from any individual model's own original training subset — see below).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params (total) | Params (trainable) | CPU train time / epoch |
|---|---|---|---|---|---|---|---|---|
| CNN (Keras) †— anomalous, not a real result | 72.9% | 85.4% | 54.9% | 66.8% | 0.756 | 20.35M | 20.34M | ~470s |
| **CNN (Keras, `momentum=0.9` fix)** — same architecture as the row above | **98.17%** | **97.07%** | **99.33%** | **98.19%** | **0.9996** | 20.35M | 20.34M | 277s |
| CNN (PyTorch) | 99.75% | 99.66% | 99.83% | 99.75% | 0.99999 | 19.58M | 19.58M | 97.5s |
| CNN-ViT (Keras) ‡— not the same size as the row below | 99.67% | 100.0% | 99.33% | 99.66% | 0.99993 | 168.60M | 151.13M | 248.9s |
| CNN-ViT (PyTorch) ‡— not the same size as the row above | 99.17% | 99.16% | 99.16% | 99.16% | 0.99985 | 39.56M | 39.56M | 251.0s |

† **The first row is the BatchNorm-momentum anomaly, not a real Keras CNN result** — see below and [`ANALYSIS.md`](ANALYSIS.md) §1. The row directly below it is the identical architecture with only `BatchNormalization(momentum=0.9)` changed, evaluated on a saved best-checkpoint (lowest val_loss, epoch 1 of 3) so every column reflects the same model snapshot — no mixing of a mid-training accuracy peak with a different epoch's other metrics. This fixed row, not the anomalous one, is the fair "plain Keras CNN" data point for any comparison.

‡ **The two CNN-ViT rows are not a controlled comparison** — the Keras and PyTorch CNN-ViT hybrids differ 4.3x in parameter count for a nominally identical architecture. See below and [`ANALYSIS.md`](ANALYSIS.md) §3 before drawing any conclusion from comparing them directly.

Raw data: [`model_comparison.json`](model_comparison.json), [`investigation/m2l1_anomaly_result4_original_head_bn_fix.json`](investigation/m2l1_anomaly_result4_original_head_bn_fix.json) (fixed row).

## Reading this table correctly

**The first Keras CNN row is not a fair architecture data point — it's the anomaly.** See [`ANALYSIS.md`](ANALYSIS.md) §1 for the full root-cause investigation (four controlled experiments, raw output in `investigation/`). The root cause is Keras `BatchNormalization`'s default `momentum=0.99` vs. PyTorch's `momentum=0.1` — same parameter name, inverted convention, ~10x difference in running-statistic convergence speed. Use the PyTorch CNN row, or the fixed-Keras row directly below the anomaly, as the true "plain CNN" baseline for any architecture comparison.

**The Keras and PyTorch CNN-ViT hybrids are not parameter-matched.** The Keras hybrid (depth=4, heads=8, embed_dim=1024) has 168.6M total params against PyTorch's 39.56M (depth=3, heads=6, embed_dim=768) — a 4.3x difference for nominally "the same" architecture. Most of that gap is a single line in the original lab code: `layers.MultiHeadAttention(num_heads, key_dim=embed_dim)` gives each attention head the *full* embedding dimension (1024) instead of the conventional `embed_dim // num_heads` (128), inflating every Q/K/V/output projection ~8x. This isn't a bug exactly — the model trains and evaluates fine — but it means "Keras CNN-ViT" and "PyTorch CNN-ViT" are two different-sized models being asked the same question, not a controlled comparison of the same architecture across frameworks.

**Eval-split caveat.** Each model's original training run used its own framework's internal train/val split (Keras's `flow_from_directory` shuffling vs. PyTorch's `random_split`), with different seeds — the exact held-out indices from each original run can't be reconstructed from the saved checkpoints. The split used here is fixed and identical across all four models (which is what makes the *relative* comparison valid), but it is not guaranteed disjoint from any individual model's own training data. Treat these as "performance on a fixed common sample," not a certified clean holdout for every row.

**Training time is CPU-only, single machine, not controlled for implementation efficiency.** It reflects wall-clock time on this specific setup (see the main README for hardware/environment notes), useful for relative "how much slower is the hybrid" comparisons within this environment, not as an absolute benchmark.
