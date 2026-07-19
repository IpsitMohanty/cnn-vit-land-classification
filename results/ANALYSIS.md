# Analysis

## TL;DR

Porting the same CNN from Keras to PyTorch produced a 28-point accuracy gap that had nothing to do with framework quality: `BatchNormalization`'s `momentum` parameter means opposite things in the two libraries, and Keras's default leaves running statistics unconverged in a short training budget. That's finding #1, and it's the portable one — it applies to anyone porting BatchNorm-using models between these two frameworks, regardless of task. Findings #2 and #3 are narrower: on this specific saturated benchmark, a CNN-ViT hybrid doesn't earn its added compute, and the Keras/PyTorch hybrid implementations in this lab aren't even the same size, so that comparison needs its own caveat.

---

## 1. Keras `BatchNormalization(momentum=0.99)` vs. PyTorch `BatchNorm(momentum=0.1)` — same name, inverted meaning

### Symptom

[`notebooks/04-keras-cnn-classifier.ipynb`](../notebooks/04-keras-cnn-classifier.ipynb) and [`notebooks/05-pytorch-cnn-classifier.ipynb`](../notebooks/05-pytorch-cnn-classifier.ipynb) train the same-family 6-conv-block CNN on the same dataset, same 3-epoch budget, same optimizer/learning rate. Keras finished at **70.1%** validation accuracy; PyTorch finished at **98.6%**. Too large a gap for epoch count alone, since epoch count was held constant.

The per-epoch training log for the Keras run was the first clue this wasn't simple undertraining:

| Epoch | Train acc | Train loss | Val acc | Val loss |
|---|---|---|---|---|
| 1 | 86.4% | 0.39 | 59.6% | **33.81** |
| 2 | 96.4% | 0.13 | 48.5% (below chance) | **10.28** |
| 3 | 97.6% | 0.08 | 70.2% | **1.38** |

Training accuracy climbs normally. Validation *loss* is catastrophically, unstably high relative to training loss, and validation accuracy briefly drops below the 50% random-guessing baseline. Healthy train-mode behavior next to incoherent eval-mode behavior is the signature of a train/inference-mode mismatch, not "needs more epochs."

### Isolating the cause: four experiments, one variable at a time

Every experiment below retrains the Keras conv backbone for 3 epochs with a **separate, augmentation-free validation generator** (see "the leak," below — this is held fixed across all four runs so it can't confound the head-depth/momentum comparison). Full scripts and raw JSON output in [`investigation/`](investigation/).

**Methodology note on what "accuracy" means in each row:** experiments 1-3 report the accuracy/precision/recall/F1/ROC-AUC of the model's state at the *end* of epoch 3 (no checkpointing) — all five metrics computed on that one consistent snapshot. Experiment 4 uses a `ModelCheckpoint(monitor='val_loss', save_best_only=True)` (matching how notebooks 04/05 actually handle epoch-to-epoch instability) and reports metrics for the *best* epoch by validation loss — also one consistent snapshot, just a different selection rule. Within each row, every column describes the same model; across rows 1-3 vs. 4, the selection rule differs, which is why experiment 4's number is the one quoted as "the fix" throughout this repo, not experiments 1-3's.

| # | Head | BN momentum | Epoch reported | Val accuracy | Val ROC-AUC | Val loss (all 3 epochs) |
|---|---|---|---|---|---|---|
| 1 | Original (6 blocks) | Keras default (0.99) | 3 of 3 (last) | 44.4% | **0.514** (chance) | 109.0 → 7.2 → 9.5 |
| 2 | Simplified (1 block, matches PyTorch) | Keras default (0.99) | 3 of 3 (last) | 53.8% | 0.754 | 4.8 → 2.1 → 3.8 |
| 3 | Simplified (1 block) | 0.9 (PyTorch-equivalent) | 3 of 3 (last) | 94.9% | 0.9998 | 0.076 → 0.032 → 0.139 |
| 4 | **Original (6 blocks)** | **0.9 (PyTorch-equivalent)** | 1 of 3 (best, checkpointed) | **98.17%** | **0.9996** | 0.049 → 0.077 → 0.275 |

Raw output: [`m2l1_anomaly_result.json`](investigation/m2l1_anomaly_result.json), [`..._result2_simplified_head.json`](investigation/m2l1_anomaly_result2_simplified_head.json), [`..._result3_bn_momentum.json`](investigation/m2l1_anomaly_result3_bn_momentum.json), [`..._result4_original_head_bn_fix.json`](investigation/m2l1_anomaly_result4_original_head_bn_fix.json).

**Experiment 1 vs. 4 is the controlled isolation: identical architecture (original 6-block head), the only variable changed is BatchNorm momentum, and the result flips from exact chance (0.514) to near-perfect (0.9996).** That is as close to definitive causal evidence as an ablation study gets. Momentum is not merely "the dominant factor" — holding everything else constant, it is **necessary and sufficient** to explain the collapse. (Note experiment 4 additionally benefits from best-checkpoint selection, which experiment 1 does not use — but experiment 1's own per-epoch trace never rises out of the 44-64% chance-adjacent range at *any* epoch, so checkpoint selection isn't what's carrying this comparison; a broken run doesn't have a good epoch to select.)

Head depth (experiment 1 vs. 2, both last-epoch, no checkpointing) looked like an independent contributor in isolation — going from 6 blocks to 1 block improved last-epoch AUC from 0.514 to 0.754 without touching momentum. Experiment 4 shows this was never a separate mechanism: with momentum fixed, the *original* 6-block head recovers fully (AUC 0.9996). **Head depth has no independent effect once momentum is corrected** — what experiment 2 actually demonstrated was that shortening the chain of mis-calibrated BatchNorm layers partially mitigates their compounding damage within a single (still momentum-broken) run, not that depth itself is an independent problem.

**The validation-augmentation leak is real but separate.** [`notebooks/04`](../notebooks/04-keras-cnn-classifier.ipynb) builds one `ImageDataGenerator` with heavy augmentation (rotation ±40°, width/height shift 0.2, shear 0.2, zoom 0.2, horizontal flip) and a `validation_split=0.2`, then calls `flow_from_directory` for both `subset="training"` and `subset="validation"` on that *same* generator — both subsets inherit the same augmentation policy, so the notebook's original "validation accuracy" was never measuring clean-image performance. This is fixed in all four experiments above (each uses a second, augmentation-free generator for validation) and is worth fixing on its own merits, but it is not what caused the 28-point gap — see the note in "what didn't cause this," below.

### The mechanism

Keras's `BatchNormalization` and PyTorch's `BatchNorm2d`/`BatchNorm1d` use the same parameter name, `momentum`, for opposite conventions:

- **Keras:** `running = momentum * running + (1 - momentum) * batch_stat`. Default `momentum=0.99` → each step moves the running statistic only **1%** toward the current batch's statistic.
- **PyTorch:** `running = (1 - momentum) * running + momentum * batch_stat`. Default `momentum=0.1` → each step moves **10%** toward the current batch's statistic.

A 10x difference in convergence speed. Over 3 epochs × 38 steps = 114 updates, Keras's default leaves the running statistic `1 - 0.99^114 ≈ 68%` converged; PyTorch's default reaches `1 - 0.9^114 ≈ 99.9998%`. "68% converged" doesn't sound catastrophic on its own — so why did experiment 1 collapse all the way to *exact chance* (AUC 0.514), not just "somewhat worse"?

Because the backbone has **6 sequential BatchNorm2d layers**, and each one's `gamma`/`beta` affine parameters are trained by gradient descent every step to expect a properly-normalized input — during *training*, BatchNorm always normalizes using the current batch's live statistics, regardless of how converged the running average is, so `gamma`/`beta` never "see" the immaturity problem. At *inference*, each layer instead normalizes using its own under-converged running mean/variance, producing a badly-scaled input that `gamma`/`beta` were never trained to handle. This mismatch happens independently at all 6 layers, and — critically — each layer's distorted output becomes the *next* layer's already-corrupted input. The errors don't add; they get re-processed and re-distorted by each subsequent mismatched normalization. Pushed through a long enough chain, this compounding is consistent with pushing the final `Dense(1)`→sigmoid into saturated, confidently-wrong outputs whose *direction* is driven by accumulated normalization noise rather than the actual input signal — which is exactly the observed signature: catastrophic loss (confident-and-wrong predictions carry a huge log-loss penalty) alongside chance-level AUC (ranking by predicted probability is no better than random when the confidence direction is decoupled from the true label).

This reading is supported by the dose-response pattern across all four experiments (more chained under-converged BatchNorm layers → larger loss, lower AUC; zero under-converged layers → normal loss, near-perfect AUC) and by the controlled experiment-1-vs-4 comparison. It is **not** independently instrumented — this analysis did not extract intermediate-layer activations to directly confirm sigmoid saturation. That would be the natural next diagnostic step if more precision were needed; the behavioral evidence (loss magnitude, AUC-at-chance, and the controlled ablation) is what the claim above rests on.

### Root cause, stated plainly

**Keras `BatchNormalization`'s default `momentum=0.99` adapts running statistics ~10x slower than PyTorch's default `momentum=0.1`.** In a short training budget — used throughout this project, deliberately, "in the interest of time" — this leaves Keras's BatchNorm layers' running statistics unconverged at evaluation time, while training-mode accuracy (which uses live batch statistics, never running averages) looks completely healthy. This is confirmed as necessary and sufficient by holding architecture constant and toggling only momentum (experiment 1 vs. 4). Head depth and the validation-augmentation leak are real, separate issues, but neither independently explains the anomaly.

This is a **generalizable Keras/PyTorch porting pitfall**, not a one-off mistake: any model ported between the two frameworks with default `BatchNormalization`/`BatchNorm` settings and a short training budget will show this exact pattern, regardless of task or architecture. The fix generalizes too: either train longer (more steps let Keras's slower default eventually converge), or set `momentum` explicitly to match the intended adaptation rate (Keras `momentum = 1 - <PyTorch momentum>`).

[`notebooks/04-keras-cnn-classifier.ipynb`](../notebooks/04-keras-cnn-classifier.ipynb) itself is left as originally executed (70.1%) — it's the authentic record of running IBM's lab code as given, and the anomaly is what triggered this investigation. Every place that number appears (in this repo or the README) carries a pointer back to this section.

**Caveat on the experiment design:** each of the four experiments above is a single run (n=1) at a single seed (7331) for 3 epochs — there is no variance estimate across seeds. The effect size here is far too large to be seed noise (chance-level AUC 0.514 vs. near-perfect 0.9996, holding architecture constant and changing one parameter) and the mechanism has an independent theoretical basis in BatchNorm's own definition, so the conclusion stands — but this was not verified across multiple seeds, and the caveat is stated rather than left for a reader to have to raise.

---

## 2. Does the CNN-ViT hybrid earn its complexity? No — on the evidence that's actually comparable.

**This conclusion rests entirely on the PyTorch pair.** See §3 for why the Keras-vs-PyTorch hybrid comparison is not a same-model comparison and is not used as corroborating evidence here.

| | CNN (PyTorch) | CNN-ViT (PyTorch) | Delta |
|---|---|---|---|
| Accuracy | 99.75% | 99.17% | **-0.58pp** |
| ROC-AUC | 0.99999 | 0.99985 | **-0.00014** |
| Params | 19.58M | 39.56M | **+102%** |
| Train time/epoch | 97.5s | 251.0s | **+157%** |

Both models trained and evaluated identically, no cross-framework confound. Adding the Vision Transformer stage made accuracy *worse* by over half a point, roughly doubled the parameter count, and cost 2.6x the wall-clock time per epoch. Per the brief: **for less than 1 percentage point of accuracy change — in the wrong direction — the hybrid costs 2x the parameters and 2.6x the training time. It doesn't earn its complexity on this task.**

That's a statement about *this task*, not about CNN-ViT hybrids generally (see §4): a 64×64, two-class, texturally well-separated satellite-tile problem is close to the easiest case for a plain CNN's local-feature bias, and gives the transformer's global-attention mechanism no long-range spatial dependency to exploit.

### The depth=12 ViT degradation

[`notebooks/08-pytorch-cnn-vit-hybrid.ipynb`](../notebooks/08-pytorch-cnn-vit-hybrid.ipynb) trained two ViT depths under an identical 5-epoch budget and identical learning rate (0.001): depth=3 finished at 99.4% validation accuracy; depth=12 was markedly noisier and finished at 91.3% (having peaked at 98.25% mid-training, epoch 3, before degrading). This is the expected consequence of using a shallow-network learning rate on a 4x-deeper transformer stack — the lab's own hyperparameter cheatsheet (reproduced in that notebook) states directly that depth-12 configurations need a lower learning rate (~0.0005) than depth-3/6 (~0.001) for training stability. The notebook used 0.001 for both. **Deeper ViT needs a lower learning rate than this lab's fixed hyperparameters provide — the degradation is the predicted outcome of that mismatch, not noise to discard.**

---

## 3. The Keras and PyTorch CNN-ViT hybrids are not the same model

Full table: [`model_comparison.md`](model_comparison.md).

| Model | Params (total) | Params (trainable) |
|---|---|---|
| CNN-ViT (Keras) | 168.60M | 151.13M |
| CNN-ViT (PyTorch) | 39.56M | 39.56M |

A 4.3x difference for a nominally identical architecture. Most of it traces to one line in the original lab code: `layers.MultiHeadAttention(num_heads, key_dim=embed_dim)` gives each of the 8 attention heads the *full* embedding dimension (1024) rather than the conventional `embed_dim // num_heads` (128) — inflating every Q/K/V/output projection roughly 8x. (The remaining difference is one extra transformer layer, 4 vs. 3, and a larger embed_dim, 1024 vs. 768.) The model trains and evaluates fine; it just isn't the same size as its PyTorch counterpart.

**Consequence for §2: the Keras CNN-ViT number (99.67%, nominally +1.5pp over the fixed Keras CNN's 98.17%) is not used as evidence that the hybrid earns its complexity.** It would be buying a rounding-error improvement with an 8.3x-larger model even if it were a fair comparison — and it isn't one, so it's presented here for completeness, not as a second data point supporting or contradicting §2's conclusion. §2's conclusion rests solely on the PyTorch pair, which is parameter-matched-enough (2x, not 4x+) and fully controlled.

---

## 4. Limitation: this benchmark is saturated

Every model in this project — the anomalous, unfixed Keras CNN excepted — lands between 98.17% and 99.9999% ROC-AUC on a binary classification task with a 1,200-image eval set. At that ceiling, a 0.5-percentage-point accuracy difference corresponds to roughly 6-12 individual images out of 1,200 — well within the range where a different random seed, train/val split, or one atypical batch would flip the ranking. **This benchmark cannot discriminate between architectures with the statistical confidence the headline numbers imply.**

The CNN-vs-CNN-ViT finding in §2 is directionally trustworthy — the hybrid's added cost with no accuracy benefit is a consistent pattern, not a single noisy number — but the exact percentage-point gaps in every table in this repository should be read as illustrative, not as precise measurements. A task with more label ambiguity, more classes, or a larger/noisier evaluation set would be a fairer testbed for whether a Vision Transformer's global-attention mechanism earns its cost. This 64×64, two-class, strongly locally-textured satellite-tile problem (crop rows vs. urban/water/forest texture) is close to the easiest case for a plain CNN and isn't a fair test of what Vision Transformers are for.
