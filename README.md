# Satellite Land Classification: CNN → CNN-ViT Hybrid

[![tests](https://github.com/IpsitMohanty/cnn-vit-land-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/IpsitMohanty/cnn-vit-land-classification/actions/workflows/tests.yml)

**Demo:** not yet deployed. Run it locally in three commands — see [`demo/`](demo/) (`pip install -r demo/requirements.txt && python demo/app.py`). Upload a satellite tile or click a bundled example for a live agricultural/non-agricultural prediction. Built for Hugging Face Spaces' free tier; deployment instructions are in [`demo/README.md`](demo/README.md).

## What this is

This is a completed capstone from IBM's *AI Engineering* Skills Network courses (Coursera). The brief IBM supplies is a scenario, not a real engagement: a fictional fertilizer company — named "NutriSphere Agritech" in IBM's lab materials — wants to assess agricultural land coverage in a new region before making market-expansion and sales-territory decisions there. Nothing in this repo was built for an actual client; the company and the business need are IBM's fictional framing for the exercise.

The underlying task is a straightforward binary classification problem: label ~6,000 64×64 satellite image tiles as agricultural or non-agricultural land. IBM's curriculum has you build the same solution twice, once in Keras and once in PyTorch, across four stages: loading and augmenting the tile data, training a CNN classifier, extending it into a CNN-ViT hybrid via transfer learning, and comparing all four resulting models. The [notebook sequence](#notebook-sequence) below maps each of the 9 notebooks onto these stages.

The lab scaffolding, dataset, architectures, and problem framing are IBM's. What's mine is everything below the notebooks: actually running all of it end-to-end on a local CPU box, finding and root-causing the BatchNorm anomaly in Finding #1, the controlled ablation that confirms it, the honest four-model comparison, and the demo app.

## Finding #1: `BatchNormalization(momentum=0.99)` and `BatchNorm(momentum=0.1)` are not the same default

A Keras CNN and a PyTorch CNN, trained on identical data with an identical 3-epoch budget, landed 28 accuracy points apart (70% vs. 98.6% — see the 70% result, with context, in [`notebooks/04`](notebooks/04-keras-cnn-classifier.ipynb)). The cause: **Keras's `BatchNormalization` and PyTorch's `BatchNorm` use the same parameter name, `momentum`, for opposite conventions** — Keras's default (`0.99`) adapts running statistics ~10x slower than PyTorch's default (`0.1`). In a short training budget, Keras's running statistics never converge, so training-mode accuracy looks healthy while inference-mode (validation) accuracy collapses — briefly to *exact chance level* in the worst configuration tested.

A controlled ablation (four experiments, changing one variable at a time, holding architecture constant across the decisive pair) confirms this is not a minor contributing factor but the complete explanation: fixing only `momentum` — same architecture, nothing else changed — recovers 98-99% validation accuracy (the exact figure varies run-to-run at this scale; see the caveat in §1), matching PyTorch. This is a **portable, generalizable finding**: it applies to anyone porting a BatchNorm-using model between these two frameworks, regardless of task or architecture. Full investigation, with the mechanism, the math, the ablation table, and why the collapse hit exact chance level rather than merely "worse": [`results/ANALYSIS.md`](results/ANALYSIS.md) §1.

## Finding #2: the CNN-ViT hybrid doesn't earn its complexity — on this task

On the clean, cross-framework-confound-free PyTorch comparison (plain CNN vs. CNN-ViT hybrid, same training/eval pipeline): adding the Vision Transformer stage made accuracy **0.58 percentage points worse** (99.75% → 99.17%) while roughly **doubling the parameter count** and costing **2.6x the training time per epoch**. This conclusion rests on the PyTorch pair only — see Finding #3 for why the Keras/PyTorch hybrid numbers can't be compared to each other directly. Also documented: a depth=12 ViT variant that degrades mid-training, which is the predicted consequence of using a depth-3 learning rate on a 4x-deeper transformer, not a stray bad number. Full breakdown: [`results/ANALYSIS.md`](results/ANALYSIS.md) §2.

## Finding #3: the Keras and PyTorch CNN-ViT hybrids in this project are not the same size

168.6M params (Keras) vs. 39.6M params (PyTorch) for a nominally identical hybrid architecture — a 4.3x difference, mostly from one line in the lab code (`MultiHeadAttention(key_dim=embed_dim)`, giving each attention head the full embedding dimension instead of `embed_dim / num_heads`, inflating attention params ~8x). **Wherever this repo shows Keras-vs-PyTorch CNN-ViT numbers side by side, they describe two different-sized models, not a controlled comparison** — that's why Finding #2's cost/benefit conclusion is drawn only from the PyTorch pair. Details: [`results/ANALYSIS.md`](results/ANALYSIS.md) §3.

## Limitation: this benchmark is saturated

The underlying task is a two-class split (agricultural vs. non-agricultural) over ~6,000 tiles — a small, simple benchmark that every model here saturates. Every model except the anomalous, unfixed Keras CNN scores between 0.9998 and 0.99999 ROC-AUC on the 1,200-image held-out eval set. At that ceiling, accuracy alone swings by nearly 5 points depending on which epoch of an otherwise-identical run you evaluate (§1) — well within the noise of a different random seed or split. Finding #2's *direction* (hybrid costs more, doesn't help) is a consistent pattern across both frameworks and is trustworthy; the exact percentage-point gaps in every table here are illustrative, not precise measurements. Details: [`results/ANALYSIS.md`](results/ANALYSIS.md) §4.

## What's here

| Path | What it is |
|---|---|
| [`notebooks/`](notebooks/) | The 9-notebook pipeline: data loading → CNN (Keras/PyTorch) → CNN-ViT hybrid (Keras/PyTorch) → final comparison. Executed end-to-end, results inline. |
| [`results/`](results/) | The actual deliverable: root-cause analysis, four-model comparison, and the four confirming experiments behind Finding #1. |
| [`demo/`](demo/) | Gradio app, deployable to Hugging Face Spaces. |

## Notebook sequence

| # | Curriculum stage | Notebook | What it covers |
|---|---|---|---|
| 1 | Data loading & augmentation | [`01-data-loading-memory-vs-generator`](notebooks/01-data-loading-memory-vs-generator.ipynb) | Sequential (lazy) vs. bulk in-memory image loading |
| 2 | Data loading & augmentation | [`02-keras-data-pipeline`](notebooks/02-keras-data-pipeline.ipynb) | Hand-written Keras generator vs. `tf.data` (`.cache()`/`.prefetch()`) |
| 3 | Data loading & augmentation | [`03-pytorch-data-pipeline`](notebooks/03-pytorch-data-pipeline.ipynb) | Custom `Dataset` vs. `ImageFolder`, both via `DataLoader` |
| 4 | CNN classifiers | [`04-keras-cnn-classifier`](notebooks/04-keras-cnn-classifier.ipynb) | 38-layer CNN in Keras. **Scores 70% here — this is the BatchNorm-momentum anomaly, not a real result. See the note in the notebook and Finding #1 above / `results/ANALYSIS.md` §1.** |
| 5 | CNN classifiers | [`05-pytorch-cnn-classifier`](notebooks/05-pytorch-cnn-classifier.ipynb) | Equivalent CNN in PyTorch (98.6%, unaffected by the Keras-specific bug) |
| 6 | CNN classifiers | [`06-keras-vs-pytorch-cnn-comparison`](notebooks/06-keras-vs-pytorch-cnn-comparison.ipynb) | Evaluation on IBM's 20-epoch reference models (accuracy/precision/recall/F1/ROC-AUC/confusion matrices) |
| 7 | CNN-ViT hybrids (transfer learning) | [`07-keras-cnn-vit-hybrid`](notebooks/07-keras-cnn-vit-hybrid.ipynb) | CNN backbone + custom Vision Transformer encoder, in Keras — see Finding #3 for why its param count isn't comparable to notebook 8's |
| 8 | CNN-ViT hybrids (transfer learning) | [`08-pytorch-cnn-vit-hybrid`](notebooks/08-pytorch-cnn-vit-hybrid.ipynb) | Same hybrid in PyTorch, plus the depth=3-vs-12 comparison referenced in Finding #2 |
| 9 | Final metric comparison | [`09-final-cnn-vit-evaluation`](notebooks/09-final-cnn-vit-evaluation.ipynb) | IBM's 20-epoch reference CNN-ViT models, both frameworks, full comparative evaluation |

## Environment compatibility fixes

These notebooks were written for IBM's cloud-based Skills Network Labs (Linux, pre-provisioned packages). Running them locally on Windows surfaced several environment-specific failures — none of which are bugs in the modeling code itself:

| Symptom | Root cause | Fix |
|---|---|---|
| `pip install` / dataset downloads fail with `SSLCertVerificationError` | Norton Antivirus performs SSL inspection; its injected root CA isn't in a fresh venv's `certifi` bundle | Appended Norton's CA cert to the venv's `certifi` bundle |
| `FileNotFoundError` on `skillsnetwork.prepare()` | The `skillsnetwork` package hardcodes a `/tmp/...` download path, which doesn't exist on Windows | Created `D:\tmp` |
| `DataLoader` hangs indefinitely with `num_workers>0` | Windows uses spawn-based multiprocessing; workers can't pickle the interactive kernel context | Set `num_workers=0` throughout |
| `SyntaxError: f-string: unmatched '['` | Nested same-quote f-strings (`f"...{d["key"]}..."`) are only valid on Python 3.12+ (PEP 701); TensorFlow/PyTorch require Python 3.11 | Rewrote nested string literals to use single quotes |
| `!wget` cells fail silently | No `wget.exe` on Windows; Jupyter's `!` magic uses `cmd.exe`, not the PowerShell `wget` alias | Replaced with `httpx`-based Python downloads |
| `httpx.ReadTimeout` downloading large (100MB+) pretrained model files | Default 5s client timeout, too short for large files | Set `timeout=300.0` on the download client |
| Gradio demo crashes on launch: `TypeError: argument of type 'bool' is not iterable` | `gradio==4.44.0`'s schema parser doesn't handle a newer `pydantic`'s `additionalProperties: true` (bare bool) JSON schema output — not a local-only quirk, would fail identically on Hugging Face Spaces | Upgraded to `gradio==6.20.0` |

Full stack: Python 3.11 · TensorFlow/Keras 2.19 · PyTorch 2.8 (CPU) · scikit-learn · pandas · matplotlib.

## Reproducing this

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
jupyter lab
```

Each notebook downloads its own dataset (a ~20MB satellite-tile archive) on first run. Trained model weights (`.keras`/`.pth`) are **not** included in this repo (several are 100MB-1.9GB); rerun the training notebooks to regenerate them.

## Origin

Built on lab notebooks from IBM's *AI Engineering* Skills Network courses (Coursera). The lab scaffolding, explanatory markdown, and starter code are IBM's; the exercises, the BatchNorm root-cause investigation, the four-model comparison, and the Gradio demo were completed here. Course authorship is credited in each notebook's "Author" section.
