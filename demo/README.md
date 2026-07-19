# Satellite Land Classification — Gradio Demo

Upload a 64×64 satellite tile (or click one of the 6 bundled examples) and get a live agricultural/non-agricultural prediction from the PyTorch CNN trained in [`notebooks/05-pytorch-cnn-classifier.ipynb`](../notebooks/05-pytorch-cnn-classifier.ipynb) (98.6-99.75% accuracy on this task).

**Deliberately not using the CNN-ViT hybrid checkpoint here** — it saves as a 100MB-1.9GB file (see [`results/ANALYSIS.md`](../results/ANALYSIS.md) §3 for why the hybrid isn't worth the extra weight anyway: it doesn't beat the plain CNN on this task). The plain CNN checkpoint bundled here is 78MB, well within Hugging Face Spaces' free tier.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
python app.py
```

Opens at `http://127.0.0.1:7860`.

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: Gradio, hardware: free CPU tier).
2. Push this folder's contents (`app.py`, `requirements.txt`, `cnn_state_dict.pth`, `examples/`) to the Space's repo.
3. Spaces auto-installs `requirements.txt` and runs `app.py`.

## Honest limitations

The app states this in its own UI, but worth repeating here: this model was trained on one specific distribution of Sentinel-2 satellite tiles, evenly split between two classes, and scores well *on that distribution*. It has not seen aerial photography, drone imagery, ground-level photos, or tiles from other sensors/resolutions/regions. The bundled example tiles are real held-out samples from the training distribution — they're the fair test of this model. Anything else, including your own uploads, is out-of-distribution and the prediction should not be trusted.
