"""
Coverage B: checkpoint loading for the (gitignored) training checkpoint.

ai_capstone_pytorch_state_dict.pth is the output of
notebooks/05-pytorch-cnn-classifier.ipynb. It's excluded from git (see
.gitignore) because it's a regenerable training artifact, not source. A
fresh clone -- and any CI runner -- will not have it. These tests must
skip cleanly rather than fail in that case; see test_repo_integrity.py for
the check that this skip behavior actually works on a clone without
weights.

This is deliberately a different file from demo/cnn_state_dict.pth (same
architecture, same weights, but a separate copy that IS tracked because the
demo needs to run standalone -- see test_demo.py, which always runs).
"""
import pytest
import torch

from conftest import PYTORCH_CNN_CHECKPOINT

pytestmark = pytest.mark.skipif(
    not PYTORCH_CNN_CHECKPOINT.exists(),
    reason=(
        f"{PYTORCH_CNN_CHECKPOINT.name} not present (gitignored training "
        "checkpoint; rerun notebooks/05 to regenerate it). This is expected "
        "on a fresh clone or CI runner, not a failure."
    ),
)


@pytest.fixture(scope="module")
def loaded_model():
    from app import build_model  # noqa: E402  (import after skip check)

    model = build_model()
    model.load_state_dict(torch.load(PYTORCH_CNN_CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def test_checkpoint_loads_without_error(loaded_model):
    assert loaded_model is not None


def test_forward_pass_produces_expected_output_shape(loaded_model):
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits = loaded_model(dummy)
    assert logits.shape == (1, 2), f"Expected (1, 2) logits, got {tuple(logits.shape)}"


def test_forward_pass_produces_valid_probabilities(loaded_model):
    dummy = torch.randn(4, 3, 64, 64)  # a small batch, not just batch size 1
    with torch.no_grad():
        logits = loaded_model(dummy)
        probs = torch.softmax(logits, dim=1)

    assert not torch.isnan(probs).any(), "Softmax output contains NaN"
    assert not torch.isinf(probs).any(), "Softmax output contains Inf"
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    # Each row's class probabilities should sum to 1 (softmax invariant)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)
