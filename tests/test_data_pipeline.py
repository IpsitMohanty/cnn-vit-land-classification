"""
Coverage D: data pipeline / preprocessing helpers.

This repo doesn't have a standalone "data utils" module -- the loading
and transform logic lives either inline in notebook cells (not meant to
be imported) or in demo/app.py's `preprocess` pipeline (real, importable,
and what actually runs at inference time). These tests cover the latter,
since it's the one piece of data-pipeline code in this repo that exists
outside a notebook.
"""
import torch
from PIL import Image

from app import preprocess


def _make_image(color=(200, 100, 50), size=(64, 64)):
    return Image.new("RGB", size, color=color)


def test_output_shape_is_chw_64x64():
    img = _make_image()
    tensor = preprocess(img)
    assert tensor.shape == (3, 64, 64), f"Expected (3, 64, 64), got {tuple(tensor.shape)}"


def test_output_dtype_is_float():
    img = _make_image()
    tensor = preprocess(img)
    assert tensor.dtype == torch.float32


def test_arbitrary_input_size_is_resized_to_64x64():
    img = _make_image(size=(512, 300))
    tensor = preprocess(img)
    assert tensor.shape == (3, 64, 64)


def test_normalization_is_imagenet_not_zero_one():
    """The pipeline uses ImageNet mean/std normalization (matching how the
    checkpoint was trained -- see notebooks/05), not a plain 0-1 rescale.
    A pure white image should land outside [0, 1] after normalization,
    proving this isn't just ToTensor() on its own."""
    white = _make_image(color=(255, 255, 255))
    tensor = preprocess(white)
    assert tensor.max().item() > 1.0, (
        "Expected values above 1.0 after ImageNet normalization of a white "
        "image; got a plain [0, 1] range instead -- has the transform changed?"
    )
    # Sanity bound: normalization shouldn't blow up to extreme values either.
    assert tensor.max().item() < 5.0
    assert tensor.min().item() > -5.0


def test_normalization_is_deterministic():
    """Resize/ToTensor/Normalize contain no randomness (unlike the training-
    time augmentation pipelines in the notebooks), so running the same
    image through preprocess() twice must give bit-identical output with no
    seeding required."""
    img = _make_image(color=(37, 142, 201))
    first = preprocess(img)
    second = preprocess(img)
    assert torch.equal(first, second)
