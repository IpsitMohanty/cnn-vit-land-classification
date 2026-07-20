"""
Coverage C: the Gradio/torch demo (demo/app_gradio.py).

Unlike test_checkpoint_loading.py, these tests always run -- demo/
bundles its own copy of the CNN checkpoint (demo/cnn_state_dict.pth),
intentionally tracked in git specifically so the demo works standalone on
a fresh clone. See the .gitignore negation rule and test_repo_integrity.py.

Mirrored by test_demo_streamlit.py, which covers the same predict-path
contract for the Streamlit/onnxruntime demo (app.py) that this repo
actually deploys.
"""
from pathlib import Path

import pytest
from PIL import Image

from app_gradio import CLASS_NAMES, predict
from conftest import DEMO_DIR

EXAMPLES_DIR = DEMO_DIR / "examples"


def _example_paths(subdir: str) -> list[Path]:
    d = EXAMPLES_DIR / subdir
    paths = sorted(d.glob("*.jpg"))
    assert paths, f"No example images found in {d}"
    return paths


AGRI_EXAMPLES = _example_paths("agri")
NON_AGRI_EXAMPLES = _example_paths("non_agri")


def test_six_bundled_examples_exist():
    assert len(AGRI_EXAMPLES) == 3
    assert len(NON_AGRI_EXAMPLES) == 3


class TestPredictReturnsValidOutput:
    def test_returns_label_and_confidence_for_each_class(self):
        img = Image.open(AGRI_EXAMPLES[0])
        result = predict(img)
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_confidences_are_in_zero_one_range(self):
        img = Image.open(AGRI_EXAMPLES[0])
        result = predict(img)
        for label, confidence in result.items():
            assert 0.0 <= confidence <= 1.0, f"{label}: {confidence} out of [0, 1]"

    def test_confidences_sum_to_one(self):
        img = Image.open(AGRI_EXAMPLES[0])
        result = predict(img)
        assert abs(sum(result.values()) - 1.0) < 1e-5


class TestBundledExamplesClassifyCorrectly:
    """These are real held-out tiles from the training distribution (see
    demo/README.md) -- the model should get all six right with high
    confidence, which was confirmed manually before this suite existed.
    Encoding it as a test catches any future regression (e.g. swapping in
    a different checkpoint, or a preprocessing change)."""

    @pytest.mark.parametrize("path", AGRI_EXAMPLES)
    def test_agri_examples_classify_as_agricultural(self, path):
        result = predict(Image.open(path))
        predicted = max(result, key=result.get)
        assert predicted == "agricultural", f"{path.name}: predicted {predicted}, {result}"

    @pytest.mark.parametrize("path", NON_AGRI_EXAMPLES)
    def test_non_agri_examples_classify_as_non_agricultural(self, path):
        result = predict(Image.open(path))
        predicted = max(result, key=result.get)
        assert predicted == "non-agricultural", f"{path.name}: predicted {predicted}, {result}"


class TestMalformedInputHandling:
    """Gradio's Image component (type="pil") only ever hands predict() a
    valid PIL.Image or None -- these tests cover the realistic space of
    "malformed" input within that contract: unusual sizes, modes, and the
    None case the app's own code explicitly guards against."""

    def test_none_input_returns_none_without_raising(self):
        assert predict(None) is None

    def test_tiny_1x1_image_does_not_raise(self):
        img = Image.new("RGB", (1, 1), color=(128, 64, 32))
        result = predict(img)
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_large_image_does_not_raise(self):
        img = Image.new("RGB", (2000, 1500), color=(0, 200, 0))
        result = predict(img)
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_grayscale_image_does_not_raise(self):
        img = Image.new("L", (64, 64), color=128)
        result = predict(img)
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_rgba_image_does_not_raise(self):
        img = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))
        result = predict(img)
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_non_square_image_does_not_raise(self):
        img = Image.new("RGB", (200, 50), color=(10, 20, 30))
        result = predict(img)
        assert set(result.keys()) == set(CLASS_NAMES)
