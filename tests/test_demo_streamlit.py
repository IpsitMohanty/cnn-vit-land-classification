"""
Coverage C: the Streamlit/onnxruntime demo (demo/app.py) -- the version
this repo actually deploys. Mirrors test_demo_gradio.py's coverage of
the torch demo's predict-path contract; see that file's docstring for
why these tests always run rather than skip.

app.py defers `import streamlit` to inside main(), so importing it here
for CLASS_NAMES/predict does not require streamlit to be installed --
onnxruntime and the bundled cnn_model.onnx are the only real
dependencies of the predict path under test.
"""
from pathlib import Path

import pytest
from PIL import Image

import app_gradio
from app import CLASS_NAMES, predict
from conftest import DEMO_DIR

EXAMPLES_DIR = DEMO_DIR / "examples"


def _example_paths(subdir: str) -> list[Path]:
    d = EXAMPLES_DIR / subdir
    paths = sorted(d.glob("*.jpg"))
    assert paths, f"No example images found in {d}"
    return paths


AGRI_EXAMPLES = _example_paths("agri")
NON_AGRI_EXAMPLES = _example_paths("non_agri")
ALL_EXAMPLES = AGRI_EXAMPLES + NON_AGRI_EXAMPLES


def test_six_bundled_examples_exist():
    assert len(AGRI_EXAMPLES) == 3
    assert len(NON_AGRI_EXAMPLES) == 3


def test_class_names_match_gradio_demo():
    """Both demos wrap the same underlying model -- their label sets must agree."""
    assert CLASS_NAMES == app_gradio.CLASS_NAMES


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
    """Same rationale as test_demo_gradio.py: these are real held-out
    tiles the model should get right with high confidence."""

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


class TestParityWithTorchDemo:
    """The whole point of shipping an ONNX export is that it reproduces
    the torch checkpoint's behavior, not a re-trained or re-derived one.
    Pins down the parity that was manually verified when cnn_model.onnx
    was generated (see export_onnx.py) as a standing regression check --
    catches the checkpoint and the ONNX export drifting apart (e.g. the
    .pth regenerated without re-running export_onnx.py)."""

    @pytest.mark.parametrize("path", ALL_EXAMPLES)
    def test_onnx_probabilities_match_torch_within_tolerance(self, path):
        img = Image.open(path)
        onnx_result = predict(img)
        torch_result = app_gradio.predict(img)
        for label in CLASS_NAMES:
            assert onnx_result[label] == pytest.approx(torch_result[label], abs=1e-3), (
                f"{path.name}/{label}: onnx={onnx_result[label]} torch={torch_result[label]}"
            )


class TestMalformedInputHandling:
    """app.py's predict() takes a PIL.Image or None -- same contract as
    the Gradio demo's Image component hands to its predict()."""

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
