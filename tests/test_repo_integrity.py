"""
Coverage E: repo integrity checks that don't require any ML framework.
"""
import subprocess

import nbformat
import pytest

from conftest import NOTEBOOKS_DIR, REPO_ROOT

ALL_NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def test_notebooks_directory_is_not_empty():
    assert len(ALL_NOTEBOOKS) == 9, f"Expected 9 notebooks, found {len(ALL_NOTEBOOKS)}"


@pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS, ids=lambda p: p.name)
class TestEachNotebook:
    def test_is_valid_nbformat_json(self, notebook_path):
        nb = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(nb)  # raises on malformed structure

    def test_has_no_error_output_cells(self, notebook_path):
        nb = nbformat.read(notebook_path, as_version=4)
        errors = []
        for i, cell in enumerate(nb.get("cells", [])):
            for output in cell.get("outputs", []) or []:
                if output.get("output_type") == "error":
                    errors.append(
                        f"cell {i} ({cell.get('id', '?')}): "
                        f"{output.get('ename')}: {output.get('evalue')}"
                    )
        assert not errors, f"{notebook_path.name} has error output cells:\n" + "\n".join(errors)


class TestGitignoreBehavior:
    """Behavioral checks via `git check-ignore`, not string-matching the
    .gitignore file -- this actually exercises the negation rule rather
    than assuming the text does what it says."""

    def _is_ignored(self, relative_path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative_path],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        # git check-ignore exit codes: 0 = ignored, 1 = not ignored, 128 = error
        assert result.returncode in (0, 1), (
            f"git check-ignore errored on {relative_path!r}: {result.stderr.decode()}"
        )
        return result.returncode == 0

    def test_venv_is_ignored(self):
        assert self._is_ignored(".venv/some_file.txt")

    def test_generic_keras_checkpoint_is_ignored(self):
        assert self._is_ignored("some_model.keras")

    def test_generic_pth_checkpoint_is_ignored(self):
        assert self._is_ignored("some_checkpoint.pth")

    def test_dataset_dir_is_ignored(self):
        # Checking the entry itself, not a path inside it: locally,
        # images_dataSAT is a symlink (created by skillsnetwork.prepare()
        # when a notebook downloads the dataset), and `git check-ignore`
        # refuses to resolve paths *through* a symlink (exit 128, not a
        # clean ignored/not-ignored 0/1). The gitignore pattern
        # ("images_dataSAT", no trailing slash) matches the entry itself
        # regardless of whether it's a plain directory (fresh clone) or a
        # symlink (this dev machine), so testing the top-level name covers
        # both without tripping over that.
        assert self._is_ignored("images_dataSAT")

    def test_demo_checkpoint_is_explicitly_not_ignored(self):
        """The one *.pth exception: the demo needs to run standalone on a
        fresh clone, so this specific file is deliberately tracked despite
        the blanket *.pth rule above it in .gitignore."""
        assert not self._is_ignored("demo/cnn_state_dict.pth")

    def test_demo_checkpoint_is_actually_tracked_by_git(self):
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "demo/cnn_state_dict.pth"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        assert result.returncode == 0, (
            "demo/cnn_state_dict.pth is not tracked by git -- the demo will "
            "not run on a fresh clone. (Not ignoring a path is not the same "
            "as it being committed.)"
        )

    def test_onnx_export_is_not_ignored(self):
        """No *.onnx rule exists in .gitignore, but this pins that down
        behaviorally rather than assuming nobody adds one later -- the
        Streamlit demo (the one this repo actually deploys) needs
        cnn_model.onnx on a fresh clone just as much as app_gradio.py
        needs cnn_state_dict.pth."""
        assert not self._is_ignored("demo/cnn_model.onnx")

    def test_onnx_export_is_actually_tracked_by_git(self):
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "demo/cnn_model.onnx"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        assert result.returncode == 0, (
            "demo/cnn_model.onnx is not tracked by git -- the Streamlit demo "
            "will not run on a fresh clone."
        )
