import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"
RESULTS_DIR = REPO_ROOT / "results"
INVESTIGATION_DIR = RESULTS_DIR / "investigation"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# demo/app.py is a standalone script, not a package (no __init__.py) --
# make it importable as `app` without restructuring the demo for pytest's sake.
sys.path.insert(0, str(DEMO_DIR))

# Gitignored training checkpoints (present locally, absent on a fresh clone / CI).
# NOT the same file as demo/cnn_state_dict.pth, which is intentionally tracked
# and covered separately by test_demo.py.
PYTORCH_CNN_CHECKPOINT = REPO_ROOT / "ai_capstone_pytorch_state_dict.pth"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
