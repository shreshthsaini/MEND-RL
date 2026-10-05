"""Repository-relative locations used by the command-line tools.

``MEND_ROOT`` is where run artifacts live: ``outputs/``, ``wandb/``, ``taskq/``, ``telemetry/`` and ``logs/``.
It defaults to the repository root and is overridden with the ``MEND_ROOT`` environment variable.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
MEND_ROOT = Path(os.environ.get("MEND_ROOT", str(REPO_ROOT)))
OUTPUT_ROOT = MEND_ROOT / "outputs"
