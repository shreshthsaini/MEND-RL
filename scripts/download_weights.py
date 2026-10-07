# SPDX-License-Identifier: Apache-2.0
"""Download a PEFT adapter without importing Torch, Diffusers, or reward models."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "mend" / "checkpoints.py"), run_name="__main__")
