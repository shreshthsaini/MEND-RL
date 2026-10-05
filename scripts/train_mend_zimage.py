"""Train MEND on Z-Image-Turbo.

Thin entry point; the implementation lives in ``mend.train.zimage``. Equivalent to ``python -m mend.train.zimage``.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("mend.train.zimage", run_name="__main__", alter_sys=True)
