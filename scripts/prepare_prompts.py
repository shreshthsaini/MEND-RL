"""Materialize the Pick-a-Pic training prompts and the DrawBench evaluation prompts under data/.

Thin entry point; the implementation lives in ``mend.data.prepare_pickapic``. Equivalent to ``python -m mend.data.prepare_pickapic``.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("mend.data.prepare_pickapic", run_name="__main__", alter_sys=True)
