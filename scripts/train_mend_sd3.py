"""Train MEND on Stable Diffusion 3.5 Medium.

Thin entry point; the implementation lives in ``mend.train.sd3``. Equivalent to ``python -m mend.train.sd3``.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("mend.train.sd3", run_name="__main__", alter_sys=True)
