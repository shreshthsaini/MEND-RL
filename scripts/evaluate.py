"""Evaluate a base model or a trained LoRA on DrawBench with the full reward and diversity suite.

Thin entry point; the implementation lives in ``mend.eval.suite``. Equivalent to ``python -m mend.eval.suite``.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("mend.eval.suite", run_name="__main__", alter_sys=True)
