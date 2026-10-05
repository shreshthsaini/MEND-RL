"""CPU plumbing check of mend/analysis/g1_sweep.py (run_chunk + summarize) with a fake Gate: restart variants
o1 / o1c / o2 / o2c, their baselines, verdict keys and the restart_variants comparison.

Run: python -m pytest -q tests/test_g1_sweep_cpu.py
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from mend.analysis import g1_sweep  # noqa: E402

from mend import algorithm as mend  # noqa: E402
from test_mend_cpu import SIG  # noqa: E402


def _v(z, t):
    t = float(t)
    return 0.5 * torch.tanh(z) + 0.3 * z * (1 - t) - 0.2 * math.sin(3 * t)


class FakeGate:
    def __init__(self):
        self.sigmas = SIG.float()

    def embed(self, prompts):
        return torch.zeros(len(prompts), 1), torch.zeros(len(prompts), 1)

    def seeds(self, ids):
        return torch.stack([torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(int(i))) for i in ids])

    def vfn(self, emb, pemb, adapter=None, reps=1, grad=False, counter=None):
        def v(z, sig):
            if counter is not None:
                counter["nfe"] = counter.get("nfe", 0) + z.shape[0]
            return _v(z, sig)
        return v

    def rollout(self, z0, emb, pemb, adapter=None, hook=None, counter=None):
        zs, vs = mend.rollout_dpm2(self.vfn(emb, pemb, counter=counter), z0, self.sigmas)
        return zs[-1], zs, vs

    def _r(self, kind, x):
        w = {"pickscore": 1.0, "hpsv2": 0.5, "aesthetic": -0.3}[kind]
        return w * x.float().flatten(1).mean(1) - 0.1 * mend.sq_mean(x.float())

    def reward(self, kind, x, prompts, grad=False):
        if not grad:
            return self._r(kind, x)
        xg = x.float().detach().requires_grad_(True)
        r = self._r(kind, xg)
        (g,) = torch.autograd.grad(r.sum(), xg)
        return r.detach(), g

    def rewards_multi(self, kinds, x, prompts, return_images=False):
        out = {k: self._r(k, x) for k in kinds}
        return (out, x.float()) if return_images else out


def test_g1_restart_variants_end_to_end():
    args = g1_sweep.parse_args(["--anchored_variants", "o1,o1c,o2,o2c", "--anchor_sigmas", "0.55,0.7",
                                "--disp_n", "2", "--taus", "0.3,3"])
    mc = SimpleNamespace(etas_explicit=[0.05, 0.1, 0.2], etas_anchored=[0.1, 0.2, 0.4], q=0.9)
    config = SimpleNamespace(mend=mc)
    grid = g1_sweep.build_grid(args, config)
    names = [c["name"] for c in grid]
    assert "grad/anchored_s0.55_o2_corr" in names and "rand/anchored_s0.7_corr" in names and "grad/explicit" in names
    assert len(grid) == 2 * (1 + 2 * 4)
    G = FakeGate()
    kinds = ["pickscore", "hpsv2", "aesthetic"]
    chunks = []
    for cid in range(2):
        prompts = [f"p{cid}{i // 4}" for i in range(8)]
        chunks.append(g1_sweep.run_chunk(G, args, grid, kinds, cid, prompts, [10 * cid + i // 4 for i in range(8)],
                                         list(range(8 * cid, 8 * cid + 8)), list(range(8 * cid, 8 * cid + 8))))
    ch = chunks[0]
    assert set(ch["baseline"]) == {"s0.55", "s0.55_o2", "s0.7", "s0.7_o2"}
    # the order-2 delta = 0 restart reproduces x (move ~ 0); order 1 does not
    assert float(ch["baseline"]["s0.55_o2"]["move"].max()) < 1e-5
    assert float(ch["baseline"]["s0.55"]["move"].mean()) > 1e-4
    # raw and corrected differ by exactly the baseline move for order 1
    raw, corr = ch["cfg"]["grad/anchored_s0.55"], ch["cfg"]["grad/anchored_s0.55_corr"]
    assert not torch.allclose(raw["move"], corr["move"])
    assert raw["nfe"] == corr["nfe"] > 0
    S = g1_sweep.summarize(args, grid, kinds, chunks, {"config": config})
    v = S["verdict"]
    assert "proximal_rb_tau0.3" in v["grad/anchored_s0.55"] and "proximal_rb_tau0.3" in v["grad/anchored_s0.55_o2"]
    assert "proximal_rb_tau0.3" not in v["grad/anchored_s0.55_corr"]
    rv = S["comparisons"]["restart_variants"]
    assert set(rv) == {f"{h}/anchored_s{s}" for h in ("grad", "rand") for s in ("0.55", "0.7")}
    assert set(rv["grad/anchored_s0.55"][0]) == {"eta", "o1", "o1c", "o2", "o2c"}
    assert S["configs"]["grad/anchored_s0.7_o2_corr"]["restart_order"] == 2


def test_g1_default_grid_unchanged():
    """Default variants reproduce the original G1 grid names (old chunk dirs stay summarizable)."""
    args = g1_sweep.parse_args([])
    config = SimpleNamespace(mend=SimpleNamespace(etas_explicit=[0.05, 0.1, 0.2], etas_anchored=[0.1, 0.2, 0.4]))
    names = [c["name"] for c in g1_sweep.build_grid(args, config)]
    assert names == ["grad/explicit", "grad/anchored_s0.4", "grad/anchored_s0.55", "grad/anchored_s0.7",
                     "rand/explicit", "rand/anchored_s0.4", "rand/anchored_s0.55", "rand/anchored_s0.7"]
