"""CPU plumbing checks of mend/analysis/g0_gate.py additions: second-order rows of check ii, the corrected repair of
check iv, and the keep term in _fit. Uses the fake Gate of test_g1_sweep_cpu.py and a tiny fake transformer.

Run: python -m pytest -q tests/test_g0_gate_cpu.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mend.analysis import g0_gate  # noqa: E402
from test_g1_sweep_cpu import FakeGate  # noqa: E402

from mend import algorithm as mend  # noqa: E402


class _TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(4))
        self.adapter = "default"

    def set_adapter(self, name):
        self.adapter = name

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict=False):
        return (hidden_states * (1.0 + self.lora_A.view(1, 4, 1, 1)),)


class FitGate(FakeGate):
    def __init__(self):
        super().__init__()
        self.transformer = _TinyTransformer()
        self.dtype = torch.bfloat16
        self.n_steps = len(self.sigmas) - 1
        self.config = SimpleNamespace(train=SimpleNamespace(adam_beta1=0.9, adam_beta2=0.999, adam_weight_decay=0.0,
                                                            adam_epsilon=1e-8, max_grad_norm=1.0))

    def lora_params(self, adapter="default"):
        return {"lora_A.default": self.transformer.lora_A}


def test_check_restart_reports_order2():
    g0_gate.G_args = g0_gate.parse_args(["--restart_orders", "1,2"])
    G = FakeGate()
    z0 = G.seeds([1, 2])
    emb, pemb = G.embed(["a", "b"])
    res = g0_gate.check_restart(G, emb, pemb, z0, ["a", "b"], "pickscore", [0, 3, 6])
    assert res["pass"] and res["o2_pass"]
    assert res["k6"]["trainer_o2"]["rms_err"]["mean"] < 1e-5 < res["k6"]["trainer"]["rms_err"]["mean"]


def test_repair_correction_and_order():
    G = FakeGate()
    prompts = ["a", "b", "c"]
    emb, pemb = G.embed(prompts)
    z0 = G.seeds([3, 4, 5])
    out = {}
    for order, corr in ((1, "none"), (1, "delta"), (2, "none"), (2, "delta")):
        g0_gate.G_args = g0_gate.parse_args(["--restart_order", str(order), "--restart_correction", corr, "--eta", "0.2"])
        X, Z, V, D, RX, RY, ks = g0_gate._repair(G, prompts, emb, pemb, z0, "pickscore", 2)
        out[(order, corr)] = (D, g0_gate._repair.last_y0_move)
        assert Z.shape[1] == len(G.sigmas) - 1
    # order 2: y0 = x, so correction changes nothing; order 1: correction removes y0 - x
    assert torch.allclose(out[(2, "none")][0], out[(2, "delta")][0], atol=1e-5)
    assert float(out[(2, "none")][1].max()) < 1e-5 < float(out[(1, "none")][1].min())
    assert not torch.allclose(out[(1, "none")][0], out[(1, "delta")][0], atol=1e-4)


def test_fit_keep_term_runs_and_is_logged():
    g0_gate.G_args = g0_gate.parse_args(["--train_mb", "4"])
    G = FitGate()
    B, N = 4, G.n_steps
    Z = torch.randn(B, N, 4, 8, 8)
    V = torch.randn(B, N, 4, 8, 8)
    D = 0.1 * torch.randn(B, 4, 8, 8)
    emb, pemb = torch.zeros(B, 1), torch.zeros(B, 1)
    Zk, Vk = torch.randn(3, N, 4, 8, 8), torch.randn(3, N, 4, 8, 8)
    keep = (Zk, Vk, torch.zeros(3, 1), torch.zeros(3, 1), 10.0, 2)
    curve = g0_gate._fit(G, 1e-2, Z, V, D, emb, pemb, 3, 2, 2, 3, lambda: {}, keep=keep)
    assert curve[-1]["step"] == 3 and curve[-1]["keep_loss"] > 0
    curve0 = g0_gate._fit(G, 1e-2, Z, V, D, emb, pemb, 2, 2, 2, 2, lambda: {})
    assert "keep_loss" not in curve0[-1]


def test_fit_fix_targets_and_state_diag_run():
    """Path-fix candidates in the iv fit: cut path, ramp, single-state x0, hybrid."""
    G = FitGate()
    B, N = 4, G.n_steps
    Z = torch.randn(B, N, 4, 8, 8)
    V = torch.randn(B, N, 4, 8, 8)
    D = 0.1 * torch.randn(B, 4, 8, 8)
    emb, pemb = torch.zeros(B, 1), torch.zeros(B, 1)
    keep = (torch.randn(3, N, 4, 8, 8), torch.randn(3, N, 4, 8, 8), torch.zeros(3, 1), torch.zeros(3, 1), 10.0, 2)
    for extra in (["--path_sigma_max", "0.5"], ["--path_sigma_max", "0.6", "--path_shape", "ramp"],
                  ["--path_sigma_max", "0.5", "--path_high", "keep"], ["--target", "single_state_x0"],
                  ["--target", "hybrid", "--path_sigma_max", "0.5"],
                  ["--target", "x0_multi", "--path_sigma_max", "0.603", "--x0_sigma_min", "0.2"]):
        g0_gate.G_args = g0_gate.parse_args(["--train_mb", "4"] + extra)
        idx, flags = g0_gate._train_idx(G, B, N, 2, torch.Generator().manual_seed(0))
        if "--path_sigma_max" in extra and "keep" not in extra:
            smax = float(extra[extra.index("--path_sigma_max") + 1])
            cols = [j for j, f in enumerate(flags) if not f]
            assert all(float(G.sigmas[int(k)]) <= smax + 1e-6 for k in idx[:, cols].flatten())
        if "x0_multi" in extra:
            assert all(flags) and all(0.2 - 1e-6 <= float(G.sigmas[int(k)]) <= 0.603 for k in idx.flatten())
        if "hybrid" in extra or "single_state_x0" in extra:
            assert flags[0] and int(idx[0, 0]) == mend.sigma_to_index(G.sigmas.double().cpu(), 0.278)
        curve = g0_gate._fit(G, 1e-2, Z, V, D, emb, pemb, 2, 2, 2, 2, lambda: {}, keep=keep)
        assert curve[-1]["step"] == 2 and curve[-1]["loss"] > 0 and curve[-1]["keep_loss"] > 0
    diag = g0_gate._state_diag(G, Z, V, D, emb, pemb, 3)
    assert len(diag) == N and all(r["path_resid_rel"] >= 0 for r in diag)
