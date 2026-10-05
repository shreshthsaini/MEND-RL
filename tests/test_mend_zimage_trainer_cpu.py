"""End-to-end CPU run of mend/train/zimage.py with a tiny fake Z-Image pipeline (tests/fake_zimage_run.py).

Checks the trainer plumbing that the tensor-level tests cannot: rollout -> hint -> cap -> Euler-restart proposals ->
verdict -> displaced-path/keep loss -> one AdamW update -> EMA/old-adapter update -> checkpoint -> exact resume.

Run: python -m pytest -q -s tests/test_mend_zimage_trainer_cpu.py
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(tmp, out, epochs, *extra, want_log=False):
    data = tmp / "data"
    if not data.exists():
        data.mkdir()
        prompts = "\n".join(f"a fake prompt number {i} with colour" for i in range(8)) + "\n"
        (data / "train.txt").write_text(prompts)
        (data / "test.txt").write_text(prompts)
    env = dict(os.environ, RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
               MASTER_PORT=str(_free_port()), PUBLIC_POLICY_WORLD_SIZE="1", PUBLIC_N_GPUS="1",
               WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="", FAKE_SEED="0")
    cmd = [sys.executable, str(REPO / "tests" / "fake_zimage_run.py"), "--config",
           str(REPO / "configs" / "mend.py") + ":zimage_pickscore", "--config.resolution=64",
           "--config.mixed_precision=no", f"--config.dataset={data}", f"--config.save_dir={out}",
           f"--config.logdir={tmp / 'wandb'}", f"--config.num_epochs={epochs}", "--config.save_freq=1",
           "--config.pretrained.model=fake", *extra]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-6000:]
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if "mend/acceptance" in l]
    return (rows, p.stdout + p.stderr) if want_log else rows


@pytest.fixture(scope="module")
def run2(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zimage_mend")
    out = tmp / "out"
    return tmp, out, _run(tmp, out, 2)


def test_rounds_train_and_log(run2):
    _, out, rows = run2
    assert len(rows) == 2
    r1 = rows[0]
    print("\nround 1:", {k: round(v, 5) for k, v in r1.items() if isinstance(v, float) and k.startswith("mend/")})
    # round 1: the trained adapter equals the rollout adapter, so the recomputed v_old matches v_theta at z exactly
    assert r1["mend/keep_residual"] < 1e-10
    assert 0 < r1["mend/failing_frac"] < 1 and r1["mend/acceptance"] > 0 and r1["mend/repaired_frac"] > 0
    assert r1["mend/path_residual_repaired"] > 0 and r1["mend/grad_norm"] > 0 and r1["mend/nfe"] > 0
    assert r1["mend/R_ystar_accepted"] > r1["mend/R_x_accepted"]  # T1: the verified endpoint is better
    # Euler restart: the delta = 0 proposal is the rollout itself (no restart bias on Z-Image)
    assert r1["mend/restart_rel_err_delta0"] < 1e-6 and abs(r1["mend/restart_dR_delta0"]) < 1e-5
    # round 2 sees an updated trained adapter (EMA old adapter lags), so the keep term is active
    assert rows[1]["mend/keep_residual"] > 0
    done = json.loads((out / "run_done.json").read_text())
    assert done["global_step"] == 2


def test_checkpoints_complete_and_params_move(run2):
    _, out, _ = run2
    c1, c2 = out / "checkpoints" / "checkpoint-1", out / "checkpoints" / "checkpoint-2"
    for c in (c1, c2):
        assert (c / "COMPLETE").is_file() and (c / "resume_params.pt").is_file() and (c / "mend_state.json").is_file()
        assert (c / "lora" / "adapter_config.json").is_file() and (c / "lora" / "old").is_dir()
    a = torch.load(c1 / "resume_params.pt", weights_only=True)
    b = torch.load(c2 / "resume_params.pt", weights_only=True)
    assert len(a["raw"]) == len(b["raw"]) > 0
    moved = max(float((x - y).abs().max()) for x, y in zip(a["raw"], b["raw"]))
    old_moved = max(float((x - y).abs().max()) for x, y in zip(a["old"], b["old"]))
    print(f"\nraw LoRA max change 1->2 {moved:.2e}; old adapter {old_moved:.2e}")
    assert moved > 0 and old_moved > 0
    cfg = json.loads((c2.parent.parent / "run_config.json").read_text())
    assert cfg["sample"]["solver"] == "flow_euler" and cfg["sample"]["num_steps"] == 9


def test_resume_continues(run2):
    tmp, out, _ = run2
    rows = _run(tmp, out, 3, f"--config.resume_from={out / 'checkpoints' / 'checkpoint-2'}")
    assert json.loads((out / "run_done.json").read_text())["global_step"] == 3
    assert (out / "checkpoints" / "checkpoint-3" / "COMPLETE").is_file()
    st = json.loads((out / "checkpoints" / "checkpoint-3" / "mend_state.json").read_text())
    assert st["global_step"] == 3
    assert rows[-1]["mend/keep_residual"] > 0  # resumed with distinct raw and old adapters


def test_explicit_single_state_variant(tmp_path):
    rows = _run(tmp_path, tmp_path / "out", 1, "--config.mend.proposal=explicit",
                "--config.mend.target_mode=single_state_x0", "--config.mend.query_sigma=0.273")
    assert rows[0]["mend/target_states_per_seed"] == 1 and rows[0]["mend/grad_norm"] > 0


def test_x0_multi_variant(tmp_path):
    """x0_multi (the SD3 default target) with the best_mend.env band [0.2, 0.603]: on the native 9-step grid it
    trains 2 of the 3 unmoved states .600/.462/.273 (grid indices 6, 7, 8) per seed, kept seeds included."""
    rows, log = _run(tmp_path, tmp_path / "out", 2, "--config.mend.target_mode=x0_multi",
                     "--config.mend.path_sigma_max=0.603", "--config.mend.x0_sigma_min=0.2", want_log=True)
    assert "x0_multi states: grid indices [6, 7, 8]" in log, log[-3000:]
    assert all(r["mend/target_states_per_seed"] == 2 for r in rows)
    assert rows[0]["mend/keep_residual"] < 1e-10 and rows[0]["mend/grad_norm"] > 0
    assert rows[1]["mend/grad_norm"] > 0
