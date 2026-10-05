"""CPU tests for preemption-safe resume of the MEND trainer (save_ckpt + launcher checkpoint choice)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import ml_collections
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

from mend.utils.ema import EMAModuleWrapper  # noqa: E402
from mend.utils.checkpointing import (  # noqa: E402
    load_resume_params, resolve_resume_checkpoint, resume_position, save_resume_params,
)


def test_resume_params_roundtrip(tmp_path):
    raw = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    old = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    save_resume_params(str(tmp_path), raw, old)
    raw2 = [torch.nn.Parameter(torch.zeros(3, 4)), torch.nn.Parameter(torch.zeros(5))]
    old2 = [torch.nn.Parameter(torch.zeros(3, 4)), torch.nn.Parameter(torch.zeros(5))]
    assert load_resume_params(str(tmp_path), raw2, old2) == {"raw": True, "old": True}
    for a, b in zip(raw + old, raw2 + old2):
        assert torch.equal(a.data, b.data)
    bad = [torch.nn.Parameter(torch.zeros(4, 3)), torch.nn.Parameter(torch.zeros(5))]
    with pytest.raises(ValueError):
        load_resume_params(str(tmp_path), bad)
    assert load_resume_params(str(tmp_path / "none"), raw2) == {"raw": False, "old": False}


def _tiny_peft():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8))
    cfg = LoraConfig(r=2, lora_alpha=4, init_lora_weights="gaussian", target_modules=["0"])
    m = get_peft_model(base, cfg)
    m.add_adapter("old", cfg)
    m.set_adapter("default")
    train = [p for p in m.parameters() if p.requires_grad]
    m.set_adapter("old")
    old = [p for p in m.parameters() if p.requires_grad]
    m.set_adapter("default")
    return m, train, old


def test_save_ckpt_keeps_raw_weights_and_old_adapter(tmp_path):
    from mend.train import sd3 as T

    m, train, old = _tiny_peft()
    ema = EMAModuleWrapper(train, decay=0.9, update_step_interval=1, device="cpu")
    with torch.no_grad():  # raw weights move away from the EMA; old adapter gets its own values
        for p in train:
            p.add_(1.0)
        for p in old:
            p.fill_(0.25)
    ema.step(train, 1)
    opt = torch.optim.AdamW(train, lr=1e-3)
    config = ml_collections.ConfigDict({"train": {"ema": True}})
    raw_before = [p.detach().clone() for p in train]
    T.save_ckpt(str(tmp_path), SimpleNamespace(module=m), 5, 0, ema, train, config, opt, None,
                epoch_completed=5, old_params=old)
    ck = tmp_path / "checkpoints" / "checkpoint-5"
    for name in ("COMPLETE", "resume_params.pt", "optimizer.pt", "trainer_state.json", "ema.pt", "lora"):
        assert (ck / name).exists(), name
    for p, r in zip(train, raw_before):  # live weights are raw again after the EMA copy for lora/
        assert torch.equal(p.data, r)
    # Resume as the trainer does: lora/ (EMA) into default and old, then the exact restore.
    m2, train2, old2 = _tiny_peft()
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    st = load_peft_weights(str(ck / "lora"), device="cpu")
    set_peft_model_state_dict(m2, st, adapter_name="default")
    set_peft_model_state_dict(m2, st, adapter_name="old")
    ema_w = [p.detach().clone() for p in train2]
    assert any(not torch.equal(a, r) for a, r in zip(ema_w, raw_before)), "lora/ should hold the EMA weights"
    assert load_resume_params(str(ck), train2, old2) == {"raw": True, "old": True}
    for a, r in zip(train2, raw_before):
        assert torch.equal(a.data, r)
    for a in old2:
        assert torch.all(a.data == 0.25)
    assert resolve_resume_checkpoint(str(tmp_path)) == str(ck)
    cfg = ml_collections.ConfigDict({"sample": {"train_batch_size": 8, "num_batches_per_epoch": 16},
                                     "train": {"batch_size": 8, "gradient_accumulation_steps": 16,
                                               "num_inner_epochs": 1}})
    assert resume_position(str(ck), cfg, 1) == (5, 5)


def _fake_ckpt(root: Path, step: int, complete: bool) -> None:
    d = root / "checkpoints" / f"checkpoint-{step}"
    (d / "lora").mkdir(parents=True)
    (d / "trainer_state.json").write_text(json.dumps({"epoch_completed": step, "global_step": step}))
    (d / "optimizer.pt").write_bytes(b"")
    (d / "resume_params.pt").write_bytes(b"")
    if complete:
        (d / "COMPLETE").write_text(f"{step}\n")


def _launch(out: Path, updates: int = 30) -> str:
    # Resume selection is independent of real prompts and downloaded weights.
    # Supply the tiny filesystem fixtures checked by the no-model dry run.
    fixtures = out.parent / "launcher_fixtures"
    prompts, model, hub = fixtures / "prompts", fixtures / "model", fixtures / "hf"
    prompts.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    for split in ("train.txt", "test.txt"):
        (prompts / split).write_text("A test image.\n")
    for repo in ("yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        snapshot = hub / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / "test"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}")
    env = dict(os.environ, MEND_DRYRUN="1", MEND_SCRATCH_PREFIX="/", NPROC="1", UPDATES=str(updates), OUTPUT_DIR=str(out),
               WANDB_DIR=str(fixtures / "wandb"), HF_HOME=str(hub), MODEL=str(model), PYTHON=sys.executable)
    r = subprocess.run(["bash", "scripts/train_mend.sh", "pickscore",
                        "--config.mend.proposal=explicit", "--config.dataset=" + str(prompts),
                        "--config.sample.num_image_per_prompt=8", "--config.sample.train_batch_size=8",
                        "--config.train.batch_size=8", "--config.sample.num_batches_per_epoch=16",
                        "--config.train.gradient_accumulation_steps=16"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    return r.stdout + r.stderr


def test_launcher_resumes_from_newest_complete_checkpoint(tmp_path):
    out = tmp_path / "run"
    _fake_ckpt(out, 5, complete=True)
    _fake_ckpt(out, 10, complete=True)
    _fake_ckpt(out, 15, complete=False)  # save cut by preemption: must be skipped
    log = _launch(out)
    assert "resuming from " + str(out / "checkpoints" / "checkpoint-10") in log, log
    lines = [line for line in log.splitlines() if line.startswith("DRYRUN_OK")]
    assert lines, log
    line = lines[0]
    info = json.loads(line.split(" ", 1)[1])
    assert info["resume"]["global_step"] == 10 and info["resume"]["first_epoch"] == 10
    (out / "run_done.json").write_text(json.dumps({"global_step": 30}))
    assert "nothing to do" in _launch(out)
    assert "nothing to do" not in _launch(out, updates=40)  # a longer target resumes instead
