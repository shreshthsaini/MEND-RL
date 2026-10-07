"""Adapter resolution tests without model imports, network requests, or real weights."""

import builtins
import hashlib
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mend_checkpoints_under_test", REPO / "mend" / "checkpoints.py")
checkpoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoints)


def write_adapter(path, model_class="SD3Transformer2DModel", weight_name="adapter_model.safetensors"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "auto_mapping": {"base_model_class": model_class},
    }))
    (path / weight_name).write_bytes(b"test adapter bytes, not a real model")
    return path


@pytest.fixture
def hub(monkeypatch, tmp_path):
    calls = []
    snapshot = tmp_path / "snapshot"

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return snapshot, calls


@pytest.mark.parametrize("inline", [True, False])
def test_hub_subfolder_downloads_only_selected_adapter(hub, inline):
    snapshot, calls = hub
    adapter = write_adapter(snapshot / "checkpoints" / "checkpoint-100" / "lora")
    folder = "checkpoints/checkpoint-100/lora"
    spec = "author/model/" + folder if inline else "author/model"
    result = checkpoints.resolve_adapter(spec, subfolder="" if inline else folder, expected_family="sd3")
    assert result == str(adapter.resolve())
    assert calls == [("author/model", {
        "revision": None, "cache_dir": None, "local_files_only": False,
        "allow_patterns": [folder + "/" + name for name in checkpoints.ADAPTER_FILES],
    })]


def test_revision_cache_and_explicit_offline_are_forwarded(hub, tmp_path):
    snapshot, calls = hub
    write_adapter(snapshot)
    checkpoints.resolve_adapter("author/model", revision="abc123", cache_dir=str(tmp_path / "cache"),
                                local_files_only=True)
    assert calls[0][1] == {
        "revision": "abc123", "cache_dir": str(tmp_path / "cache"), "local_files_only": True,
        "allow_patterns": list(checkpoints.ADAPTER_FILES),
    }


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("TRUE", True), ("yes", True),
                                           ("on", True), ("0", False), ("false", False), ("", False)])
def test_offline_environment_is_forwarded(monkeypatch, hub, value, expected):
    snapshot, calls = hub
    write_adapter(snapshot)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    checkpoints.resolve_adapter("author/model")
    assert calls[0][1]["local_files_only"] is expected


@pytest.mark.parametrize("weight_name", ["adapter_model.safetensors", "adapter_model.bin"])
def test_local_subfolder_uses_no_network(hub, tmp_path, weight_name):
    _, calls = hub
    adapter = write_adapter(tmp_path / "local" / "lora", weight_name=weight_name)
    assert checkpoints.resolve_adapter(str(adapter.parent), subfolder="lora") == str(adapter.resolve())
    assert calls == []


def test_explicit_missing_local_path_never_contacts_hub(hub, tmp_path):
    _, calls = hub
    with pytest.raises(FileNotFoundError, match="Local adapter directory"):
        checkpoints.resolve_adapter(str(tmp_path / "missing"))
    assert calls == []


@pytest.mark.parametrize("defect,error,match", [
    ("missing_config", FileNotFoundError, "adapter_config.json"),
    ("invalid_json", ValueError, "Invalid adapter configuration"),
    ("wrong_type", ValueError, "PEFT LORA"),
    ("missing_weights", FileNotFoundError, "adapter_model"),
    ("empty_weights", ValueError, "empty or a Git LFS pointer"),
    ("lfs_pointer", ValueError, "Git LFS pointer"),
])
def test_invalid_local_adapter_fails_without_network(hub, tmp_path, defect, error, match):
    _, calls = hub
    adapter = write_adapter(tmp_path / "local")
    config, weights = adapter / "adapter_config.json", adapter / "adapter_model.safetensors"
    if defect == "missing_config":
        config.unlink()
    elif defect == "invalid_json":
        config.write_text("{broken")
    elif defect == "wrong_type":
        config.write_text('{"peft_type": "PREFIX_TUNING"}')
    elif defect == "missing_weights":
        weights.unlink()
    elif defect == "empty_weights":
        weights.write_bytes(b"")
    else:
        weights.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    with pytest.raises(error, match=match):
        checkpoints.resolve_adapter(str(adapter))
    assert calls == []


def test_downloaded_snapshot_is_validated(hub):
    snapshot, calls = hub
    write_adapter(snapshot)
    (snapshot / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_model"):
        checkpoints.resolve_adapter("author/model")
    assert len(calls) == 1


def test_family_mismatch_rejected_for_local_adapter(tmp_path):
    adapter = write_adapter(tmp_path / "zimage", model_class="ZImageTransformer2DModel")
    with pytest.raises(ValueError, match="for zimage.*requires sd3"):
        checkpoints.resolve_adapter(str(adapter), expected_family="sd3")


@pytest.fixture
def named_release(monkeypatch, hub):
    snapshot, _ = hub
    adapter = write_adapter(snapshot / "policy")
    preset = {
        "repo_id": "author/released", "revision": "a" * 40, "subfolder": "policy", "family": "sd3",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()},
    }
    monkeypatch.setitem(checkpoints.CHECKPOINTS, "mend_test", preset)
    return adapter, preset


def test_named_release_pins_revision_subfolder_and_verifies_hashes(hub, named_release):
    _, calls = hub
    adapter, preset = named_release
    assert checkpoints.resolve_adapter("mend_test", expected_family="sd3") == str(adapter.resolve())
    assert calls[0][0] == preset["repo_id"]
    assert calls[0][1]["revision"] == preset["revision"]
    assert calls[0][1]["allow_patterns"] == ["policy/" + name for name in checkpoints.ADAPTER_FILES]
    (adapter / "adapter_model.safetensors").write_bytes(b"changed after release")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        checkpoints.resolve_adapter("mend_test")


@pytest.mark.parametrize("kwargs", [{"revision": "main"}, {"subfolder": "old"}])
def test_named_release_rejects_overrides_before_download(hub, named_release, kwargs):
    _, calls = hub
    with pytest.raises(ValueError, match="Named releases are pinned"):
        checkpoints.resolve_adapter("mend_test", **kwargs)
    assert calls == []


def test_named_release_family_mismatch_fails_before_download(hub, named_release):
    _, calls = hub
    with pytest.raises(ValueError, match="requires the sd3 pipeline"):
        checkpoints.resolve_adapter("mend_test", expected_family="zimage")
    assert calls == []


@pytest.mark.parametrize("subfolder", ["../other", "/absolute", "nested/../../other", "nested\\other"])
def test_unsafe_subfolder_fails_before_download(hub, subfolder):
    _, calls = hub
    with pytest.raises(ValueError, match="relative path"):
        checkpoints.resolve_adapter("author/model", subfolder=subfolder)
    assert calls == []


@pytest.mark.parametrize("option,expected", [("--help", "--local_files_only"),
                                            ("--list", "No official MEND adapters are published yet.")])
def test_download_cli_never_imports_model_runtime(monkeypatch, capsys, option, expected):
    original_import = builtins.__import__
    forbidden = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "diffusers", "transformers"}:
            forbidden.append(name)
            raise AssertionError(f"Download command attempted model import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", [str(REPO / "scripts" / "download_weights.py"), option])
    if option == "--help":
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(REPO / "scripts" / "download_weights.py"), run_name="__main__")
        assert exc.value.code == 0
    else:
        runpy.run_path(str(REPO / "scripts" / "download_weights.py"), run_name="__main__")
    assert expected in capsys.readouterr().out
    assert forbidden == []
