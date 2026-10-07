# SPDX-License-Identifier: Apache-2.0
"""Download and validate PEFT adapters without importing a model runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

# Only published, verified releases belong here. Entries pin a Hub commit and hashes.
CHECKPOINTS: dict[str, dict] = {}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora_revision", default=None, help="Adapter Hub commit, tag or branch.")
    parser.add_argument("--lora_subfolder", default="", help="Adapter folder inside the Hub repo or local directory.")
    parser.add_argument("--cache_dir", default=None, help="Hub cache for adapter and base model; defaults to HF cache.")
    parser.add_argument("--local_files_only", action="store_true", help="Use cached/local base and adapter files only.")


def offline_requested(local_files_only: bool = False) -> bool:
    return local_files_only or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}


def _subfolder(value: str) -> str:
    if not value:
        return ""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Adapter subfolder must be a relative path without '..'.")
    return str(path) if str(path) != "." else ""


def validate_adapter(path: str | Path, expected_family: str | None = None, hashes: dict | None = None) -> str:
    path = Path(path).expanduser()
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No adapter_config.json in {path}. Select checkpoint-N/lora, not its parent.")
    try:
        config = json.loads(config_path.read_text())
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid adapter configuration: {config_path}") from exc
    if not isinstance(config, dict) or str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"{config_path} must describe a PEFT LORA adapter.")
    mapping = config.get("auto_mapping") or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid auto_mapping in {config_path}; expected an object.")  # noqa: TRY004
    model_class = mapping.get("base_model_class", "")
    families = {"SD3Transformer2DModel": "sd3", "ZImageTransformer2DModel": "zimage"}
    family = families.get(model_class)
    if expected_family and family and family != expected_family:
        raise ValueError(f"Adapter is for {family}, but this pipeline requires {expected_family}.")
    weights = [path / name for name in ADAPTER_FILES[1:] if (path / name).is_file()]
    if not weights:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {path}.")
    for weight in weights:
        with weight.open("rb") as stream:
            prefix = stream.read(128)
        if not prefix or prefix.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{weight} is empty or a Git LFS pointer. Download the actual weights.")
    for name, expected in (hashes or {}).items():
        digest = hashlib.sha256()
        with (path / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {path / name}; download the pinned release again.")
    return str(path.resolve())


def resolve_adapter(
    spec: str, *, revision: str | None = None, subfolder: str = "", cache_dir: str | None = None,
    local_files_only: bool = False, expected_family: str | None = None,
) -> str:
    """Resolve a local directory, namespace/repo[/subfolder], or a pinned release name."""
    if not spec or spec == "base":
        if revision or subfolder:
            raise ValueError("Adapter revision/subfolder requires an adapter, not the base model.")
        return ""
    preset = CHECKPOINTS.get(spec)
    hashes = None
    if preset:
        if revision or subfolder:
            raise ValueError("Named releases are pinned; use a Hub repo id to override revision or subfolder.")
        if expected_family and preset["family"] != expected_family:
            raise ValueError(f"{spec} requires the {preset['family']} pipeline.")
        spec, revision, subfolder = preset["repo_id"], preset["revision"], preset["subfolder"]
        hashes = preset["sha256"]
    subfolder = _subfolder(subfolder)
    local = Path(spec).expanduser()
    if local.is_dir():
        if revision:
            raise ValueError("A Hub revision cannot be used with a local adapter directory.")
        return validate_adapter(local / subfolder, expected_family, hashes)
    if spec.startswith(("/", "./", "../", "~")) or local.exists():
        raise FileNotFoundError(f"Local adapter directory does not exist: {local}")
    parts = spec.split("/")
    if len(parts) < 2 or not all(parts[:2]) or any(p in {".", ".."} for p in parts):
        raise ValueError(f"Unknown adapter {spec!r}. Use a local directory or namespace/repo[/subfolder].")
    repo_id = "/".join(parts[:2])
    if len(parts) > 2:
        if subfolder:
            raise ValueError("Specify the adapter subfolder in the repo path OR --lora_subfolder, not both.")
        subfolder = _subfolder("/".join(parts[2:]))
    prefix = f"{subfolder}/" if subfolder else ""
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id, revision=revision, cache_dir=cache_dir,
            local_files_only=offline_requested(local_files_only),
            allow_patterns=[prefix + name for name in ADAPTER_FILES],
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot load adapter {repo_id} (revision {revision or 'main'}, folder {subfolder or '/'}). "
            "Check the repo/revision and cache. For private or gated repositories, run `hf auth login` "
            "and request access on Hugging Face. Offline use requires downloading this revision first."
        ) from exc
    return validate_adapter(Path(snapshot) / subfolder, expected_family, hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", nargs="?", help="Local directory, Hub repo[/subfolder], or release name.")
    parser.add_argument("--list", action="store_true", help="List published MEND checkpoint presets.")
    add_download_args(parser)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CHECKPOINTS, indent=2) if CHECKPOINTS else "No official MEND adapters are published yet.")
        return
    if not args.adapter:
        parser.error("provide an adapter or --list")
    print(resolve_adapter(args.adapter, revision=args.lora_revision, subfolder=args.lora_subfolder,
                          cache_dir=args.cache_dir, local_files_only=args.local_files_only))


if __name__ == "__main__":
    main()
