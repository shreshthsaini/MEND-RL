# SPDX-License-Identifier: Apache-2.0
"""Download and validate PEFT adapters without importing a model runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

# Only published, verified releases belong here. Entries pin a Hub commit and hashes.
CHECKPOINTS: dict[str, dict] = {'mend_pickscore': {'repo_id': 'shreshthsaini/MEND-SD3.5M-PickScore',
                    'revision': 'e99538d7493f5b13df728e77f84a181a0d1845ec',
                    'subfolder': '',
                    'family': 'sd3',
                    'guidance_scale': 4.5,
                    'num_steps': 40,
                    'resolution': 512,
                    'training_updates': 100,
                    'sha256': {'adapter_config.json': '3ffc7026a28504dba1aa125e6fb71a97783947becda966dc8ef2965266ba793c',
                               'adapter_model.safetensors': 'ef798a199eb13f456f67929385d5e259690a18e34a1762401c679e5ecad621f6'},
                    'base_model': 'stabilityai/stable-diffusion-3.5-medium'},
 'mend_open3': {'repo_id': 'shreshthsaini/MEND-SD3.5M-ThreeReward',
                'revision': '6f2e86e6cff81b46810b3a175d504129de66db95',
                'subfolder': '',
                'family': 'sd3',
                'guidance_scale': 1.0,
                'num_steps': 40,
                'resolution': 512,
                'training_updates': 300,
                'sha256': {'adapter_config.json': 'cebfa81c4bd16c3c80eb5cee4a1b8e88c43c394d3310346367e59265120876a8',
                           'adapter_model.safetensors': '53703d117773cb3bcceaa4ce4c5aebd27cc2b66d6613ee14cc871f1709b7b28a'},
                'base_model': 'stabilityai/stable-diffusion-3.5-medium'},
 'mend_clipscore': {'repo_id': 'shreshthsaini/MEND-SD3.5M-CLIPScore',
                    'revision': '8cf35e7820f85ae548337c01390fa2181e88dbe0',
                    'subfolder': '',
                    'family': 'sd3',
                    'base_model': 'stabilityai/stable-diffusion-3.5-medium',
                    'guidance_scale': 1.0,
                    'num_steps': 40,
                    'resolution': 512,
                    'training_updates': 100,
                    'sha256': {'adapter_config.json': '19a606e54149c3bf56fdc3949eade7bc19802ff163daa2cb8ad96395d0c5d4ee',
                               'adapter_model.safetensors': 'a9d083cf8b3d2bdd90766a775acff7008f0e429d1d6a14d98beebc3380fb01f9'}},
 'mend_hpsv2': {'repo_id': 'shreshthsaini/MEND-SD3.5M-HPSv2.1',
                'revision': '728c8f13acb06e9c9f46e6ac86615453ddab0f2f',
                'subfolder': '',
                'family': 'sd3',
                'base_model': 'stabilityai/stable-diffusion-3.5-medium',
                'guidance_scale': 1.0,
                'num_steps': 40,
                'resolution': 512,
                'training_updates': 100,
                'sha256': {'adapter_config.json': 'ef564f6447e2abc7f774f01a25aa9d042084f0e87ab7ec5e5b4187917bf74c61',
                           'adapter_model.safetensors': 'd95728aa7960bec8a5c229d8a9a2c6585cb4178362b8a431317a4e17f85de1d1'}},
 'mend_imagereward': {'repo_id': 'shreshthsaini/MEND-SD3.5M-ImageReward',
                      'revision': 'c07523e8f019787b28e0d8b8cc982ecef5727da3',
                      'subfolder': '',
                      'family': 'sd3',
                      'base_model': 'stabilityai/stable-diffusion-3.5-medium',
                      'guidance_scale': 1.0,
                      'num_steps': 40,
                      'resolution': 512,
                      'training_updates': 100,
                      'sha256': {'adapter_config.json': '1b71e04a70389de5b4b7e32f0c17431e6aeeb43d8e53d5fcdd0ffa9657d146da',
                                 'adapter_model.safetensors': '04fff7027e63730acd0cf8943b4112da85fdbc81e62202977970116688f97429'}},
 'mend_sd3m_pickscore_s1': {'repo_id': 'shreshthsaini/MEND-SD3M-PickScore-Seed1',
                            'revision': '53f01ae754784ad0987cc849ab8df1a4a315a23a',
                            'subfolder': '',
                            'family': 'sd3',
                            'base_model': 'stabilityai/stable-diffusion-3-medium-diffusers',
                            'guidance_scale': 1.0,
                            'num_steps': 40,
                            'resolution': 512,
                            'training_updates': 100,
                            'sha256': {'adapter_config.json': 'a1cd51a4194da4bcb64ed5e02021ed0bb65462561135371f21ddc4ad64b81300',
                                       'adapter_model.safetensors': 'a34e968004fdc86e08996520aa08fd0d51d8de0c679c3d9767af03004478252c'}},
 'mend_sd3m_pickscore_s2': {'repo_id': 'shreshthsaini/MEND-SD3M-PickScore-Seed2',
                            'revision': 'd0b65c41edc72f21d79988b098dc3f753ead9818',
                            'subfolder': '',
                            'family': 'sd3',
                            'base_model': 'stabilityai/stable-diffusion-3-medium-diffusers',
                            'guidance_scale': 1.0,
                            'num_steps': 40,
                            'resolution': 512,
                            'training_updates': 100,
                            'sha256': {'adapter_config.json': 'af1229d13831b2295be667357ace394232286037adcf58f5d17d010fd91b0fef',
                                       'adapter_model.safetensors': '666866120643e61a8dcb3cca59fa71f46cc146575849335531d49b4e7a59ef5a'}},
 'mend_zimage_hpsv2': {'repo_id': 'shreshthsaini/MEND-Z-Image-Turbo-HPSv2.1',
                       'revision': '234ef5f0ee4ed97b021a167e5945cd8657289b7f',
                       'subfolder': '',
                       'family': 'zimage',
                       'base_model': 'Tongyi-MAI/Z-Image-Turbo',
                       'guidance_scale': 0.0,
                       'num_steps': 9,
                       'resolution': 1024,
                       'training_updates': 100,
                       'sha256': {'adapter_config.json': '72042d0c06e04cbd18b475e3a21a0b3843df7618878a998a599c20e4a7067439',
                                  'adapter_model.safetensors': '9a9b1c61bce876248be5c03a310eab6f484c3455163d8feffc6555853dfb1613'}},
 'mend_zimage_pickscore': {'repo_id': 'shreshthsaini/MEND-Z-Image-Turbo-PickScore',
                           'revision': 'ae66ee7bc2664d1836eb46c7f23b514fd27a317a',
                           'subfolder': '',
                           'family': 'zimage',
                           'base_model': 'Tongyi-MAI/Z-Image-Turbo',
                           'guidance_scale': 0.0,
                           'num_steps': 9,
                           'resolution': 1024,
                           'training_updates': 100,
                           'sha256': {'adapter_config.json': 'b9deaba48126bd17ae1b1176dbfb1b4adbc03260af8e426144eb12fabb336e47',
                                      'adapter_model.safetensors': '09cdbbf51ff7340864a893c4b5b3bd7d7ca69749066bb5f77d75db4060feeed2'}}}
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
