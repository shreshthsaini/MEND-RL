"""CPU test of scripts/generate.py: prompt collection and the output layout, with the suite's fake generator."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("mend_generate_cli", REPO / "scripts" / "generate.py")
gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gen)


def test_fake_generation_layout_and_resume(tmp_path):
    listed = tmp_path / "list.txt"
    listed.write_text("a red cube\n\na blue sphere\na red cube\n")
    out = tmp_path / "samples" / "run1"
    argv = ["--prompt", "a green cone", "--prompt", "a red cube", "--prompts", str(listed),
            "--seeds", "0,3", "--out_dir", str(out), "--fake"]
    rdir = gen.main(argv)
    assert Path(rdir) == out
    # --prompt entries first, then the file, duplicates removed
    assert (out / "prompts.txt").read_text().splitlines() == ["a green cone", "a red cube", "a blue sphere"]
    names = sorted(p.name for p in (out / "images").glob("*.png"))
    assert names == ["p000_s0.png", "p000_s3.png", "p001_s0.png", "p001_s3.png", "p002_s0.png", "p002_s3.png"]
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert {(r["prompt"], r["seed"]) for r in rows} == {(p, s) for p in ("a green cone", "a red cube", "a blue sphere")
                                                        for s in (0, 3)}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["guidance_scale"] == 1.0 and meta["num_steps"] == 40 and meta["lora_spec"] == ""
    before = {p.name: p.stat().st_mtime_ns for p in (out / "images").glob("*.png")}
    gen.main(argv)  # rerun: nothing is regenerated
    assert before == {p.name: p.stat().st_mtime_ns for p in (out / "images").glob("*.png")}


def test_requires_a_prompt(tmp_path):
    with pytest.raises(SystemExit):
        gen.main(["--out_dir", str(tmp_path / "x"), "--fake"])


def test_download_options_forwarded(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(gen.suite, "generate", lambda args: captured.append(args))
    gen.main(["--prompt", "a book", "--out_dir", str(tmp_path / "x"), "--lora", "author/repo",
              "--lora_revision", "abc123", "--lora_subfolder", "pickscore", "--cache_dir", str(tmp_path),
              "--local_files_only"])
    args = captured[0]
    assert args.lora == "author/repo" and args.lora_revision == "abc123"
    assert args.lora_subfolder == "pickscore" and args.cache_dir == str(tmp_path)
    assert args.local_files_only and args.batch_size == 1


def test_resume_rejects_different_adapter_revision(tmp_path):
    argv = ["--prompt", "a book", "--out_dir", str(tmp_path / "run"), "--fake", "--lora", "author/repo"]
    gen.main(argv + ["--lora_revision", "commit-a"])
    with pytest.raises(RuntimeError, match="different generation config"):
        gen.main(argv + ["--lora_revision", "commit-b"])


def test_old_metadata_has_compatible_default_download_options():
    before = {"lora_spec": "base", "fake": False}
    after = dict(before, lora_revision=None, lora_subfolder="")
    assert gen.suite.gen_signature(before) == gen.suite.gen_signature(after)


def test_shared_resolver_accepts_zimage_for_hires(tmp_path):
    adapter = tmp_path / "zimage"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "auto_mapping": {"base_model_class": "ZImageTransformer2DModel"},
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"mock weights")
    assert gen.suite.resolve_lora(str(adapter), expected_family="zimage") == str(adapter)
    with pytest.raises(ValueError, match="requires sd3"):
        gen.suite.resolve_lora(str(adapter))


@pytest.mark.parametrize("alias,override,expected", [
    ("mend_sd3m_pickscore_s1", "", "stabilityai/stable-diffusion-3-medium-diffusers"),
    ("mend_sd3m_pickscore_s2", "", "stabilityai/stable-diffusion-3-medium-diffusers"),
    ("mend_pickscore", "", "stabilityai/stable-diffusion-3.5-medium"),
    ("mend_sd3m_pickscore_s1", "/custom/base", "/custom/base"),
])
def test_official_generation_pairs_adapter_with_base(tmp_path, monkeypatch, alias, override, expected):
    config = SimpleNamespace(resolution=512, pretrained=SimpleNamespace(model="wrong-fallback-base"),
                             sample=SimpleNamespace(noise_level=0.0))
    monkeypatch.setitem(sys.modules, "mend.eval.cross_eval", SimpleNamespace(load_config=lambda *args: config))
    monkeypatch.setattr(gen.suite, "resolve_lora", lambda *args, **kwargs: "/adapter")
    captured = []

    def generate(args, config, model, lora, protocol, resolution, noise, todo, save):
        captured.append((model, lora))
        for item in todo:
            save(gen.suite._fake_image(item["seed"], item["pidx"], resolution), item)

    monkeypatch.setattr(gen.suite, "_generate_sd3", generate)
    out = tmp_path / "sample"
    gen.main(["--prompt", "a book", "--lora", alias, "--model", override,
              "--resolution", "8", "--out_dir", str(out)])
    assert captured == [(expected, "/adapter")]
    assert json.loads((out / "meta.json").read_text())["model"] == expected
    from mend.eval import gen_compare

    comparison = gen_compare.parse_method(alias + ":res=8", gen_compare.builtin_methods())
    args = gen_compare.parse_args(["--model", override, "--out_root", str(tmp_path / "compare")])
    gen_compare.run_method(args, comparison,
                           [{"prompt_id": "p0", "source": "test", "tag": "", "prompt": "a book"}], [0])
    assert captured == [(expected, "/adapter"), (expected, "/adapter")]
    metadata = json.loads((tmp_path / "compare" / alias / "meta.json").read_text())
    assert metadata["signature"]["model"] == expected


def test_zimage_release_cannot_use_sd3_generator(tmp_path, monkeypatch):
    config = SimpleNamespace(resolution=512, pretrained=SimpleNamespace(model="unused-base"),
                             sample=SimpleNamespace(noise_level=0.0))
    monkeypatch.setitem(sys.modules, "mend.eval.cross_eval", SimpleNamespace(load_config=lambda *args: config))

    def forbidden_generation(*args):
        pytest.fail("Z-Image adapter must be rejected before building an SD3 pipeline")

    monkeypatch.setattr(gen.suite, "_generate_sd3", forbidden_generation)
    with pytest.raises(ValueError, match="requires the zimage pipeline"):
        gen.main(["--prompt", "a book", "--lora", "mend_zimage_pickscore", "--out_dir", str(tmp_path)])
    from mend.eval import gen_compare

    with pytest.raises(ValueError, match="unknown method"):
        gen_compare.parse_method("mend_zimage_pickscore", gen_compare.builtin_methods())
