"""CPU tests for the evaluation suite (metrics, seeding, argument parsing, and a fake end-to-end run).

Run: python -m pytest -q tests/test_eval_suite_cpu.py
No model weights are loaded; generation uses eval_suite --fake and the judge uses --fake_judge.
"""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from mend.eval import image_metrics as em
from mend.eval.vlm_common import JUDGE_CRITERIA, judge_messages, parse_final_score

from mend.eval import bootstrap_table  # noqa: E402
from mend.analysis import mine_failures  # noqa: E402
from mend.eval import suite as eval_suite  # noqa: E402
from mend.eval import vlm_judge  # noqa: E402


# ------------------------------------------------------------------------------------------------ paper HF energy
def _png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)
    return str(path)


def test_hf_energy_default_is_paper_band():
    assert em.PAPER_HF_BAND == (0.08, 0.25)
    assert eval_suite.parse_args(["score", "--run", "x", "--hf_ref", "r"]).hf_band == "0.08,0.25"


def test_hf_energy_constant_image_is_zero():
    out = em.hf_energy(torch.full((2, 3, 32, 32), 0.3, dtype=torch.float64))
    assert torch.allclose(out["hf"], torch.zeros(2, dtype=torch.float64), atol=1e-20)


def test_hf_energy_white_noise_fraction_matches_band_area():
    rng = np.random.default_rng(0)
    frac = em.hf_energy(rng.random((8, 3, 128, 128)))["frac"].mean().item()
    # flat spectrum -> fraction = area of the annulus 0.08 <= r < 0.25 in the unit square (Hann blurs it slightly)
    assert abs(frac - math.pi * (0.25 ** 2 - 0.08 ** 2)) < 0.03


def test_hf_energy_orders_smooth_below_sharp():
    g = torch.Generator().manual_seed(2)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    low = (0.5 + 0.3 * torch.cos(2 * math.pi * 0.03 * xx) * torch.cos(2 * math.pi * 0.02 * yy)).expand(1, 3, 64, 64)
    sharp = (low + 0.05 * torch.randn(1, 3, 64, 64, generator=g)).clamp(0, 1)
    assert em.hf_energy(sharp)["hf"].item() > 10 * em.hf_energy(low)["hf"].item()


def test_hf_energy_matches_mine_failures_bitwise(tmp_path):
    """eval_suite's HF energy is the number mend/analysis/mine_failures.py computed for the paper (same function)."""
    rng = np.random.default_rng(3)
    arrs = [rng.integers(0, 256, (48, 40, 3), dtype=np.uint8) for _ in range(3)]
    rad, win = em.rfft_radius(48, 40), em.hann2d(48, 40)
    mine = [mine_failures.image_stats(_png(tmp_path / f"{i}.png", a), rad, win) for i, a in enumerate(arrs)]
    ours = em.hf_energy(torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2))["hf"].tolist()
    assert ours == [m["hf_mid"] for m in mine]
    # uint8 input equals the float64 x/255 input exactly
    f = em.hf_energy(np.stack(arrs).transpose(0, 3, 1, 2).astype(np.float64) / 255.0)["hf"].tolist()
    assert f == ours
    grain = em.spectral_band_energy(em.luma601(arrs[0].astype(np.float64) / 255.0), [(0.25, np.inf)], rad, win)
    assert grain[0] == mine[0]["hf_high"]


# ------------------------------------------------------------------------------------------------ grain energy (old hf)
def test_hf_grain_energy_constant_image_is_zero():
    out = em.hf_grain_energy(torch.full((2, 3, 32, 32), 0.3))
    assert torch.allclose(out["hf"], torch.zeros(2, dtype=torch.float64), atol=1e-20)


def test_hf_grain_energy_white_noise_fraction_matches_area():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 128, 128, generator=g)
    frac = em.hf_grain_energy(x, cutoff=0.25)["frac"].mean().item()
    # flat spectrum -> fraction = area of {r >= 0.25} in the unit square = 1 - pi/16 (Hann blurs it slightly)
    assert abs(frac - (1 - math.pi / 16)) < 0.03


def test_hf_grain_energy_parseval_total_is_windowed_variance():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(1, 3, 64, 48, generator=g)
    out = em.hf_grain_energy(x)
    y = (x.double() * torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64).view(1, 3, 1, 1)).sum(1)[0]
    y = y - y.mean()
    w = torch.outer(torch.hann_window(64, periodic=False, dtype=torch.float64),
                    torch.hann_window(48, periodic=False, dtype=torch.float64))
    wy = w * y  # Parseval: sum|F|^2 = HW sum wy^2; the excluded DC bin holds (sum wy)^2
    expected = ((wy ** 2).sum() - wy.sum() ** 2 / wy.numel()) / (w ** 2).sum()
    assert out["total"].item() == pytest.approx(expected.item(), rel=1e-9)


def test_hf_grain_energy_orders_smooth_below_sharp():
    g = torch.Generator().manual_seed(2)
    low = torch.nn.functional.interpolate(torch.rand(1, 3, 8, 8, generator=g), size=(64, 64), mode="bilinear")
    sharp = (low + 0.05 * torch.randn(1, 3, 64, 64, generator=g)).clamp(0, 1)
    e_low, e_sharp = em.hf_grain_energy(low)["hf"].item(), em.hf_grain_energy(sharp)["hf"].item()
    assert e_sharp > 10 * e_low


# ------------------------------------------------------------------------------------------------ diversity
def test_vendi_bounds():
    same = np.ones((5, 7))
    assert em.vendi_score(same) == pytest.approx(1.0, abs=1e-6)
    assert em.vendi_score(np.eye(5)) == pytest.approx(5.0, abs=1e-6)
    two = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    assert em.vendi_score(two) == pytest.approx(2.0, abs=1e-6)


def test_mean_pairwise_distance():
    assert em.mean_pairwise_cosine_distance(np.ones((4, 3))) == pytest.approx(0.0, abs=1e-12)
    assert em.mean_pairwise_cosine_distance(np.eye(3)) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    e = rng.normal(size=(5, 16))
    k = em.cosine_gram(e)
    brute = np.mean([1 - k[i, j] for i in range(5) for j in range(i + 1, 5)])
    assert em.mean_pairwise_cosine_distance(e) == pytest.approx(brute)


# ------------------------------------------------------------------------------------------------ statistics
def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    shifted = base + 0.1 + 0.05 * rng.normal(size=200)
    r = em.paired_bootstrap(shifted, base, n_boot=4000)
    assert r["lo"] > 0 and r["p"] < 0.01 and r["delta"] == pytest.approx(0.1, abs=0.02)
    null = em.paired_bootstrap(base + 0.05 * rng.normal(size=200), base, n_boot=4000)
    assert null["lo"] < 0 < null["hi"]


def test_bootstrap_mean_ci_covers_mean():
    r = em.bootstrap_mean(np.arange(100.0), n_boot=2000)
    assert r["lo"] < 49.5 < r["hi"] and r["n"] == 100


def test_debias_cancels_constant_position_bias():
    true_p, bias = 0.7, 0.15
    p_ab = true_p + bias              # A shown first
    p_ba_first = (1 - true_p) + bias  # B shown first
    assert em.debiased_pair_prob(p_ab, 1 - p_ba_first) == pytest.approx(true_p)


def test_spearman_matrix_symmetric():
    c = em.spearman_matrix({"a": [1, 2, 3, 4], "b": [2, 4, 6, 9], "c": [4, 3, 2, 1]})
    assert c["a"]["b"] == pytest.approx(1.0) and c["a"]["c"] == pytest.approx(-1.0) and c["b"]["a"] == c["a"]["b"]


# ------------------------------------------------------------------------------------------------ VLM helpers
def test_parse_final_score():
    assert parse_final_score("blah\nFinal Score: 4") == 4.0
    assert parse_final_score("Final Score: 3.5 because") == 3.5
    assert math.isnan(parse_final_score("no score here"))


def test_judge_messages_structure():
    from PIL import Image

    a, b = Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8), "white")
    for crit in JUDGE_CRITERIA:
        msgs = judge_messages("a cat", a, b, crit)
        content = msgs[0]["content"]
        assert [c["type"] for c in content] == ["text", "image", "text", "image", "text"]
        assert content[1]["image"] is a and content[3]["image"] is b and "a cat" in content[4]["text"]


# ------------------------------------------------------------------------------------------------ seeding / args
def test_initial_latent_is_per_image():
    a = eval_suite.initial_latent(42, 7, (16, 8, 8))
    b = eval_suite.initial_latent(42, 7, (16, 8, 8))
    c = eval_suite.initial_latent(43, 7, (16, 8, 8))
    d = eval_suite.initial_latent(42, 8, (16, 8, 8))
    assert torch.equal(a, b) and not torch.equal(a, c) and not torch.equal(a, d)


def test_unique_prompts_drawbench():
    ps = eval_suite.load_unique_prompts(eval_suite.DEFAULT_PROMPTS)
    assert len(ps) == 200 and len(set(ps)) == 200


def test_argparse_all_scripts(tmp_path):
    a = eval_suite.parse_args(["all", "--run", "x", "--lora", "opsd_pickscore", "--protocol", "flowgrpo"])
    assert a.cmd == "all" and a.protocol == "flowgrpo" and a.metrics.split(",") == list(eval_suite.ALL_METRICS)
    s = eval_suite.parse_args(["score", "--run", "x", "--hf_ref", "b", "--metrics", "hf"])
    assert s.hf_ref == "b"
    assert eval_suite.parse_args(["score", "--run", "x", "--base_run", "b"]).hf_ref == "b"  # deprecated alias
    j = vlm_judge.parse_args(["--run_a", "a", "--run_b", "b", "--criteria", "overall,fidelity"])
    assert j.criteria == "overall,fidelity"
    t = bootstrap_table.parse_args(["--ref", "r.json", "--runs", "a.json", "b.json"])
    assert t.runs == ["a.json", "b.json"]
    for name, path in eval_suite.KNOWN_LORAS.items():
        assert name == "base" or path


# ------------------------------------------------------------------------------------------------ end to end (fake)
def test_fake_end_to_end(tmp_path):
    root = str(tmp_path)
    common = ["--out_root", root, "--fake", "--n_prompts", "6", "--seeds", "1,2,3", "--metrics", "hf",
              "--n_boot", "500", "--device", "cpu"]
    with pytest.raises(ValueError, match="hf_ref"):  # the HF reference must be explicit
        eval_suite.main(["all", "--run", "base", *common])
    eval_suite.main(["all", "--run", "base", "--hf_ref", "base", *common])
    eval_suite.main(["all", "--run", "noisy", "--fake_noise", "0.08", "--hf_ref", "base", *common])
    # rerun: everything cached, nothing regenerated
    eval_suite.main(["all", "--run", "noisy", "--fake_noise", "0.08", "--hf_ref", "base", *common])
    with pytest.raises(RuntimeError):  # different generation config into the same run dir
        eval_suite.main(["generate", "--run", "noisy", "--out_root", root, "--fake", "--n_prompts", "6",
                         "--seeds", "1,2,3", "--fake_noise", "0.2"])
    ev = json.load(open(os.path.join(root, "noisy", "eval.json")))
    assert ev["n_images"] == 18 and ev["n_prompts"] == 6
    assert ev["summary"]["hf_ratio"]["mean"] > 1.5  # noisier images carry more HF energy than base
    assert len(ev["per_prompt"][0]["per_seed"]["hf_energy"]) == 3
    assert ev["hf_ref"]["run"] == "base" and ev["metric_info"]["hf"]["band"] == [0.08, 0.25]
    assert ev["hf_ref"]["paper_reference"] is False  # the fake base is not SD3.5-M at CFG 4.5
    self_ev = json.load(open(os.path.join(root, "base", "eval.json")))
    assert self_ev["summary"]["hf_ratio"]["mean"] == pytest.approx(1.0)
    # per-image ratio = hf_energy(run PNG) / hf_energy(ref PNG) from the shared function
    from PIL import Image
    f0 = ev["per_prompt"][0]["per_seed"]
    a, b = (np.asarray(Image.open(os.path.join(root, r, "images", "p000_s1.png")).convert("RGB")) for r in ("noisy", "base"))
    e = em.hf_energy(torch.from_numpy(np.stack([a, b])).permute(0, 3, 1, 2))["hf"].numpy()
    assert f0["hf_log_ratio"][0] == pytest.approx(math.log(e[0] / e[1]), rel=1e-12)

    # a cached hf.json made with another definition (the old 0.25 cutoff) is recomputed, not reused
    hf_path = os.path.join(root, "noisy", "scores", "hf.json")
    d = json.load(open(hf_path))
    d["info"] = {"cutoff": 0.25}
    d["hf_energy"] = [1.0] * len(d["hf_energy"])
    json.dump(d, open(hf_path, "w"))
    eval_suite.main(["score", "--run", "noisy", "--hf_ref", "base", "--out_root", root, "--metrics", "hf",
                     "--n_boot", "500", "--device", "cpu"])
    again = json.load(open(os.path.join(root, "noisy", "eval.json")))
    assert json.load(open(hf_path))["info"]["band"] == [0.08, 0.25]
    assert again["summary"]["hf_ratio"]["mean"] == pytest.approx(ev["summary"]["hf_ratio"]["mean"])

    # the old grain metric stays available under its own name, against the same reference
    eval_suite.main(["score", "--run", "noisy", "--hf_ref", "base", "--out_root", root, "--metrics", "hf,hf_grain",
                     "--n_boot", "500", "--device", "cpu"])
    g = json.load(open(os.path.join(root, "noisy", "eval.json")))
    assert g["summary"]["hf_grain_ratio"]["mean"] > 1.5 and "hf_ratio" in g["summary"]
    assert g["metric_info"]["hf_grain"]["cutoff"] == 0.25
    # the spool tasks pass the generate args (--lora, --protocol) to score too: accepted, checked against meta.json
    eval_suite.main(["score", "--run", "noisy", "--lora", "", "--protocol", "opsd", "--hf_ref", "base",
                     "--out_root", root, "--metrics", "hf", "--n_boot", "500", "--device", "cpu"])
    with pytest.raises(SystemExit, match="protocol"):
        eval_suite.main(["score", "--run", "noisy", "--protocol", "flowgrpo", "--hf_ref", "base",
                         "--out_root", root, "--metrics", "hf", "--n_boot", "500", "--device", "cpu"])
    # --hf_ref none: energies only, no ratio
    eval_suite.main(["score", "--run", "noisy", "--hf_ref", "none", "--out_root", root, "--metrics", "hf",
                     "--n_boot", "500", "--device", "cpu"])
    assert "hf_ratio" not in json.load(open(os.path.join(root, "noisy", "eval.json")))["summary"]

    cmp = bootstrap_table.main(["--ref", os.path.join(root, "base", "eval.json"),
                                "--runs", os.path.join(root, "noisy", "eval.json"),
                                "--names", "Base", "Noisy", "--metrics", "hf_energy,hf_frac",
                                "--md", os.path.join(root, "t.md"), "--csv", os.path.join(root, "t.csv"),
                                "--tex", os.path.join(root, "t.tex"), "--n_boot", "500"])
    cell = cmp["rows"][1]["cells"]["hf_energy"]["paired"]
    assert cell["delta"] > 0 and cell["lo"] > 0
    assert os.path.getsize(os.path.join(root, "t.tex")) > 0

    out = vlm_judge.main(["--run_a", "noisy", "--run_b", "base", "--out_root", root, "--fake_judge",
                          "--criteria", "overall", "--n_boot", "500", "--batch_size", "4"])
    r = out["results"]["overall"]
    assert r["n_pairs"] == 18 and r["wins_A"] + r["wins_B"] + r["ties"] == 18
    assert r["position_bias"] == pytest.approx(0.1, abs=0.02)  # the fake judge adds +0.1 to Image 1
    # resumed run reads the cache and gives the same numbers
    again = vlm_judge.main(["--run_a", "noisy", "--run_b", "base", "--out_root", root, "--fake_judge",
                            "--criteria", "overall", "--n_boot", "500"])
    assert again["results"]["overall"]["soft_win_rate_A"] == r["soft_win_rate_A"]
