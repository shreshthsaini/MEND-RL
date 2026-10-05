#!/usr/bin/env python3
"""Write the deferred spool tasks for the paper's results section.

Groups (task name prefixes; all go to taskq/deferred and carry #REQUIRES lines, so infra/release_ready.sh moves each
one to pending once its inputs exist; best_mend.final is the marker the coordinator touches after G2 fixes
config/best_mend.env):
  43/44  joint PickScore/26 + CLIPScore + HPSv2.1 (OPSD sd35_open3, 300 updates), Pareto and scalar verdict
  47     reduced method (explicit + cap + path) at Protocol O, 100 updates, one per reward
  49     tab:pareto regression rates (CPU-light)
  52     Z-Image-Turbo MEND on CLIPScore and ImageReward (after the Z-Image PickScore run has checkpoint 25)
  60/61  qualitative compare samples of MEND checkpoints (48 compare prompts x 8 seeds) + collapse/mining/figures
  70     design-space / P6 ablation arms (configs/p6_arms.json), 79 their tables
  80     VLM judge (Qwen2.5-VL-7B, both orders) MEND vs released baselines on DrawBench 200 x 5
  82     paired held-out / diversity / HF tables (bootstrap_table.py)
  85-88  theory checks: T3/T5 report, T4 mode mass, T6 seed displacement, repair-step figure renders
Usage: python infra/results_tasks.py [--out DIR] [--force] [--list]
Existing files (in the target dir or any spool state) are left alone unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex

CODE = os.environ.get("MEND_CODE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("MEND_ROOT", CODE)  # run artifacts: taskq/, outputs/, config/
Q = f"{ROOT}/taskq"
EV = f"{ROOT}/outputs/eval"
FINAL = f"{ROOT}/config/best_mend.final"
PAPER = os.environ.get("MEND_PAPER_DIR", f"{ROOT}/paper")  # figure and table output for the paper source
PRE = f"""source {CODE}/infra/env.sh
source {CODE}/infra/mend_task_lib.sh
export CODE_COMMIT=$(git -C {CODE} rev-parse --short HEAD)
check_requires
task_begin
"""


def header(nodes=1, ngpu=1, pool=None, requires=(), lines=()):
    h = [f"#FLEET NODES={nodes}", f"#FLEET NGPU={ngpu}"]
    if pool:
        h.append(f"#FLEET POOL={pool}")
    h += [f"#REQUIRES {r}" for r in requires]
    h += [f"# {ln}" for ln in lines]
    return "\n".join(h) + "\n"


TASKS = {}  # name -> (text, gpu_h, gpus, output, fills)


def add(name, text, gpu_h, gpus, output, fills):
    TASKS[name] = (text, gpu_h, gpus, output, fills)


# ---------------------------------------------------------------- 43/44 joint objective
def joint(name, run, pareto, est):
    out = f"{ROOT}/outputs/p4/{run}_O"
    grp = f"p4_{run}"
    flags = ["--config.mend.verdict_mode=pareto", "--config.mend.pareto_rewards=('pickscore','clipscore','hpsv2')",
             "--config.mend.pareto_eps=(0.0,)"] if pareto else ["--config.mend.verdict_mode=proximal"]
    what = "Pareto verdict (every component must not fall below its value at x, eps 0)" if pareto else \
        "scalar verdict on the sum (no per-component constraint)"
    txt = header(3, 3, "train", [FINAL], [
        f"P4 joint objective, {what}: MEND on OPSD's sd35_open3 (PickScore/26 + CLIPScore + HPSv2.1, weights 1),",
        "Protocol O 48 x 24, 300 updates as OPSD Table 1 (OPSD 25.51 / 0.333 / 0.389). best_mend.env frozen at first launch.",
        f"Cheap evals of checkpoints 100/200/300, full DrawBench eval of 300 ({grp}_full_c300); logs",
        "mend/open3_accepted_below_*_frac, mend/n_pareto_blocked, mend/feasible_frac, probe/*.",
        f"Est. {est} GH200 GPU-h (3 scorers with gradient; uncertain). Outputs {out}/, {EV}/{grp}_c*.json."])
    body = PRE + f"""OUT={out}
mapfile -t F < <(best_mend_flags $OUT)
F+=({' '.join(shlex.quote(f) for f in flags)})
if is_rank0; then watch_ckpts $OUT 45_{grp}_eval pickscore+clipscore+hpsv2 "100 200 300" 300 {grp} & WATCH_PID=$!; fi
RUN_NAME={grp} WANDB_RUN_GROUP={grp} WANDB_TAGS=open3,protocol_O mend_train open3 $OUT 300 "${{F[@]}}"; rc=$?
[[ -n "${{WATCH_PID:-}}" ]] && kill $WATCH_PID 2>/dev/null
(( rc == 0 )) && WATCH_ONCE=1 watch_ckpts $OUT 45_{grp}_eval pickscore+clipscore+hpsv2 "100 200 300" 300 {grp}
exit $rc
"""
    add(name, txt + body, est, 3, out, "tab:pareto / tab:ext-pareto / T3-p*, fig:qual-pareto")


joint("43_p4_mend_open3_pareto.sh", "mend_open3_pareto", True, 135)
joint("44_p4_mend_open3.sh", "mend_open3", False, 125)

add("49_p4_pareto_regress.sh", header(1, 1, "eval", [f"{EV}/p4_mend_open3_pareto_full_c300.json", f"{EV}/base_opsd.json"], [
    "tab:pareto Regr. column: fraction of samples below the base model (same prompt and seed) on at least one of",
    "PickScore, CLIPScore, HPSv2.1, for the MEND joint runs and the released NFT multi-reward LoRA (12_released_nft_",
    "multireward_O; released OPSD joint checkpoint does not exist, so OPSD Regr. = n/a). CPU-light (~1 min)."]) + PRE + f"""python -m mend.analysis.pareto_regress --base base_opsd --allow_missing \\
  --runs p4_mend_open3_pareto_full_c300,p4_mend_open3_full_c300,nft_multireward_O,opsd_pickscore_O \\
  --out {EV}/pareto_regress.json
""", 0.05, 1, f"{EV}/pareto_regress.json", "tab:pareto Regr., abs-regr-*, T3-pp-regr")

# ---------------------------------------------------------------- 47 reduced method per reward
RED = {"pickscore": 28, "hpsv2": 28, "clipscore": 26, "imagereward": 36, "hpsv3": 60}
for r, est in RED.items():
    grp = f"p4_reduced_{r}"
    out = f"{ROOT}/outputs/p4/reduced_{r}_O"
    txt = header(3, 3, "train", [FINAL], [
        f"P4 reduced method (explicit x + delta proposals + cap + path targets; everything else = best_mend.env), {r},",
        "Protocol O 48 x 24, 100 updates (Table 1 'Reduced' row, fig:curves dashed). Cheap evals 25/50/100, full of 100.",
        f"Est. {est} GH200 GPU-h. Outputs {out}/, {EV}/{grp}_c*.json, {EV}/{grp}_full_c100.json."])
    body = PRE + f"""OUT={out}
mapfile -t F < <(best_mend_flags $OUT)
F+=(--config.mend.proposal=explicit)
if is_rank0; then watch_ckpts $OUT 48_{grp}_eval {r} "25 50 100" 100 {grp} & WATCH_PID=$!; fi
RUN_NAME={grp} WANDB_RUN_GROUP={grp} WANDB_TAGS={r},protocol_O,reduced mend_train {r} $OUT 100 "${{F[@]}}"; rc=$?
[[ -n "${{WATCH_PID:-}}" ]] && kill $WATCH_PID 2>/dev/null
(( rc == 0 )) && WATCH_ONCE=1 watch_ckpts $OUT 48_{grp}_eval {r} "25 50 100" 100 {grp}
exit $rc
"""
    add(f"47_p4_reduced_{r}.sh", txt + body, est, 3, out, f"Table 1 T1-red-{r}, fig:curves, tab:ext-{r}")

# ---------------------------------------------------------------- 52 Z-Image CLIPScore / ImageReward
for r in ("clipscore", "imagereward"):
    grp = f"p5_mend_zimage_{r}"
    out = f"{ROOT}/outputs/p5/mend_zimage_{r}"
    txt = header(4, 4, "train", [FINAL, f"{ROOT}/outputs/p5/mend_zimage_pickscore/checkpoints/checkpoint-25/COMPLETE"], [
        f"P5 Z-Image-Turbo MEND, {r} (1024 px, native 9-step Euler, 48 x 12, 100 updates, 4 GPUs), same as",
        "50_p5_mend_zimage_pickscore with another reward; waits until that run has checkpoint 25 (trainer verified on GPU).",
        f"Evals 51_p5_eval_mend_zimage_{r}_* (9 and 4 steps). Est. ~160 GH200 GPU-h (uncertain, as the PickScore run).",
        f"Outputs {out}/, {EV}/{grp}_*.json."])
    body = PRE + f"""OUT={out}
export BACKBONE=zimage
mapfile -t F < <(best_mend_flags $OUT)
if is_rank0; then watch_ckpts_zimage $OUT 51_{grp} {r} "25 50 100" 100 {grp} & WATCH_PID=$!; fi
RUN_NAME={grp} WANDB_RUN_GROUP={grp} WANDB_TAGS={r},zimage,few_step mend_train {r} $OUT 100 "${{F[@]}}"; rc=$?
[[ -n "${{WATCH_PID:-}}" ]] && kill $WATCH_PID 2>/dev/null
(( rc == 0 )) && WATCH_ONCE=1 watch_ckpts_zimage $OUT 51_{grp} {r} "25 50 100" 100 {grp}
exit $rc
"""
    add(f"52_p5_mend_zimage_{r}.sh", txt + body, 160, 4, out, f"Table 1 T1-z-{'clip' if r == 'clipscore' else 'ir'}, tab:ext-zimage")

# ---------------------------------------------------------------- 60/61 qualitative compare samples
CMP = f"{ROOT}/outputs/compare"
COMPARE = [  # name, lora checkpoint, cfg, requires
    ("mend_pickscore", f"{ROOT}/outputs/g3/mend_pickscore_O/checkpoints/checkpoint-100", 1.0),
    ("reduced_pickscore", f"{ROOT}/outputs/p4/reduced_pickscore_O/checkpoints/checkpoint-100", 1.0),
    ("mend_hpsv2", f"{ROOT}/outputs/p4/mend_hpsv2_O/checkpoints/checkpoint-100", 1.0),
    ("mend_hpsv3", f"{ROOT}/outputs/p4/mend_hpsv3_O/checkpoints/checkpoint-100", 1.0),
    ("mend_clipscore", f"{ROOT}/outputs/p4/mend_clipscore_O/checkpoints/checkpoint-100", 1.0),
    ("mend_imagereward", f"{ROOT}/outputs/p4/mend_imagereward_O/checkpoints/checkpoint-100", 1.0),
    ("mend_open3_pareto", f"{ROOT}/outputs/p4/mend_open3_pareto_O/checkpoints/checkpoint-300", 1.0),
    ("mend_cfg_pickscore", f"{ROOT}/outputs/p4/mend_cfg_pickscore_F/checkpoints/checkpoint-100", 4.5),
]
BASELINES = "opsd_pickscore,opsd_hpsv2,opsd_hpsv3,opsd_clipscore,flowgrpo_pickscore,nft_multireward"
for i, (m, ck, cfg) in enumerate(COMPARE):
    name = f"{60 if i == 0 else 61}_compare_{m}.sh"
    lines = [f"Qualitative compare samples of {m} ({ck}/lora, cfg {cfg:g}) on the 48 compare prompts x 8 seeds",
             f"(data/compare_prompts.txt, 40-step flow ODE, same noise as every other method) -> {CMP}/{m}/, then",
             "rerun collapse_stats (cached embeddings), make_grid rank, mine_failures, and the paper figures that read them",
             "(fig_failures adds the MEND row when compare/mend_pickscore exists). ~0.25 GPU-h."]
    extra = ""
    if m == "mend_pickscore":
        lines.append("Also copies pk15 seeds 0/1 ('a couple dancing lindy hop at street') to paper/figures/img/hero_mend_s{0,1}.png.")
        extra = f"""for s in 0 1; do cp -f {CMP}/{m}/pk15_$s.png {PAPER}/figures/img/hero_mend_s$s.png || true; done
(cd {PAPER}/figures/src && python fig_hero.py) || echo "[compare] fig_hero.py failed (figure not rebuilt)"
"""
    body = PRE + f"""O={CMP}
python -m mend.eval.gen_compare --out_root $O --batch_size 48 --methods {m}={ck}/lora:cfg={cfg:g} || exit 1
python -m mend.tracking.log_bank --out_root $O || echo "[compare] wandb log failed"
python -m mend.analysis.collapse_stats --root $O --bases base,base_cfg4.5 --embedders dreamsim,dinov2,clip --embed_batch_size 64 || exit 1
python -m mend.analysis.make_grid rank --stats $O/collapse_stats.json --baselines {BASELINES} || true
python -m mend.analysis.mine_failures --root $O || true
(cd {PAPER}/figures/src && python fig_failures.py) || echo "[compare] fig_failures.py failed (figure not rebuilt)"
{extra}"""
    add(name, header(1, 1, "eval", [f"{ck}/COMPLETE"], lines) + body, 0.25, 1, f"{CMP}/{m}/",
        "Fig 2 collapse MEND row, tab:failures, hero (b) row" if m == "mend_pickscore" else "app:visuals galleries, collapse stats")

# ---------------------------------------------------------------- 70 design space / P6 ablations
spec = json.load(open(os.path.join(CODE, "configs", "p6_arms.json")))["arms"]
for i, a in enumerate(spec):
    arm = a["arm"]
    seed = a.get("seed", 1)
    flags = " ".join(shlex.quote(f) for f in a["flags"])
    txt = header(1, 1, None, [FINAL, f"{EV}/base_cfg45_cheap64.json"], [
        f"Design-space / P6 arm {arm} (axis {a['axis']}: {a['label']}).",
        "Regime: best_mend.env frozen once for all arms (outputs/ds/launch_config.env) + PickScore, SD3.5-M Protocol O",
        "sampler/LoRA, 16 prompts x 16 images per update, 50 updates, 1 GPU, MEND_SEED fixed (paired arms) + the flags",
        "below. If the resolved config equals the reference it trains nothing (outputs/ds/aliases). Then the cheap eval.",
        f"Est. {a['est']:.1f} GH200 GPU-h. Outputs {ROOT}/outputs/ds/{arm}/, {EV}/ds_{arm}.json."])
    body = PRE + f"MEND_SEED={seed} p6_arm {arm} {flags} || exit 1\n"
    add(f"70_ds_{i:02d}_{arm}.sh", txt + body, a["est"], 1, f"{ROOT}/outputs/ds/{arm}/",
        "app:designspace (fig/tab:ext-design), tab:ablation" + (f" row '{a['paper']}'" if a.get("paper") else ""))

add("79_ds_tables.sh", header(1, 1, "eval", [f"{EV}/ds_ref_seed1.json", f"{EV}/base_opsd_cheap64.json"], [
    "Design-space / ablation tables from whatever arms have finished (rerun after the last arm; idempotent) and the",
    "T3/T5 probe report of the three reference seeds. CPU-light."]) + PRE + f"""python -m mend.analysis.p6_table --out {ROOT}/outputs/ds/ds_table
python -m mend.analysis.theory_report {ROOT}/outputs/ds/ref_seed1 {ROOT}/outputs/ds/ref_seed2 {ROOT}/outputs/ds/ref_seed3 \\
  --names ref_seed1,ref_seed2,ref_seed3 --out {ROOT}/outputs/theory/ds_ref --plot || true
""", 0.05, 1, f"{ROOT}/outputs/ds/ds_table.{{json,csv,tex}}", "tab:ablation, tab:ext-design, fig:ext-designspace")

# ---------------------------------------------------------------- 80 VLM judge
JUDGE = [
    ("g3_mend_pickscore_full_c100", "opsd_pickscore_O"), ("g3_mend_pickscore_full_c100", "nft_multireward_O"),
    ("g3_mend_pickscore_full_c100", "flowgrpo_pickscore_O"), ("g3_mend_pickscore_full_c100", "base_opsd"),
    ("g3_mend_pickscore_full_c100", "p4_reduced_pickscore_full_c100"),
    ("p4_mend_hpsv2_full_c100", "opsd_hpsv2_O"), ("p4_mend_hpsv3_full_c100", "opsd_hpsv3_O"),
    ("p4_mend_clipscore_full_c100", "opsd_clipscore_O"), ("p4_mend_imagereward_full_c100", "nft_multireward_O"),
    ("p4_mend_imagereward_full_c100", "base_opsd"), ("p4_mend_open3_pareto_full_c300", "nft_multireward_O"),
    ("p4_mend_open3_pareto_full_c300", "opsd_pickscore_O"), ("p4_mend_cfg_pickscore_full_c100", "flowgrpo_pickscore_F"),
    ("p4_mend_cfg_pickscore_full_c100", "base_flowgrpo"),
]
PRODUCER = {"opsd_pickscore_O": "12_released_opsd_pickscore_O", "opsd_hpsv2_O": "12_released_opsd_hpsv2_O",
            "opsd_hpsv3_O": "12_released_opsd_hpsv3_O", "opsd_clipscore_O": "12_released_opsd_clipscore_O",
            "nft_multireward_O": "12_released_nft_multireward_O", "flowgrpo_pickscore_O": "12_released_flowgrpo_pickscore_O",
            "flowgrpo_pickscore_F": "12_released_flowgrpo_pickscore_F", "base_opsd": "30_g3_00_base_opsd_full",
            "base_flowgrpo": "30_g3_00_base_opsd_full (run_hf_ref full)"}
for i, (a, b) in enumerate(JUDGE):
    lines = [f"VLM judge (Qwen2.5-VL-7B, both presentation orders, position-debiased, 4 criteria, prompt-bootstrap CI):",
             f"{a} vs {b} on DrawBench 200 x 5 (same prompts, seeds and protocol). Opponent images come from",
             f"{PRODUCER.get(b, 'the MEND eval task of that run')} (deferred 12_* tasks must be released for released baselines).",
             f"~0.6 GH200 GPU-h. Output {ROOT}/outputs/vlm/{a}_vs_{b}.json (cache outputs/eval_images/judge/)."]
    body = PRE + f"""mkdir -p {ROOT}/outputs/vlm
python -m mend.eval.vlm_judge --run_a {a} --run_b {b} --criteria overall,alignment,quality,fidelity \\
  --out {ROOT}/outputs/vlm/{a}_vs_{b}.json
"""
    add(f"80_vlm_{i:02d}_{a}__vs__{b}.sh", header(1, 1, "eval", [f"{EV}/{a}.json", f"{EV}/{b}.json"], lines) + body,
        0.6, 1, f"{ROOT}/outputs/vlm/{a}_vs_{b}.json", "tab:vlm (app:vlm), Table 2 VLM column, abs-vlm")

# ---------------------------------------------------------------- 82 paired held-out / diversity / HF tables
HELD = {
    "pickscore": ("base_opsd", ["g3_mend_pickscore_full_c100", "p4_reduced_pickscore_full_c100", "opsd_pickscore_O",
                                "nft_multireward_O", "flowgrpo_pickscore_O"]),
    "rewards": ("base_opsd", ["p4_mend_hpsv2_full_c100", "opsd_hpsv2_O", "p4_mend_hpsv3_full_c100", "opsd_hpsv3_O",
                              "p4_mend_clipscore_full_c100", "opsd_clipscore_O", "p4_mend_imagereward_full_c100"]),
    "protF": ("base_flowgrpo", ["p4_mend_cfg_pickscore_full_c100", "flowgrpo_pickscore_F"]),
    "vs_opsd_pick": ("opsd_pickscore_O", ["g3_mend_pickscore_full_c100"]),
}
for key, (ref, runs) in HELD.items():
    files = " ".join(f"$E/{r}.json" for r in runs)
    body = PRE + f"""E={EV}
have=(); for f in {files}; do [[ -f $f ]] && have+=($f); done
(( ${{#have[@]}} )) || {{ echo "[table] no run evals yet"; exit 0; }}
python -m mend.eval.bootstrap_table --ref $E/{ref}.json --runs "${{have[@]}}" \\
  --md $E/table_{key}.md --csv $E/table_{key}.csv --tex $E/table_{key}.tex --json $E/table_{key}.json
"""
    first = runs[0]
    add(f"82_table_{key}.sh", header(1, 1, "eval", [f"{EV}/{ref}.json", f"{EV}/{first}.json"], [
        f"Paired prompt-bootstrap table (bootstrap_table.py): every metric incl. diversity (DreamSim, Vendi) and HF ratio,",
        f"runs minus {ref}; runs present at run time among: {' '.join(runs)}. CPU-light; rerun when more evals land."]) + body,
        0.05, 1, f"{EV}/table_{key}.{{md,csv,tex,json}}", "Table 2 (T2-*), tab:heldout-ci, tab:ext-*")

# ---------------------------------------------------------------- 85-88 theory checks and repair steps
G3 = f"{ROOT}/outputs/g3/mend_pickscore_O"
TH = f"{ROOT}/outputs/theory"
add("85_theory_g3_report.sh", header(1, 1, "eval", [f"{G3}/checkpoints/checkpoint-100/COMPLETE"], [
    "T3/T5 on the G3 run (+ PickScore seeds 2/3 when present): realization ratio on training and held-out probe seeds per",
    "round, realized vs certified gain, L_R lower estimate, realized E||m||^2 per round and cumulative vs the T5 budget",
    f"2 tau_max (kappa - E R_k). CPU-light. Outputs {TH}/g3_mend_pickscore.{{json,csv,pdf}} (json 'paper' block = app:diag)."]) + PRE + f"""mkdir -p {TH}
runs=({G3}); names=g3
for s in 2 3; do d={ROOT}/outputs/p4/mend_pickscore_O_seed$s; [[ -f $d/metrics.jsonl ]] && {{ runs+=($d); names+=,seed$s; }}; done
python -m mend.analysis.theory_report "${{runs[@]}}" --names $names --out {TH}/g3_mend_pickscore --plot
""", 0.05, 1, f"{TH}/g3_mend_pickscore.json", "app:diag diag-*, fig:ext-realtheory (b,c,f)")

add("86_theory_mode_mass.sh", header(1, 1, "eval", [f"{CMP}/mend_pickscore/manifest.jsonl"], [
    "T4 mode mass: per compare prompt, DINOv2 clusters of the base samples (8 seeds), paired cluster-mass change of each",
    "method on the same seeds (max |dmass|, TV, dropped modes; prompt-bootstrap CI). CFG-free methods vs base, CFG 4.5",
    f"methods vs base_cfg4.5. ~0.1 GPU-h. Outputs {CMP}/mode_mass.{{json,csv}}, {CMP}/mode_mass_cfg45.*"]) + PRE + f"""O={CMP}
cf=opsd_pickscore,opsd_hpsv2,opsd_hpsv3,opsd_clipscore,nft_multireward,flowgrpo_pickscore_cfg1,mend_pickscore
for m in reduced_pickscore mend_hpsv2 mend_hpsv3 mend_clipscore mend_imagereward mend_open3_pareto; do
  [[ -f $O/$m/manifest.jsonl ]] && cf=$cf,$m
done
python -m mend.analysis.mode_mass --root $O --base base --methods $cf || exit 1
g=flowgrpo_pickscore; [[ -f $O/mend_cfg_pickscore/manifest.jsonl ]] && g=$g,mend_cfg_pickscore
python -m mend.analysis.mode_mass --root $O --base base_cfg4.5 --methods $g --out $O/mode_mass_cfg45
""", 0.1, 1, f"{CMP}/mode_mass.json", "app:diag (T4 cluster-mass change), fig:ext-realtheory (e)")

add("87_theory_seed_displacement.sh", header(1, 1, "eval", [f"{G3}/checkpoints/checkpoint-100/COMPLETE"], [
    "T6 / prop:filter: endpoints of each method from the compare noise (32 prompts x 4 seeds, 40-step flow ODE), inverted",
    "through the BASE model ODE (100 Euler steps): seed displacement, displacement per unit endpoint move, typicality of",
    "the implied seeds. Adds the reduced (explicit) PickScore run when present, for anchored vs explicit at matched protocol.",
    f"~1 GH200 GPU-h. Output {TH}/seed_displacement.json."]) + PRE + f"""mkdir -p {TH}
M="base,opsd_pickscore,nft_multireward,flowgrpo_pickscore_cfg1,mend_pickscore={G3}/checkpoints/checkpoint-100/lora"
R={ROOT}/outputs/p4/reduced_pickscore_O/checkpoints/checkpoint-100
[[ -f $R/COMPLETE ]] && M="$M,reduced_pickscore=$R/lora"
python -m mend.analysis.seed_displacement --methods "$M" --n_prompts 32 --seeds 0,1,2,3 --inv_steps 100 \\
  --out {TH}/seed_displacement.json
""", 1.0, 1, f"{TH}/seed_displacement.json", "app:diag (Filter / seed displacement), fig:ext-realtheory")

DSR = f"{ROOT}/outputs/ds"
add("87b_theory_seed_displacement_ds.sh", header(1, 1, "eval", [f"{DSR}/ref_seed1/checkpoints/checkpoint-50/COMPLETE",
                                                              f"{DSR}/hint_grad_explicit/checkpoints/checkpoint-50/COMPLETE"], [
    "prop:filter at a matched regime: anchored (ref_seed1) vs explicit (hint_grad_explicit) design-space checkpoints,",
    f"same inversion protocol as 87. ~0.5 GH200 GPU-h. Output {TH}/seed_displacement_ds.json."]) + PRE + f"""mkdir -p {TH}
python -m mend.analysis.seed_displacement --n_prompts 32 --seeds 0,1,2,3 --inv_steps 100 \\
  --methods "base,ds_anchored={DSR}/ref_seed1/checkpoints/checkpoint-50/lora,ds_explicit={DSR}/hint_grad_explicit/checkpoints/checkpoint-50/lora" \\
  --out {TH}/seed_displacement_ds.json
""", 0.5, 1, f"{TH}/seed_displacement_ds.json", "app:diag anchored vs explicit seed displacement")

add("88_repair_steps_render.sh", header(1, 1, "eval", [f"{G3}/debug_round_*.pt"], [
    "fig:repair-steps: decode the trainer's debug dumps (debug_round_<step>.pt: held-out probe seeds of one prompt, x,",
    "candidates, rewards, costs, verdict, y*, updated endpoint x + m) with the SD3.5-M VAE into PNGs + summary.json.",
    "Rerun after the G3 run finishes to render every dumped step (1, 5, 25, 50, 100). ~0.05 GPU-h.",
    f"Output {ROOT}/outputs/figures_data/repair_steps/mend_pickscore_O/step<step>/."]) + PRE + f"""python -m mend.analysis.render_repair_steps --run {G3} --out {ROOT}/outputs/figures_data/repair_steps
""", 0.05, 1, f"{ROOT}/outputs/figures_data/repair_steps/", "fig:repair-steps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{Q}/deferred")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="Print the task table (TSV) and exit.")
    a = ap.parse_args()
    if a.list:
        print("task\tgpus\tgpu_h\toutput\tfills")
        for n, (_, h, g, o, f) in sorted(TASKS.items()):
            print(f"{n}\t{g}\t{h:g}\t{o}\t{f}")
        print(f"TOTAL\t\t{sum(v[1] for v in TASKS.values()):.1f}")
        return
    os.makedirs(a.out, exist_ok=True)
    wrote = skipped = 0
    for n, (txt, *_rest) in sorted(TASKS.items()):
        exists = [st for st in ("pending", "running", "done", "failed", "deferred")
                  if os.path.exists(os.path.join(Q, st, n))] + (["out"] if os.path.exists(os.path.join(a.out, n)) else [])
        if exists and not a.force:
            skipped += 1
            continue
        tmp = os.path.join(a.out, f".{n}.tmp")
        with open(tmp, "w") as f:
            f.write(txt)
        os.replace(tmp, os.path.join(a.out, n))
        wrote += 1
    print(f"[results_tasks] wrote {wrote}, skipped {skipped} existing, into {a.out}")


if __name__ == "__main__":
    main()
