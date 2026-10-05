# Prompt data

- `pickapic_recipe.json`: recipe for the 25,415-prompt Pick-a-Pic training split (the prompt set of the DiffusionOPSD protocol, reused unchanged so that MEND and the baselines train on identical prompts). It pins a revision of the text-only [`sayakpaul/pick-a-pic-v2-unique-prompts`](https://huggingface.co/datasets/sayakpaul/pick-a-pic-v2-unique-prompts) dataset, keeps the original prompt order, and records the expected SHA-256.
- `drawbench/test.txt`: DrawBench prompts for held-out evaluation.
- `showcase_prompts.tsv`, `hires_prompts.tsv`, `compare_prompts.txt`: prompt lists for qualitative figures and side-by-side comparisons.

Materialize the training split (about 3 MB of text; no Pick-a-Pic images are downloaded):

```bash
python -m mend.data.prepare_pickapic
```

This writes `data/pickapic/{train,test}.txt`, which git ignores. `scripts/train_mend.sh` runs it automatically. To train on another one-prompt-per-line file, pass `--config.dataset=/path/to/dataset` to the trainer.
