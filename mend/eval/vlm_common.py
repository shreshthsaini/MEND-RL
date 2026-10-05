"""Qwen2.5-VL helpers shared by the UnifiedReward scorer and the pairwise VLM judge.

Both run in plain transformers (4.51) with no sglang/vLLM server, so they work inside a fleet task on one GPU.
Nothing here is imported at module load by the training code.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

# Qwen2.5-VL judge used for the pairwise preference study.
JUDGE_MODEL = os.environ.get("MEND_JUDGE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# UnifiedReward variant. Flow-GRPO and DiffusionNFT score "UnifiedReward" with CodeGoat24/UnifiedReward-7b-v1.5
# (a LLaVA-OneVision checkpoint served by sglang with the chatml-llava template). That stack does not run in
# plain transformers on aarch64, so we use the same authors' Qwen2.5-VL port with the identical point-score
# prompt. Absolute numbers are therefore NOT directly comparable to the UniRwd column in those papers; all rows
# we report are scored with this one model, so comparisons within our tables are consistent.
UNIFIEDREWARD_MODEL = os.environ.get("MEND_UNIFIEDREWARD_MODEL", "CodeGoat24/UnifiedReward-qwen-7b")

# Verbatim point-score prompt from flow_grpo/DiffusionNFT unifiedreward_scorer.py (minus the LLaVA "<image>\n"
# prefix, which the Qwen chat template replaces with its own vision tokens).
UNIFIEDREWARD_PROMPT = (
    "You are given a text caption and a generated image based on that caption. Your task is to evaluate this "
    "image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the "
    "provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described "
    "in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail "
    "preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from "
    "1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
)
_FINAL_SCORE_RE = re.compile(r"Final Score:\s*([1-5](?:\.\d+)?)")


def parse_final_score(text: str) -> float:
    """Parse ``Final Score: x`` (x in [1, 5]); NaN when absent (flow_grpo silently used 0.0 instead)."""
    m = _FINAL_SCORE_RE.search(text or "")
    return float(m.group(1)) if m else float("nan")


def to_pil_list(images) -> List[Image.Image]:
    """Accept a list of PIL images, or a ``[B,3,H,W]`` float tensor in [0, 1]."""
    if isinstance(images, torch.Tensor):
        arr = (images.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        return [Image.fromarray(a) for a in arr.transpose(0, 2, 3, 1)]
    return [im.convert("RGB") for im in images]


def load_qwen25_vl(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                   max_pixels: int = 512 * 512, min_pixels: int = 256 * 28 * 28):
    """Load a Qwen2.5-VL checkpoint (base or fine-tune) and its processor, left-padded for batched decoding."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).to(device).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"
    return model, processor


class UnifiedRewardScorer:
    """Pointwise UnifiedReward score in [1, 5] via greedy decoding of the ``Final Score:`` template."""

    def __init__(self, device: str = "cuda", model_id: str = UNIFIEDREWARD_MODEL, max_new_tokens: int = 256,
                 dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model, self.processor = load_qwen25_vl(model_id, device=device, dtype=dtype)
        self.last_texts: List[str] = []

    @torch.no_grad()
    def __call__(self, images, prompts: Sequence[str]) -> np.ndarray:
        pils = to_pil_list(images)
        texts, img_inputs = [], []
        for im, p in zip(pils, prompts):
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": UNIFIEDREWARD_PROMPT.format(prompt=p)},
            ]}]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            img_inputs.append(im)
        inputs = self.processor(text=texts, images=img_inputs, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)
        self.last_texts = decoded
        return np.array([parse_final_score(t) for t in decoded], dtype=np.float64)


# ----------------------------------------------------------------------------------------------------------
# Pairwise judge
# ----------------------------------------------------------------------------------------------------------
JUDGE_CRITERIA = {
    "overall": ("Which image is better overall? Consider both how faithfully it follows the text prompt and its "
                "visual quality (realism, detail, absence of artifacts, aesthetics)."),
    "alignment": "Which image follows the text prompt more faithfully (objects, counts, attributes, relations, text)?",
    "quality": ("Which image has higher visual quality (sharpness without artificial over-sharpening, natural "
                "textures, absence of artifacts, aesthetics)? Ignore the prompt."),
    "fidelity": ("Which image looks more like a natural, artifact-free picture? Penalise unnatural high-frequency "
                 "texture, noise, oversaturation, and repeated patterns."),
}

JUDGE_TEMPLATE = (
    "You are an expert judge of text-to-image generation. You are shown two images generated for the same "
    "text prompt.\nText prompt: \"{prompt}\"\nImage 1 is the first image and Image 2 is the second image.\n"
    "{criterion}\nAnswer with a single digit: 1 if Image 1 is better, 2 if Image 2 is better."
)


def judge_messages(prompt: str, img_first: Image.Image, img_second: Image.Image, criterion: str) -> list:
    text = JUDGE_TEMPLATE.format(prompt=prompt, criterion=JUDGE_CRITERIA[criterion])
    return [{"role": "user", "content": [
        {"type": "text", "text": "Image 1:"},
        {"type": "image", "image": img_first},
        {"type": "text", "text": "Image 2:"},
        {"type": "image", "image": img_second},
        {"type": "text", "text": text},
    ]}]


def choice_token_ids(tokenizer) -> List[int]:
    """Token ids for the answers "1" and "2" (single tokens in the Qwen2 vocabulary)."""
    ids = []
    for s in ("1", "2"):
        tok = tokenizer.encode(s, add_special_tokens=False)
        if len(tok) != 1:
            raise RuntimeError(f"answer {s!r} is not a single token: {tok}")
        ids.append(tok[0])
    return ids


class PairwiseJudge:
    """P(first image preferred) from the next-token distribution restricted to {"1", "2"}.

    Reading the two answer logits (instead of sampling text) gives a calibrated soft preference and never fails
    to parse. ``p_first = softmax([l_1, l_2])[0]``.
    """

    def __init__(self, device: str = "cuda", model_id: str = JUDGE_MODEL, dtype: torch.dtype = torch.bfloat16,
                 max_pixels: int = 512 * 512):
        self.device = device
        self.model_id = model_id
        self.model, self.processor = load_qwen25_vl(model_id, device=device, dtype=dtype, max_pixels=max_pixels)
        self.ids = choice_token_ids(self.processor.tokenizer)

    @torch.no_grad()
    def p_first(self, prompts: Sequence[str], firsts: Sequence[Image.Image], seconds: Sequence[Image.Image],
                criterion: str = "overall") -> np.ndarray:
        texts, imgs = [], []
        for p, a, b in zip(prompts, firsts, seconds):
            msgs = judge_messages(p, a, b, criterion)
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.extend([a, b])
        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inputs, use_cache=False).logits[:, -1, :]  # left padding -> last position is real
        two = logits[:, self.ids].float()
        return torch.softmax(two, dim=-1)[:, 0].cpu().numpy().astype(np.float64)


def free_model(obj: Optional[object]) -> None:
    import gc

    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
