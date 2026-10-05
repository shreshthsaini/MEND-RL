"""Run mend/train/zimage.py end to end on CPU with a tiny fake Z-Image pipeline and a fake reward.

Used by tests/test_mend_zimage_trainer_cpu.py (one process, gloo, world 1). Everything MEND does is real: the
zimage_rollout loop, PEFT LoRA "default"/"old" adapters on to_q/to_k/to_v/to_out.0/w1/w2/w3, the hint through a
decoder, cap, Euler-restart anchored proposals, verdict, displaced-path and keep losses, AdamW, EMA, the old-adapter
update, checkpoint save and resume. Only the networks and the reward are small stand-ins.

    python tests/fake_zimage_run.py --config configs/mend.py:zimage_pickscore --config.save_dir=... [flags]
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

from mend.train import zimage as tz  # noqa: E402

C_LAT, HID, CAP = 4, 16, 8


def _seed_of(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


class FakeS3DiT(torch.nn.Module):
    """List-based forward(x_list, t, cap_feats, return_dict) like ZImageTransformer2DModel, with LoRA-targetable
    Linear layers named as in ZIMAGE_LORA_TARGETS."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.proj_in = torch.nn.Linear(C_LAT, HID)
        self.t_emb = torch.nn.Linear(1, HID)
        self.cap_proj = torch.nn.Linear(CAP, HID)
        self.to_q = torch.nn.Linear(HID, HID)
        self.to_k = torch.nn.Linear(HID, HID)
        self.to_v = torch.nn.Linear(HID, HID)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(HID, HID)])
        self.w1 = torch.nn.Linear(HID, 2 * HID)
        self.w3 = torch.nn.Linear(HID, 2 * HID)
        self.w2 = torch.nn.Linear(2 * HID, HID)
        self.proj_out = torch.nn.Linear(HID, C_LAT)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.15)
        self.config = type("Cfg", (), {"in_channels": C_LAT, "to_dict": lambda self: {"in_channels": C_LAT}})()
        self.gc = False

    @property
    def dtype(self):
        return self.proj_in.weight.dtype

    def enable_gradient_checkpointing(self):
        self.gc = True

    def forward(self, x_list, t, cap_feats, return_dict=False):
        x = torch.stack(x_list, 0).squeeze(2)  # (B, C, H, W)
        B, C, H, W = x.shape
        h = self.proj_in(x.flatten(2).transpose(1, 2))  # (B, HW, HID)
        cap = torch.stack([c.float().mean(0) for c in cap_feats]).to(h.dtype)
        h = h + self.t_emb(t.view(B, 1).to(h.dtype)).unsqueeze(1) + self.cap_proj(cap).unsqueeze(1)
        q, k, v = self.to_q(h), self.to_k(h), self.to_v(h)
        att = torch.softmax(q @ k.transpose(1, 2) / HID ** 0.5, dim=-1) @ v
        h = h + self.to_out[0](att)
        h = h + self.w2(torch.nn.functional.silu(self.w1(h)) * self.w3(h))
        out = 0.5 * self.proj_out(h).transpose(1, 2).reshape(B, C, H, W) + 0.8 * x  # v ~ eps - x0 scale
        return (list(out.unsqueeze(2).unbind(0)),)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = torch.nn.Conv2d(C_LAT, 3, 3, padding=1)
        self.config = type("Cfg", (), {"scaling_factor": 0.3611, "shift_factor": 0.1159})()

    @property
    def dtype(self):
        return self.conv.weight.dtype

    def enable_gradient_checkpointing(self):
        pass

    def enable_tiling(self):
        pass

    def decode(self, lat, return_dict=False):
        return (torch.tanh(self.conv(self.up(lat))),)


class FakeTokenizer:
    def __call__(self, prompts, padding=None, max_length=256, truncation=True, return_tensors="pt"):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            b = list(p.encode())[:max_length]
            ids[i, :len(b)] = torch.tensor(b)
        return type("Tok", (), {"input_ids": ids})()

    def batch_decode(self, ids, skip_special_tokens=True):
        return [bytes([int(c) for c in row if int(c) > 0]).decode() for row in ids]


class FakeZImagePipeline:
    def __init__(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.transformer = FakeS3DiT()
        self.vae = FakeVAE()
        self.text_encoder = torch.nn.Linear(1, 1)
        self.tokenizer = FakeTokenizer()
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0,
                                                         use_dynamic_shifting=False)

    def set_progress_bar_config(self, **kw):
        pass

    def prepare_latents(self, B, C, H, W, dtype, device, generator):
        return torch.randn(B, C, H // 8, W // 8, generator=generator, device=device, dtype=torch.float32)

    def encode_prompt(self, prompt, device, do_classifier_free_guidance=False, max_sequence_length=512):
        out = []
        for p in prompt:
            g = torch.Generator().manual_seed(_seed_of(p))
            out.append(torch.randn(3 + len(p) % 4, CAP, generator=g))
        return out, None


class FakeScorer:
    device = torch.device("cpu")
    dtype = torch.float32


def fake_reward_scores_grad(scorer, kind, images01, prompts):
    """Differentiable: closeness of the image to a prompt-specific colour (higher is better)."""
    tgt = torch.stack([torch.rand(3, generator=torch.Generator().manual_seed(_seed_of(p))) for p in prompts])
    return -((images01 - tgt.view(-1, 3, 1, 1)) ** 2).mean(dim=(1, 2, 3)) * 10.0


def fake_make_reward_fn(device, reward_cfg):
    def reward_fn(images, prompts, metadata, only_strict=True):
        with torch.no_grad():
            r = fake_reward_scores_grad(None, "fake", images.float(), prompts).cpu().numpy()
        return {"avg": r, "pickscore": r}, {}
    return reward_fn


def install_fakes():
    tz.load_pipeline = lambda config, dtype: FakeZImagePipeline()
    tz.make_reward_fn = fake_make_reward_fn
    tz._load_reward_scorer = lambda kind, device: FakeScorer()
    tz._reward_scores_grad = fake_reward_scores_grad


if __name__ == "__main__":
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WANDB_MODE", "disabled")
    torch.set_num_threads(4)
    np.random.seed(0)
    install_fakes()
    from absl import app

    def _main(argv):
        tz.FLAGS.config.seed = int(os.environ.get("FAKE_SEED", "0"))  # None-typed field: not settable by flag
        tz.main(argv)

    app.run(_main)
