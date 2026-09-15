"""Adapt Emuru to one writer: a short LoRA fine-tune on that writer's own lines.

Zero-shot, Emuru is shown one style line. With quality control it carries 65% of
a writer's identity (T28), and it cannot be shown more: two lines joined side by
side broke its text (T27). A person enrolling brings a page, twenty-odd lines.
Training is the other way to use them -- show the model each of the writer's
lines and move it a little towards them.

**What one training step is**, read from Emuru's own code. ``Emuru.forward(img,
input_ids, attention_mask, noise)`` encodes a line image with the frozen VAE into
latent slices, one per 8px of width, and asks T5 to predict each slice from the
text and the slices before it. The loss is the mean squared error between the
predicted slices and the real ones. Emuru's ``train_T5.py`` trains on single lines
with their full text, so a writer's training data is simply their lines and
transcriptions -- which is what a dictated page provides.

**LoRA, not the whole model.** Emuru's T5 is t5-large: 24 encoder and 24 decoder
layers, d_model 1024, about 0.7B parameters. Training all of it with AdamW needs
about 16 bytes a parameter before activations, more than a T4 holds with room to
spare, and would leave a 3 GB copy per user. LoRA -- low-rank adaptation --
freezes every original weight and learns, beside each chosen weight matrix W, a
correction B·A where A and B are two small matrices of rank r. On T5's attention
projections at rank 8 that is 4.72M trainable parameters of 719M, 0.66%, and a
writer's adaptation is 19 MB in fp32 -- measured on the released checkpoint.

**White after every training line.** Emuru decides a line has ended when it emits
ten consecutive slices resembling its padding token, and it learned that from
training lines padded with white to a fixed 768px. Fine-tuning on tightly cropped
lines would teach it that writing never stops. So each line is given white space
after its last stroke.

**One adapter, reset between writers.** The adapter is attached once. Moving to
the next writer re-initialises it with B at zero, which makes every correction
exactly zero and the model exactly the released one again -- nothing reloaded.

The defaults below are starting points, not measurements. T29 measures them.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

NATIVE_HEIGHT = 64
PIXELS_PER_SLICE = 8
"""The VAE's width compression: one latent slice, and one generated token, per 8px."""

TRAILING_WHITE = 128
"""White pixels after each training line: sixteen slices, past the ten that the
stopping criterion looks for."""

LORA_TARGETS = ("q", "k", "v", "o")
"""T5's attention projections, in self- and cross-attention alike: the weights
that decide what each position attends to, in the text and in the line so far."""

LORA_MARKER = "lora_"


@dataclass(frozen=True)
class FinetuneConfig:
    rank: int = 8
    alpha: int = 16
    """LoRA scales its correction by alpha / rank."""

    dropout: float = 0.05
    learning_rate: float = 2e-4
    steps: int = 150
    batch_size: int = 2

    noise: float = 0.1
    """Gaussian noise on the teacher-forced slices, as Emuru's own training uses
    (``--teacher_noise`` 0.1). Without it the model learns to lean on perfect
    previous slices, which it never has while generating."""

    max_grad_norm: float = 1.0
    seed: int = 0


DEFAULT_CONFIG = FinetuneConfig()


def prepare_line(
    image: np.ndarray,
    height: int = NATIVE_HEIGHT,
    trailing: int = TRAILING_WHITE,
) -> np.ndarray:
    """A grayscale line, ink dark on white, as the model trains on it.

    Scaled to ``height``, given ``trailing`` white pixels after the text, widened
    to a whole number of slices, and mapped to [-1, 1] in three channels -- the
    same convention ``EmuruGenerator`` feeds the model at generation time.
    """
    gray = np.asarray(image)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"expected a non-empty grayscale line, got shape {gray.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {gray.dtype}")

    if gray.shape[0] != height:
        scale = height / gray.shape[0]
        gray = cv2.resize(
            gray,
            (max(1, round(gray.shape[1] * scale)), height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )

    width = math.ceil((gray.shape[1] + trailing) / PIXELS_PER_SLICE) * PIXELS_PER_SLICE
    canvas = np.full((height, width), 255, dtype=np.uint8)
    canvas[:, : gray.shape[1]] = gray
    array = canvas.astype(np.float32) / 127.5 - 1.0
    return np.repeat(array[None], 3, axis=0)


def pad_batch(lines: Sequence[np.ndarray]) -> np.ndarray:
    """Prepared lines stacked into one batch, the narrower ones extended with white.

    White is also what follows every line, so padding a short line out to a long
    one's width teaches nothing false.
    """
    if not lines:
        raise ValueError("empty batch")
    width = max(line.shape[-1] for line in lines)
    batch = np.ones((len(lines), *lines[0].shape[:-1], width), dtype=np.float32)
    for index, line in enumerate(lines):
        batch[index, ..., : line.shape[-1]] = line
    return batch


def attach_lora(model, config: FinetuneConfig = DEFAULT_CONFIG, device=None) -> int:
    """Freeze the model and give its T5 a LoRA adapter. Returns trainable parameters.

    Once per model: :func:`reset_lora` prepares it for the next writer.
    """
    from peft import LoraConfig, inject_adapter_in_model

    _ignore_stale_torchao()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(LORA_TARGETS),
    )
    inject_adapter_in_model(lora, model.T5)
    if device is not None:
        model.to(device)

    trainable = [p for name, p in model.named_parameters() if LORA_MARKER in name]
    for parameter in trainable:
        parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in trainable)


def _ignore_stale_torchao() -> bool:
    """Stop an old, unused torchao from blocking LoRA. Returns whether it had to.

    For every layer it adapts, peft asks each of its dispatchers whether the
    layer is theirs. The torchao dispatcher answers by checking torchao's version,
    and raises when an old one is installed -- which Colab does, 0.10.0 against a
    required 0.16.0 -- even for a model with no torchao weights in it. The first
    cell 7d run died there, before training a step. Emuru's weights are ordinary
    tensors, so the true answer is always "not mine", and it is given directly.
    """
    try:
        from peft.import_utils import is_torchao_available

        is_torchao_available()
    except ImportError as exc:
        if "torchao" not in str(exc):
            raise
    else:
        return False

    import peft.tuners.lora.torchao as dispatch

    dispatch.is_torchao_available = lambda: False
    return True


def reset_lora(model) -> int:
    """Re-initialise every LoRA adapter, which returns the model to its released
    behaviour exactly. Returns how many adapted layers were reset."""
    import torch

    count = 0
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "reset_lora_parameters"):
                for adapter in list(module.lora_A.keys()):
                    module.reset_lora_parameters(adapter, init_lora_weights=True)
                count += 1
    return count


def lora_state(model) -> dict:
    """The adapter's weights alone: what would be stored for one writer."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if LORA_MARKER in name
    }


@dataclass
class TrainReport:
    steps: int
    seconds: float
    losses: list[float] = field(default_factory=list)

    def summary(self) -> str:
        head = float(np.mean(self.losses[: max(1, len(self.losses) // 10)]))
        tail = float(np.mean(self.losses[-max(1, len(self.losses) // 10) :]))
        return (
            f"trained {self.steps} steps in {self.seconds:.0f}s, "
            f"loss {head:.4f} -> {tail:.4f} (first and last tenth)"
        )


def train_writer(
    model,
    images: Sequence[np.ndarray],
    texts: Sequence[str],
    config: FinetuneConfig = DEFAULT_CONFIG,
    device: str = "cpu",
) -> TrainReport:
    """Fine-tune the attached adapter on one writer's lines and their texts.

    Lines are drawn in shuffled passes, so every line is seen before any repeats.
    The model is left in eval mode, ready to generate.
    """
    import torch

    if not images or len(images) != len(texts):
        raise ValueError(f"{len(images)} lines for {len(texts)} texts")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("nothing to train: call attach_lora first")

    prepared = [prepare_line(image) for image in images]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    model.train()
    model.vae.eval()
    losses: list[float] = []
    order: list[int] = []
    started = time.perf_counter()

    for step in range(config.steps):
        size = min(config.batch_size, len(prepared))
        while len(order) < size:
            order.extend(rng.sample(range(len(prepared)), len(prepared)))
        picked = [order.pop() for _ in range(size)]

        batch = torch.from_numpy(pad_batch([prepared[i] for i in picked])).to(device)
        tokens = model.tokenizer([texts[i] for i in picked], return_tensors="pt", padding=True)
        loss, _, _ = model(
            batch,
            input_ids=tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
            noise=config.noise,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became {loss.item()} at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    model.eval()
    return TrainReport(steps=config.steps, seconds=time.perf_counter() - started, losses=losses)
