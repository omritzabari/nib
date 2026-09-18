"""Adapt DiffBrush to one writer: a short LoRA fine-tune on that writer's lines.

Zero-shot, DiffBrush has only IAM's hands to draw on, and IAM-trained models lose
their imitation off IAM. So each writer's page is used as training data: a few
hundred steps that move the model towards this hand, then generation as usual.

**Why this can work where Emuru's fine-tune did not.** Emuru copies the hand from
the style line in front of the text, so its weights never needed to hold it: a
fine-tune on the writer's own lines changed nothing (7d), and withholding the
hand broke the copying and cost 17.9 identity points (7f). DiffBrush has no line
in front to copy. Each training step hides the writer's line under noise and asks
the model to recover it from the text and *a different line* by the same writer;
the only way to lower that loss is for the weights to learn the hand. That is the
setting in which few-image adaptation of image diffusion models is routine. Whether
it carries to handwriting from 10-22 lines is what stage 3 measures.

**What one step is.** DiffBrush's released code includes no training script, so
the step is the standard one its sampler implies: encode the line with Stable
Diffusion's VAE (times 0.18215), draw a timestep from its linear schedule -- 1,000
steps, beta from 1e-4 to 0.02, as its ``Diffusion`` class sets up -- mix in that
much Gaussian noise, and take the mean squared error between the noise and the
UNet's prediction of it. The UNet is called as the sampler calls it, without the
two proxy losses that need an IAM writer id.

**LoRA on the UNet's attention.** Every ``CrossAttention`` has ``to_q``, ``to_k``,
``to_v`` and ``to_out.0``; those are adapted and nothing else is trained. The
adapter is reset between writers with :func:`nib.models.finetune.reset_lora`, which
returns the model to the released one exactly.

**Lines wider than the canvas are squeezed.** DiffBrush trained on 64 x 1024
canvases. A wider line is resized to 1024 wide, which narrows its letters; the
alternative, cutting it, would pair an image with text it does not show. Squeezed
lines are counted in the report. Five of the ten lines Amri's page learns from
are wider, by up to 17%.

The defaults are starting points, not measurements.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from nib.models.diffbrush import CANVAS_WIDTH, NATIVE_HEIGHT, encode_text, prepare_style
from nib.models.finetune import TrainReport, _ignore_stale_torchao

LORA_SCOPES = {
    "style": ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"),
    "all": ("to_q", "to_k", "to_v", "to_out.0"),
}
"""Which attention projections the adapter sits on.

Each of DiffBrush's transformer blocks holds three attentions: ``attn1`` over the
drawing alone, ``attnc`` over the conditioning, and ``attn2`` which ties the
drawing to the style and to the glyphs. ``all`` adapts every one of them, which is
what T37's run did: identity fell and CER went from 12.1% to 34.6% -- the words
came out as the right words drawn as mush, so the layers that carry the text had
moved too. ``style`` leaves those alone."""

LORA_MARKER = "lora_"

NOISE_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
"""DiffBrush's linear noise schedule, from its ``Diffusion`` class."""

LATENT_SCALE = 0.18215
"""Stable Diffusion 1.5's latent scaling; DiffBrush's sampler divides by it."""


@dataclass(frozen=True)
class DiffBrushFinetuneConfig:
    rank: int = 8
    alpha: int = 8
    """LoRA scales its correction by alpha / rank: 1 here."""

    dropout: float = 0.0
    learning_rate: float = 2e-5
    steps: int = 60
    """Sixty steps at 2e-5, after 300 at 1e-4 blurred the letters (T37). Sixteen
    lines at batch 2 is seven or eight passes over each, not thirty-seven."""

    batch_size: int = 2
    scope: str = "style"
    skip_wide: bool = True
    """Leave out training lines wider than the canvas rather than squeezing them:
    67 of T37's 384 lines were squeezed, up to 13 of one writer's 16."""

    max_grad_norm: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rank < 1 or self.steps < 1 or self.batch_size < 1:
            raise ValueError(f"rank, steps and batch_size must be positive: {self}")
        if self.scope not in LORA_SCOPES:
            raise ValueError(f"scope must be one of {tuple(LORA_SCOPES)}, got {self.scope!r}")


DEFAULT_CONFIG = DiffBrushFinetuneConfig()


@dataclass
class DiffBrushTrainReport(TrainReport):
    squeezed: int = 0
    """Training lines wider than the canvas, resized to fit."""

    skipped: int = 0
    """Training lines left out for being wider than the canvas."""


def prepare_target(
    image: np.ndarray, height: int = NATIVE_HEIGHT, width: int = CANVAS_WIDTH
) -> tuple[np.ndarray, bool]:
    """A training line on DiffBrush's canvas: three channels in [-1, 1], ink dark,
    the line at the left and white after it. Returns whether it had to be squeezed."""
    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError(f"expected a grayscale line, got shape {gray.shape}")
    if gray.shape[0] != height:
        scale = height / gray.shape[0]
        gray = cv2.resize(
            gray,
            (max(1, round(gray.shape[1] * scale)), height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    squeezed = gray.shape[1] > width
    if squeezed:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width), 255, np.uint8)
    canvas[:, : gray.shape[1]] = gray
    array = canvas.astype(np.float32) / 127.5 - 1.0
    return np.repeat(array[None], 3, axis=0), squeezed


def pick_style(rng: random.Random, target: int, count: int) -> int:
    """Another of the writer's lines to show as style while learning ``target`` --
    never the line itself, or the model could learn to copy rather than to write."""
    if count < 2:
        raise ValueError("a writer needs at least two lines: one to learn, one as style")
    choice = rng.randrange(count - 1)
    return choice if choice < target else choice + 1


def pad_contents(contents: Sequence[np.ndarray]) -> np.ndarray:
    """Glyph sequences as one batch, the shorter ones padded with the blank glyph --
    which, inverted as every glyph is, is all ones. As DiffBrush's own collate does."""
    length = max(content.shape[0] for content in contents)
    batch = np.ones((len(contents), length, *contents[0].shape[1:]), np.float32)
    for index, content in enumerate(contents):
        batch[index, : content.shape[0]] = content
    return batch


def pad_styles(styles: Sequence[np.ndarray]) -> np.ndarray:
    """Style lines as one batch of one channel, the narrower ones extended with white."""
    width = max(style.shape[1] for style in styles)
    batch = np.ones((len(styles), 1, styles[0].shape[0], width), np.float32)
    for index, style in enumerate(styles):
        batch[index, 0, :, : style.shape[1]] = style
    return batch


def attach_lora(unet, config: DiffBrushFinetuneConfig = DEFAULT_CONFIG, device=None) -> int:
    """Freeze the UNet and give its attention projections a LoRA adapter. Returns
    the number of trainable parameters. Once per model; reset between writers."""
    from peft import LoraConfig, inject_adapter_in_model

    _ignore_stale_torchao()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    # DiffBrush's transformer blocks checkpoint by default, with their own function
    # that asks for a gradient on every parameter of the block. Once the base
    # weights are frozen that raises on the first backward pass -- the first CPU run
    # of stage 3 died there -- so the blocks run plainly, at some cost in memory.
    for module in unet.modules():
        if isinstance(getattr(module, "checkpoint", None), bool):
            module.checkpoint = False
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(LORA_SCOPES[config.scope]),
    )
    inject_adapter_in_model(lora, unet)
    if device is not None:
        unet.to(device)
    trainable = [p for name, p in unet.named_parameters() if LORA_MARKER in name]
    for parameter in trainable:
        parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in trainable)


def train_writer(
    unet,
    vae,
    images: Sequence[np.ndarray],
    texts: Sequence[str],
    glyphs: np.ndarray,
    config: DiffBrushFinetuneConfig = DEFAULT_CONFIG,
    device: str = "cpu",
) -> DiffBrushTrainReport:
    """Fine-tune the attached adapter on one writer's lines and their texts.

    Each line's latent is encoded once; every step samples from its posterior,
    picks another line by the writer as style, noises the latent to a random
    timestep and trains the adapter to predict the noise. Lines are drawn in
    shuffled passes. The UNet is left in eval mode, ready to generate.
    """
    import torch

    if len(images) != len(texts):
        raise ValueError(f"{len(images)} lines for {len(texts)} texts")
    if len(images) < 2:
        raise ValueError("a writer needs at least two lines: one to learn, one as style")
    parameters = [p for p in unet.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("nothing to train: call attach_lora first")

    rng = random.Random(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    alpha_hat = torch.cumprod(1.0 - torch.linspace(BETA_START, BETA_END, NOISE_STEPS), dim=0)

    prepared = [prepare_target(image) for image in images]
    keep = list(range(len(images)))
    skipped = 0
    if config.skip_wide:
        narrow = [i for i in keep if not prepared[i][1]]
        # A writer whose lines are nearly all wide would otherwise be left with
        # nothing to learn from; a squeezed line beats no training at all.
        if len(narrow) >= 2:
            skipped = len(keep) - len(narrow)
            keep = narrow
    prepared = [prepared[i] for i in keep]
    images = [images[i] for i in keep]
    texts = [texts[i] for i in keep]
    squeezed = sum(was_squeezed for _, was_squeezed in prepared)
    with torch.no_grad():
        canvases = torch.from_numpy(np.stack([canvas for canvas, _ in prepared])).to(device)
        posteriors = [vae.encode(canvases[i : i + 1]).latent_dist for i in range(len(prepared))]
        means = torch.cat([posterior.mean for posterior in posteriors])
        stds = torch.cat([posterior.std for posterior in posteriors])
    styles = [prepare_style(image) for image in images]
    contents = [encode_text(text, glyphs) for text in texts]

    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    unet.train()
    losses: list[float] = []
    order: list[int] = []
    started = time.perf_counter()

    for step in range(config.steps):
        size = min(config.batch_size, len(images))
        while len(order) < size:
            order.extend(rng.sample(range(len(images)), len(images)))
        picked = [order.pop() for _ in range(size)]

        latent_noise = torch.randn(means[picked].shape, generator=generator).to(device)
        x0 = (means[picked] + stds[picked] * latent_noise) * LATENT_SCALE
        timesteps = torch.randint(0, NOISE_STEPS, (size,), generator=generator)
        noise = torch.randn(x0.shape, generator=generator).to(device)
        blend = alpha_hat[timesteps].to(device)[:, None, None, None]
        noisy = blend.sqrt() * x0 + (1.0 - blend).sqrt() * noise

        style = pad_styles([styles[pick_style(rng, i, len(images))] for i in picked])
        content = pad_contents([contents[i] for i in picked])
        predicted = unet(
            noisy,
            timesteps.to(device),
            torch.from_numpy(style).to(device),
            torch.from_numpy(content).to(device),
            tag="test",
        )
        loss = torch.nn.functional.mse_loss(predicted, noise)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became {loss.item()} at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    unet.eval()
    return DiffBrushTrainReport(
        steps=config.steps,
        seconds=time.perf_counter() - started,
        losses=losses,
        squeezed=squeezed,
        skipped=skipped,
    )
