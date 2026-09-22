"""Mend the strokes Emuru breaks: a small network that may only add ink.

People tell a generated line at a glance, without looking at the writer: its
strokes are not whole. Measured teacher-forced, the breaks are already in Emuru's
own predictions -- 8.09 pieces of ink per 100 columns against 3.34 on the real
line, 3.81 after the VAE alone -- the regression's average of what it is unsure
of. Pushing the predicted latents away from paper closes them but thickens every
stroke by 65%, and closing the image thickens too (PROGRESS.md, 2026-09-22).
Mending a break without thickening needs to know what a stroke looks like.

**How it learns.** Real lines by the training-split writers are broken the way
Emuru breaks them, through Emuru's own VAE: each line's latents are drawn toward
the latent of blank paper by a smooth random factor (:func:`weakening`), then
decoded. At 0.85-1.0 the broken lines have 5.17 pieces per 100 columns where
Emuru's generated lines have 5.18. The network sees the broken line and learns
the real one.

**It may only add ink.** Its output is the darker of its prediction and its
input, pixel by pixel, so it can join and darken a stroke but never erase or
move one: letterforms stay the model's. A U-Net: an encoder that halves the
image twice to see a stroke's surroundings, a decoder back to full size, and
skip connections that carry the fine detail across.
"""

from __future__ import annotations

import numpy as np

WEAKEN = (0.82, 1.0)
"""The range each line's latents are drawn toward paper, from 1.0 (untouched)
down. Chosen so the broken lines match Emuru's pieces per 100 columns (see the
module docstring); lines near 1.0 teach the network to leave whole strokes alone."""

SPREAD = (0.05, 0.3)
"""How much the factor wanders along a line, so some places break and others not."""


def weakening(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """A factor per latent position: a level for the line, a smooth wander along it."""
    level = rng.uniform(*WEAKEN)
    spread = rng.uniform(*SPREAD)
    wander = rng.normal(0.0, spread, shape)
    kernel = np.ones(3) / 3
    wander = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, wander)
    return np.clip(level + wander, 0.15, 1.1).astype(np.float32)


VAE_ID = "blowing-up-groundhogs/emuru_vae"
"""Emuru's own VAE: the same decoder that draws the model's broken strokes."""


def load_vae(device: str = "cpu"):
    """Emuru's VAE, frozen, and the latent of blank paper to draw lines toward."""
    import torch
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_ID).eval().to(device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        blank = torch.ones(1, 3, 64, 64, device=device)
        vae.register_buffer("paper", vae.encode(blank).latent_dist.mean.mean(-1, keepdim=True))
    return vae


def break_like_emuru(vae, lines, rng: np.random.Generator):
    """A batch of grey lines in [0, 1], broken through the VAE the way Emuru breaks them.

    Each line's latents are drawn toward paper by its own :func:`weakening`, then
    decoded. Widths must be a multiple of 8, the VAE's slice.
    """
    import torch

    with torch.no_grad():
        latents = vae.encode(lines.repeat(1, 3, 1, 1) * 2.0 - 1.0).latent_dist.mean
        factor = np.concatenate([weakening((1, *latents.shape[1:]), rng) for _ in latents])
        factor = torch.from_numpy(factor).to(latents.device)
        decoded = vae.decode(vae.paper + factor * (latents - vae.paper)).sample
    return ((decoded[:, :1].clamp(-1.0, 1.0) + 1.0) / 2.0).contiguous()


def load(path, device: str = "cpu"):
    """The trained repair network, ready to :func:`repair` lines."""
    import torch

    network = build_network()
    network.load_state_dict(torch.load(path, map_location=device))
    return network.to(device).eval()


def build_network(width: int = 32):
    """The repair network. Input and output: a batch of grey lines in [0, 1]."""
    import torch
    from torch import nn

    def block(inputs: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.GroupNorm(8, outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(8, outputs),
            nn.SiLU(),
        )

    class RepairNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = block(1, width)
            self.down2 = block(width, 2 * width)
            self.middle = block(2 * width, 4 * width)
            self.up2 = block(6 * width, 2 * width)
            self.up1 = block(3 * width, width)
            self.out = nn.Conv2d(width, 1, 1)
            self.pool = nn.AvgPool2d(2)
            self.grow = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        def forward(self, line):
            first = self.down1(line)
            second = self.down2(self.pool(first))
            middle = self.middle(self.pool(second))
            up = self.up2(torch.cat([self.grow(middle), second], 1))
            up = self.up1(torch.cat([self.grow(up), first], 1))
            predicted = torch.sigmoid(self.out(up))
            return torch.minimum(line, predicted)  # ink may only be added

    return RepairNet()


def repair(network, image: np.ndarray, device: str = "cpu") -> np.ndarray:
    """One grey line, ink dark on white, mended. Any width; the shape is kept."""
    import torch

    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError(f"expected a grey line, got shape {gray.shape}")
    height, width = gray.shape
    padded = np.pad(gray, ((0, -height % 4), (0, -width % 4)), constant_values=255)
    line = torch.from_numpy(padded.astype(np.float32) / 255.0)[None, None].to(device)
    network.eval()
    with torch.no_grad():
        mended = network(line)[0, 0, :height, :width].cpu().numpy()
    return np.clip(np.round(mended * 255.0), 0, 255).astype(np.uint8)
