"""Mend the strokes Emuru breaks: a small network that may only add ink.

People tell a generated line at a glance, without looking at the writer: its
strokes are not whole. Measured teacher-forced, the breaks are already in Emuru's
own predictions -- 8.09 pieces of ink per 100 columns against 3.34 on the real
line, 3.81 after the VAE alone -- the regression's average of what it is unsure
of. Pushing the predicted latents away from paper closes them but thickens every
stroke by 65%, and closing the image thickens too (PROGRESS.md, 2026-09-22).
Mending a break without thickening needs to know what a stroke looks like.

**How it learns.** Real lines by the training-split writers are broken the way
Emuru breaks them, through Emuru's own VAE: narrow cuts where the latents fall
toward the latent of blank paper (:func:`weakening`), then decoded. **Cuts, not
fading**, and the first attempt turned on that: drawing the whole line toward
paper matched Emuru's 5.18 pieces per 100 columns only at 66% of a real line's
ink, where Emuru's lines keep 85%, so the network learned to put back a third of
the ink and thickened every stroke by 43% (PROGRESS.md, 2026-09-23). Cutting
pieces out instead leaves 4.5 pieces per 100 columns at 82% of the ink. The
network sees the cut line and learns the real one.

**It may only add ink.** Its output is the darker of its prediction and its
input, pixel by pixel, so it can join and darken a stroke but never erase or
move one: letterforms stay the model's. A U-Net: an encoder that halves the
image twice to see a stroke's surroundings, a decoder back to full size, and
skip connections that carry the fine detail across.
"""

from __future__ import annotations

import cv2
import numpy as np

LEVEL = (0.97, 1.0)
"""How much of the ink survives away from the cuts. Near 1: Emuru's lines are not
faded, they have pieces missing, and lines at 1.0 teach the network to leave a
whole stroke alone."""

CUTS = (3.0, 15.0)
"""Cuts per 100 columns, drawn per line: enough to span Emuru's own rate."""

DEPTH = (0.05, 0.45)
"""How little of the ink is left inside a cut."""

CUT = (1, 4)
"""A cut's width, in latent slices -- 8 to 32 pixels. Wide enough at the top to
teach the network to bridge a break that splits a word in two: "Lines" came back
as "L ines" on a line nothing else caught."""

BARS = (0, 2)
"""Thin horizontal strokes rubbed out of a line before it is broken further.
The eye catches a letter that lost its crossbar at once -- an A drawn as a bow,
"fixed" read as "fined" -- and a cut in the latents cannot take a whole bar out,
only a piece of one, so it is done to the image instead."""

BAR_LENGTH = 9
"""A horizontal run at least this long, and at most :data:`BAR_HEIGHT` tall, is a
bar: the crossbar of an A or a t, not a stroke of the writing's own body."""

BAR_HEIGHT = 4


def rub_out_bars(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A grey line with a few of its thin horizontal strokes rubbed out."""
    wanted = rng.integers(*BARS) if BARS[1] > BARS[0] else 0
    if wanted <= 0:
        return image
    ink = (image < 160).astype(np.uint8)
    flat = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, BAR_LENGTH), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(flat, 8)
    bars = [i for i in range(1, count) if stats[i, cv2.CC_STAT_HEIGHT] <= BAR_HEIGHT]
    if not bars:
        return image
    out = image.copy()
    for index in rng.choice(bars, size=min(wanted, len(bars)), replace=False):
        out[labels == index] = 255
    return out


def weakening(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """A factor per latent position: near 1 over the line, with narrow cuts in it.

    Each cut covers part of the line's height, so a stroke can be cut through while
    the one above it survives -- which is what a break inside a letter looks like.
    """
    rows, columns = shape[-2], shape[-1]
    field = np.full(shape, rng.uniform(*LEVEL), dtype=np.float32)
    for _ in range(rng.poisson(rng.uniform(*CUTS) * columns / 100)):
        left = rng.integers(0, columns)
        top = rng.integers(0, rows)
        field[..., top : top + rng.integers(1, rows), left : left + rng.integers(*CUT) + 1] = (
            rng.uniform(*DEPTH)
        )
    return field


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
