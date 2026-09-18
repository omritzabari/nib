"""Tests for the per-writer DiffBrush fine-tune.

Batch preparation is pure and tested exactly. The LoRA plumbing and the training
loop run on a tiny UNet built like DiffBrush's -- attention projections named
``to_q``, ``to_k``, ``to_v`` and ``to_out.0``, a forward taking noise, timesteps,
a style line and glyphs -- and a VAE stand-in, so nothing is downloaded and the
whole file runs on a CPU in seconds.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from nib.models.diffbrush import CANVAS_WIDTH, CHARSET
from nib.models.diffbrush_finetune import (
    DiffBrushFinetuneConfig,
    pad_contents,
    pad_styles,
    pick_style,
    prepare_target,
)


def _line(width=800, height=64):
    image = np.full((height, width), 255, dtype=np.uint8)
    image[20:44, 10 : width - 10] = 0
    return image


# ---------------------------------------------------------------------------
# preparing a batch
# ---------------------------------------------------------------------------


def test_a_target_line_sits_at_the_left_of_a_white_canvas_in_minus_one_to_one():
    canvas, squeezed = prepare_target(_line(800))

    assert canvas.shape == (3, 64, CANVAS_WIDTH)
    assert canvas.dtype == np.float32
    assert canvas.min() == pytest.approx(-1.0)
    assert (canvas[..., 800:] == 1.0).all()
    assert not squeezed


def test_a_line_wider_than_the_canvas_is_squeezed_to_fit_and_says_so():
    canvas, squeezed = prepare_target(_line(1200))

    assert canvas.shape == (3, 64, CANVAS_WIDTH)
    assert squeezed
    assert canvas[..., CANVAS_WIDTH - 20 : CANVAS_WIDTH - 10].min() == pytest.approx(-1.0)


def test_a_taller_line_is_scaled_to_the_native_height_first():
    canvas, squeezed = prepare_target(_line(1000, height=128))

    assert canvas.shape == (3, 64, CANVAS_WIDTH)
    assert (canvas[..., 500:] == 1.0).all()
    assert not squeezed


def test_the_style_line_is_always_another_of_the_writers_lines():
    rng = random.Random(0)

    picks = {pick_style(rng, target=2, count=4) for _ in range(200)}

    assert picks == {0, 1, 3}


def test_a_writer_with_one_line_cannot_be_trained():
    with pytest.raises(ValueError, match="at least two"):
        pick_style(random.Random(0), target=0, count=1)


def test_glyph_sequences_of_different_lengths_pad_with_the_blank_glyph():
    batch = pad_contents([np.zeros((3, 16, 16), np.float32), np.zeros((5, 16, 16), np.float32)])

    assert batch.shape == (2, 5, 16, 16)
    assert (batch[0, 3:] == 1.0).all()
    assert (batch[1] == 0.0).all()


def test_style_lines_of_different_widths_pad_with_white():
    batch = pad_styles([np.zeros((64, 300), np.float32), np.zeros((64, 500), np.float32)])

    assert batch.shape == (2, 1, 64, 500)
    assert (batch[0, 0, :, 300:] == 1.0).all()


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError):
        DiffBrushFinetuneConfig(steps=0)


# ---------------------------------------------------------------------------
# LoRA and training on a UNet shaped like DiffBrush's
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
nn = torch.nn


class TinyAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))

    def forward(self, x, context):
        weights = torch.softmax(self.to_q(x) @ self.to_k(context).transpose(1, 2), dim=-1)
        return self.to_out(weights @ self.to_v(context))


class _Checkpoint(torch.autograd.Function):
    """DiffBrush's own gradient checkpointing, reduced: it recomputes the block in
    backward and asks for gradients with respect to *every* parameter of the block,
    frozen or not."""

    @staticmethod
    def forward(ctx, run, length, *args):
        ctx.run, ctx.inputs, ctx.params = run, list(args[:length]), list(args[length:])
        with torch.no_grad():
            return run(*ctx.inputs)

    @staticmethod
    def backward(ctx, *grads):
        inputs = [x.detach().requires_grad_(True) for x in ctx.inputs]
        with torch.enable_grad():
            outputs = ctx.run(*[x.view_as(x) for x in inputs])
        found = torch.autograd.grad(outputs, inputs + ctx.params, grads, allow_unused=True)
        return (None, None, *found)


class TinyTransformerBlock(nn.Module):
    """Like DiffBrush's ``BasicTransformerBlock``: ``attn1`` over the image alone,
    ``attn2`` bringing in the style and the glyphs, and checkpointing by default."""

    def __init__(self, checkpoint=True):
        super().__init__()
        self.attn1 = TinyAttention()
        self.attn2 = TinyAttention()
        self.checkpoint = checkpoint

    def forward(self, x, context):
        if self.checkpoint:
            return _Checkpoint.apply(self._forward, 2, x, context, *self.parameters())
        return self._forward(x, context)

    def _forward(self, x, context):
        x = x + self.attn1(x, x)
        return x + self.attn2(x, context)


class TinyUNet(nn.Module):
    """Noise, timesteps, a style line and glyphs in; predicted noise out."""

    in_channels = 4

    def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(4, 8, 1)
        self.style = nn.Linear(64, 8)
        self.glyph = nn.Linear(256, 8)
        self.attention = TinyTransformerBlock()
        self.out = nn.Conv2d(8, 4, 1)

    def forward(self, x, timesteps, style, content, tag="test"):
        h = self.inp(x)
        batch, channels, height, width = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        context = torch.cat(
            [self.style(style.mean(dim=-1).flatten(1))[:, None], self.glyph(content.flatten(2))],
            dim=1,
        )
        tokens = self.attention(tokens, context)
        h = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.out(h) * (1 + timesteps[:, None, None, None].float() / 1000)


class TinyVAE:
    """8x average pooling into four channels: a fixed encoder with no weights."""

    def encode(self, images):
        pooled = torch.nn.functional.avg_pool2d(images, 8)
        latent = torch.cat([pooled, pooled[:, :1]], dim=1)
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mean=latent, std=torch.zeros_like(latent))
        )


def _glyphs():
    return np.stack([np.full((16, 16), i / 100, np.float32) for i in range(len(CHARSET))])


def _writer(count=3):
    return [_line(600 + 100 * i) for i in range(count)], ["a line", "another", "the third"][:count]


def test_attaching_lora_trains_only_the_adapter_and_changes_nothing_yet():
    from nib.models.diffbrush_finetune import attach_lora

    torch.manual_seed(0)
    unet = TinyUNet()
    x = torch.randn(1, 4, 8, 128)
    style = torch.rand(1, 1, 64, 300)
    content = torch.rand(1, 5, 16, 16)
    t = torch.tensor([10])
    before = unet(x, t, style, content)

    trainable = attach_lora(unet, DiffBrushFinetuneConfig(rank=2, alpha=2, scope="all"))

    names = [name for name, p in unet.named_parameters() if p.requires_grad]
    assert trainable == sum(p.numel() for p in unet.parameters() if p.requires_grad)
    assert names and all("lora_" in name for name in names)
    assert {name.split(".lora_")[0].rsplit(".", 1)[-1] for name in names} >= {
        "to_q",
        "to_k",
        "to_v",
    }
    assert any("to_out.0.lora_" in name for name in names)
    torch.testing.assert_close(unet(x, t, style, content), before)


def test_the_default_scope_leaves_the_layers_that_carry_the_text_alone():
    """300 steps on every attention layer tripled CER (T37): the adapter sat on the
    cross-attention that ties the drawing to the glyphs as well."""
    from nib.models.diffbrush_finetune import attach_lora

    unet = TinyUNet()

    attach_lora(unet, DiffBrushFinetuneConfig(rank=2, alpha=2))

    adapted = {name for name, p in unet.named_parameters() if p.requires_grad}
    assert adapted, "nothing was adapted"
    assert all("attn1." in name for name in adapted)


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="scope"):
        DiffBrushFinetuneConfig(scope="everything")


def test_training_moves_only_the_adapter_and_leaves_the_model_ready_to_generate():
    from nib.models.diffbrush_finetune import attach_lora, train_writer

    torch.manual_seed(0)
    unet = TinyUNet()
    attach_lora(unet, DiffBrushFinetuneConfig(rank=2, alpha=2))
    frozen = {n: p.detach().clone() for n, p in unet.named_parameters() if not p.requires_grad}
    adapter = {n: p.detach().clone() for n, p in unet.named_parameters() if p.requires_grad}
    images, texts = _writer()

    report = train_writer(
        unet, TinyVAE(), images, texts, _glyphs(), DiffBrushFinetuneConfig(rank=2, alpha=2, steps=5)
    )

    assert report.steps == 5 and len(report.losses) == 5
    assert all(np.isfinite(report.losses))
    for name, value in frozen.items():
        torch.testing.assert_close(dict(unet.named_parameters())[name], value)
    assert any(
        not torch.equal(dict(unet.named_parameters())[name], value)
        for name, value in adapter.items()
    )
    assert not unet.training


def test_a_line_wider_than_the_canvas_is_left_out_of_training_and_counted():
    """Squeezing a line to the canvas narrows its letters, so by default those lines
    are not trained on at all."""
    from nib.models.diffbrush_finetune import attach_lora, train_writer

    unet = TinyUNet()
    attach_lora(unet, DiffBrushFinetuneConfig(rank=2, alpha=2))

    report = train_writer(
        unet,
        TinyVAE(),
        [_line(1200), _line(700), _line(650)],
        ["a wide line", "a short one", "another short one"],
        _glyphs(),
        DiffBrushFinetuneConfig(rank=2, alpha=2, steps=1),
    )

    assert report.skipped == 1
    assert report.squeezed == 0


def test_a_line_wider_than_the_canvas_is_squeezed_when_asked_to_keep_it():
    from nib.models.diffbrush_finetune import attach_lora, train_writer

    unet = TinyUNet()
    attach_lora(unet, DiffBrushFinetuneConfig(rank=2, alpha=2))

    report = train_writer(
        unet,
        TinyVAE(),
        [_line(1200), _line(700)],
        ["a wide line", "a short one"],
        _glyphs(),
        DiffBrushFinetuneConfig(rank=2, alpha=2, steps=1, skip_wide=False),
    )

    assert report.squeezed == 1 and report.skipped == 0


def test_a_writer_left_with_too_few_narrow_lines_keeps_the_wide_ones():
    """Better a squeezed line than a writer who cannot be trained at all."""
    from nib.models.diffbrush_finetune import attach_lora, train_writer

    unet = TinyUNet()
    attach_lora(unet, DiffBrushFinetuneConfig(rank=2, alpha=2))

    report = train_writer(
        unet,
        TinyVAE(),
        [_line(1200), _line(1300), _line(700)],
        ["a wide line", "another wide one", "a short one"],
        _glyphs(),
        DiffBrushFinetuneConfig(rank=2, alpha=2, steps=1),
    )

    assert report.skipped == 0 and report.squeezed == 2


def test_training_without_an_adapter_is_refused():
    from nib.models.diffbrush_finetune import train_writer

    images, texts = _writer()
    with pytest.raises(RuntimeError, match="attach_lora"):
        train_writer(TinyUNet().requires_grad_(False), TinyVAE(), images, texts, _glyphs())


def test_lines_and_texts_must_pair_up():
    from nib.models.diffbrush_finetune import attach_lora, train_writer

    unet = TinyUNet()
    attach_lora(unet)
    images, _ = _writer()
    with pytest.raises(ValueError):
        train_writer(unet, TinyVAE(), images, ["only one text"], _glyphs())
