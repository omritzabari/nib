"""Tests for the per-writer fine-tune.

The line preparation is pure and tested exactly. The LoRA plumbing is tested on a
model built like Emuru -- a T5 at ``.T5``, a frozen encoder, projections either
side, a start token, a tokenizer -- but tiny and randomly initialised, so it runs
on a CPU in seconds and needs no download.
"""

from __future__ import annotations

import numpy as np
import pytest

from nib.models import finetune
from nib.models.finetune import FinetuneConfig, pad_batch, prepare_line

# ---------------------------------------------------------------------------
# Line preparation
# ---------------------------------------------------------------------------


def _line(width=100, height=64):
    image = np.full((height, width), 255, dtype=np.uint8)
    image[20:44, 10 : width - 10] = 0
    return image


def test_a_prepared_line_is_three_channels_in_minus_one_to_one():
    array = prepare_line(_line())

    assert array.shape[0] == 3
    assert array.dtype == np.float32
    assert array.min() == pytest.approx(-1.0)
    assert array.max() == pytest.approx(1.0)


def test_it_ends_in_white_and_fills_whole_slices():
    array = prepare_line(_line(width=101), trailing=128)

    assert array.shape[-1] % finetune.PIXELS_PER_SLICE == 0
    assert array.shape[-1] >= 101 + 128
    assert (array[..., 101:] == 1.0).all(), "everything after the line must be white"


def test_a_line_of_another_height_is_scaled_keeping_its_proportions():
    array = prepare_line(_line(width=200, height=128), trailing=0)

    assert array.shape[1] == 64
    assert array.shape[2] == 104  # 100px at 64 high, rounded up to whole slices


def test_colour_or_float_input_is_refused():
    with pytest.raises(ValueError):
        prepare_line(np.zeros((64, 50, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_line(np.zeros((64, 50), dtype=np.float32))


def test_a_batch_pads_the_narrower_lines_with_white():
    short, long = prepare_line(_line(80)), prepare_line(_line(300))

    batch = pad_batch([short, long])

    assert batch.shape == (2, 3, 64, long.shape[-1])
    assert (batch[0, ..., short.shape[-1] :] == 1.0).all()
    np.testing.assert_array_equal(batch[1], long)


# ---------------------------------------------------------------------------
# LoRA on a model shaped like Emuru
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


class CharTokenizer:
    def __call__(self, texts, return_tensors="pt", padding=True):
        width = max(len(t) for t in texts) + 1
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            codes = [3 + ord(c) % 29 for c in text] + [1]
            ids[row, : len(codes)] = torch.tensor(codes)
            mask[row, : len(codes)] = 1
        return type("Tokens", (), {"input_ids": ids, "attention_mask": mask})()


class TinyEmuru(torch.nn.Module):
    """Emuru's forward, in miniature: slices of the image predicted by T5."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        config = transformers.T5Config(
            vocab_size=32,
            d_model=16,
            d_kv=4,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dropout_rate=0.0,
            decoder_start_token_id=0,
            pad_token_id=0,
        )
        self.T5 = transformers.T5ForConditionalGeneration(config)
        self.T5.lm_head = torch.nn.Identity()
        self.vae = torch.nn.Identity()
        self.vae_to_t5 = torch.nn.Linear(8, 16)
        self.t5_to_vae = torch.nn.Linear(16, 8, bias=False)
        self.sos = torch.nn.Embedding(1, 16)
        self.tokenizer = CharTokenizer()

    def forward(self, img, input_ids=None, attention_mask=None, noise=0.0):
        slices = img.mean(dim=(1, 2)).unfold(1, 8, 8)
        noisy = slices + torch.randn_like(slices) * noise if noise > 0 else slices
        embeds = torch.cat([self.sos.weight.expand(img.size(0), 1, -1), self.vae_to_t5(noisy)], 1)
        out = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=embeds)
        predicted = self.t5_to_vae(out.logits[:, :-1])
        return torch.nn.functional.mse_loss(predicted, slices), predicted, slices


def _outputs(model, lines, texts):
    model.eval()
    batch = torch.from_numpy(pad_batch([prepare_line(line) for line in lines]))
    tokens = model.tokenizer(texts)
    with torch.no_grad():
        _, predicted, _ = model(batch, tokens.input_ids, tokens.attention_mask)
    return predicted


LINES = [_line(120), _line(200)]
TEXTS = ["a line", "another line"]


def test_attaching_lora_trains_only_the_adapter_and_changes_nothing_yet():
    model = TinyEmuru()
    before = _outputs(model, LINES, TEXTS)

    trainable = finetune.attach_lora(model, FinetuneConfig(dropout=0.0))

    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable > 0
    assert names and all(finetune.LORA_MARKER in name for name in names)
    assert all(".T5." in f".{name}" or name.startswith("T5.") for name in names)
    torch.testing.assert_close(_outputs(model, LINES, TEXTS), before)


def test_training_lowers_the_loss_and_leaves_the_model_ready_to_generate():
    model = TinyEmuru()
    finetune.attach_lora(model, FinetuneConfig(dropout=0.0))
    config = FinetuneConfig(steps=80, learning_rate=1e-2, dropout=0.0, noise=0.0, batch_size=2)

    report = finetune.train_writer(model, LINES, TEXTS, config)

    assert report.steps == 80 and len(report.losses) == 80
    assert np.mean(report.losses[-8:]) < np.mean(report.losses[:8])
    assert not model.training
    assert "loss" in report.summary()


def test_reset_returns_the_model_to_its_released_behaviour_exactly():
    model = TinyEmuru()
    released = _outputs(model, LINES, TEXTS)
    finetune.attach_lora(model, FinetuneConfig(dropout=0.0))
    finetune.train_writer(
        model, LINES, TEXTS, FinetuneConfig(steps=20, learning_rate=1e-2, dropout=0.0)
    )
    assert not torch.allclose(_outputs(model, LINES, TEXTS), released)

    reset = finetune.reset_lora(model)

    assert reset > 0
    torch.testing.assert_close(_outputs(model, LINES, TEXTS), released)


def test_an_old_torchao_does_not_stop_the_adapter_being_attached(monkeypatch):
    """Colab ships torchao 0.10.0; peft raises on anything under 0.16.0 while
    merely checking whether a layer is torchao's, which killed the first 7d run."""
    import peft.import_utils
    import peft.tuners.lora.torchao

    def stale():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported"
        )

    monkeypatch.setattr(peft.import_utils, "is_torchao_available", stale)
    monkeypatch.setattr(peft.tuners.lora.torchao, "is_torchao_available", stale)
    model = TinyEmuru()

    trainable = finetune.attach_lora(model, FinetuneConfig(dropout=0.0))

    assert trainable > 0
    assert peft.tuners.lora.torchao.is_torchao_available() is False


def test_the_adapter_state_is_only_the_adapter():
    model = TinyEmuru()
    finetune.attach_lora(model, FinetuneConfig())

    state = finetune.lora_state(model)

    assert state and all(finetune.LORA_MARKER in name for name in state)
    total = sum(p.numel() for p in model.parameters())
    assert sum(t.numel() for t in state.values()) < total


def test_training_without_an_adapter_is_refused():
    model = TinyEmuru()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(RuntimeError, match="attach_lora"):
        finetune.train_writer(model, LINES, TEXTS, FinetuneConfig(steps=1))


def test_lines_and_texts_must_pair_up():
    model = TinyEmuru()
    finetune.attach_lora(model, FinetuneConfig())

    with pytest.raises(ValueError):
        finetune.train_writer(model, LINES, TEXTS[:1], FinetuneConfig(steps=1))
