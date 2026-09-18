"""Tests for saving and loading a trained adapter.

An adaptation is worth training once and using for ever, which means it has to
survive leaving the process. These tests use a tiny model with a LoRA adapter on
it, so nothing is downloaded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
nn = torch.nn

from nib.models import adapters  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_v = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.to_v(self.to_q(x))


def adapted(seed: int = 0):
    """A tiny model with a fresh adapter. The seed fixes the frozen weights, so two
    calls differ only by what the adapter has learned -- which is the point of
    saving one: it is meaningless without the model it was trained against."""
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(seed)
    model = Tiny()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]), model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(adapters.MARKER in name)
    return model


def train_a_little(model):
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(4, 8)
    for _ in range(5):
        loss = (model(x) - torch.ones(4, 8)).pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


def test_the_saved_file_holds_the_adapter_and_nothing_else(tmp_path):
    model = train_a_little(adapted())

    path = adapters.save(model, tmp_path / "adapter.pt")

    stored = torch.load(path, map_location="cpu", weights_only=True)
    assert stored and all(adapters.MARKER in name for name in stored)
    assert not any("base_layer" in name for name in stored)


def test_a_loaded_adapter_makes_the_model_draw_as_it_did_when_saved(tmp_path):
    trained = train_a_little(adapted())
    torch.manual_seed(99)
    x = torch.randn(4, 8)
    expected = trained(x)
    path = adapters.save(trained, tmp_path / "adapter.pt")

    fresh = adapted()
    assert not torch.allclose(fresh(x), expected), "a fresh adapter should differ"
    adapters.load(fresh, path)

    torch.testing.assert_close(fresh(x), expected)


def test_an_adapter_that_does_not_fit_the_model_is_refused(tmp_path):
    path = adapters.save(train_a_little(adapted()), tmp_path / "adapter.pt")
    stored = torch.load(path, map_location="cpu", weights_only=True)
    torch.save({name.replace("to_q", "to_x"): value for name, value in stored.items()}, path)

    with pytest.raises(ValueError, match="does not fit"):
        adapters.load(adapted(), path)


def test_a_model_without_an_adapter_cannot_be_loaded_into(tmp_path):
    path = adapters.save(train_a_little(adapted()), tmp_path / "adapter.pt")

    with pytest.raises(ValueError, match="no adapter"):
        adapters.load(Tiny(), path)
