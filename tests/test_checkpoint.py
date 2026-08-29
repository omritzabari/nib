"""Tests for checkpointing.

The headline test is `test_resume_is_bit_identical_to_an_uninterrupted_run`. Every
other test here exists to catch a specific way that guarantee gets quietly broken.

Note how the fake training loop below draws its batches from the global RNG. That
is deliberate: it means a resume that restores weights but forgets the random
state produces *different* results, and the test catches it. A loop with fixed
data would pass while the real thing silently diverged.
"""

from __future__ import annotations

import pytest

from nib.engine import checkpoint as ckpt

torch = pytest.importorskip("torch", reason="torch is an optional extra")
import torch.nn as nn  # noqa: E402


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def train_steps(model, optimizer, n_steps: int) -> None:
    """A stand-in training loop whose batches come from the global RNG."""
    for _ in range(n_steps):
        batch = torch.randn(4, 8)  # RNG-dependent on purpose
        target = torch.randn(4, 4)
        optimizer.zero_grad()
        loss = ((model(batch) - target) ** 2).mean()
        loss.backward()
        optimizer.step()


def weights_of(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def assert_identical(a, b, message: str) -> None:
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert torch.equal(x, y), f"{message} (tensor {i} differs, max delta {(x - y).abs().max()})"


# --------------------------------------------------------------------------
# the guarantee
# --------------------------------------------------------------------------


def test_resume_is_bit_identical_to_an_uninterrupted_run(tmp_path):
    """Train 20 straight. Then train 10, die, resume, train 10 more. The weights
    must match exactly -- not approximately."""
    ckpt.seed_everything(1234)
    reference = make_model()
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    train_steps(reference, optimizer, 20)
    expected = weights_of(reference)

    # interrupted run
    ckpt.seed_everything(1234)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_steps(model, optimizer, 10)
    ckpt.save(
        tmp_path / "ck.pt",
        models={"net": model},
        optimizers={"opt": optimizer},
        state=ckpt.TrainingState(step=10, epoch=0),
    )

    del model, optimizer  # nothing survives the "crash"

    resumed = make_model(seed=999)  # deliberately different init
    resumed_opt = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    state = ckpt.load(tmp_path / "ck.pt", models={"net": resumed}, optimizers={"opt": resumed_opt})
    assert state.step == 10
    train_steps(resumed, resumed_opt, 10)

    assert_identical(weights_of(resumed), expected, "resumed run diverged")


def test_forgetting_the_rng_state_actually_changes_the_result(tmp_path):
    """Proves the previous test is not passing by accident. With restore_rng=False
    the resumed run must differ -- if it does not, the loop is not RNG-sensitive
    and the guarantee is untested."""
    ckpt.seed_everything(7)
    reference = make_model()
    opt = torch.optim.Adam(reference.parameters(), lr=1e-3)
    train_steps(reference, opt, 20)
    expected = weights_of(reference)

    ckpt.seed_everything(7)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_steps(model, opt, 10)
    ckpt.save(
        tmp_path / "ck.pt",
        models={"net": model},
        optimizers={"opt": opt},
        state=ckpt.TrainingState(step=10),
    )

    resumed = make_model(seed=1)
    resumed_opt = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    ckpt.load(
        tmp_path / "ck.pt",
        models={"net": resumed},
        optimizers={"opt": resumed_opt},
        restore_rng=False,
    )
    torch.manual_seed(555)  # some other state, as a fresh process would have
    train_steps(resumed, resumed_opt, 10)

    same = all(torch.equal(x, y) for x, y in zip(weights_of(resumed), expected, strict=True))
    assert not same, "the fake training loop does not depend on the RNG, so the test is vacuous"


def test_optimizer_state_is_restored_not_reinitialised(tmp_path):
    """Adam's moment estimates cannot be reconstructed from the weights. Losing
    them makes the first steps after a resume behave like a fresh optimiser."""
    ckpt.seed_everything(3)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_steps(model, opt, 15)
    ckpt.save(
        tmp_path / "ck.pt",
        models={"net": model},
        optimizers={"opt": opt},
        state=ckpt.TrainingState(step=15),
    )

    fresh = make_model(seed=5)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-2)
    assert not fresh_opt.state_dict()["state"], "precondition: a new optimiser has no state"

    ckpt.load(tmp_path / "ck.pt", models={"net": fresh}, optimizers={"opt": fresh_opt})
    restored = fresh_opt.state_dict()["state"]
    assert restored, "optimiser state was not restored"
    assert any("exp_avg" in v for v in restored.values()), "Adam moments missing"


def test_every_network_is_saved_including_the_discriminator(tmp_path):
    """A GAN's discriminator is not disposable mid-run. Resuming without it
    restarts the adversarial game."""
    generator, discriminator = make_model(1), make_model(2)
    ckpt.save(
        tmp_path / "ck.pt",
        models={"generator": generator, "discriminator": discriminator},
        state=ckpt.TrainingState(step=5),
    )
    g2, d2 = make_model(9), make_model(9)
    ckpt.load(tmp_path / "ck.pt", models={"generator": g2, "discriminator": d2})
    assert_identical(weights_of(g2), weights_of(generator), "generator")
    assert_identical(weights_of(d2), weights_of(discriminator), "discriminator")


# --------------------------------------------------------------------------
# atomicity -- the Colab failure mode
# --------------------------------------------------------------------------


def test_no_temporary_file_is_left_behind(tmp_path):
    ckpt.save(tmp_path / "ck.pt", models={"net": make_model()}, state=ckpt.TrainingState())
    assert not list(tmp_path.glob("*.tmp"))


def test_a_crash_mid_write_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """The specific way this goes wrong on Colab: the kill arrives without warning.
    A partially written file that fails to load would lose the whole run."""
    path = tmp_path / "ck.pt"
    good = make_model(1)
    ckpt.save(path, models={"net": good}, state=ckpt.TrainingState(step=100))

    real_save = torch.save

    def exploding_save(*args, **kwargs):
        real_save(*args, **kwargs)
        raise OSError("disk died mid-write")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(OSError):
        ckpt.save(path, models={"net": make_model(2)}, state=ckpt.TrainingState(step=200))

    # the old checkpoint must still load, and still be the old one
    recovered = make_model(9)
    state = ckpt.load(path, models={"net": recovered})
    assert state.step == 100
    assert_identical(weights_of(recovered), weights_of(good), "previous checkpoint was damaged")


# --------------------------------------------------------------------------
# loud failures
# --------------------------------------------------------------------------


def test_a_missing_model_entry_raises_rather_than_loading_partially(tmp_path):
    ckpt.save(tmp_path / "ck.pt", models={"generator": make_model()}, state=ckpt.TrainingState())
    with pytest.raises(ckpt.CheckpointError, match="no model entry"):
        ckpt.load(
            tmp_path / "ck.pt",
            models={"generator": make_model(), "discriminator": make_model()},
        )


def test_an_unexpected_entry_raises(tmp_path):
    """Silently ignoring a saved discriminator would look like training
    instability rather than a bug."""
    ckpt.save(
        tmp_path / "ck.pt",
        models={"generator": make_model(), "discriminator": make_model()},
        state=ckpt.TrainingState(),
    )
    with pytest.raises(ckpt.CheckpointError, match="not passed in"):
        ckpt.load(tmp_path / "ck.pt", models={"generator": make_model()})


def test_strict_can_be_relaxed_deliberately(tmp_path):
    ckpt.save(
        tmp_path / "ck.pt",
        models={"generator": make_model(), "discriminator": make_model()},
        state=ckpt.TrainingState(step=4),
    )
    state = ckpt.load(tmp_path / "ck.pt", models={"generator": make_model()}, strict=False)
    assert state.step == 4


def test_missing_file_raises(tmp_path):
    with pytest.raises(ckpt.CheckpointError, match="no checkpoint at"):
        ckpt.load(tmp_path / "nope.pt")


def test_format_version_mismatch_raises(tmp_path):
    path = tmp_path / "ck.pt"
    ckpt.save(path, models={"net": make_model()}, state=ckpt.TrainingState())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ckpt.CheckpointError, match="format 99"):
        ckpt.load(path, models={"net": make_model()})


# --------------------------------------------------------------------------
# state and provenance
# --------------------------------------------------------------------------


def test_training_state_round_trips(tmp_path):
    state = ckpt.TrainingState(step=1234, epoch=7, best_metric=0.42, extra={"note": "run a"})
    ckpt.save(tmp_path / "ck.pt", models={"net": make_model()}, state=state)
    loaded = ckpt.load(tmp_path / "ck.pt", models={"net": make_model()})
    assert (loaded.step, loaded.epoch, loaded.best_metric) == (1234, 7, 0.42)
    assert loaded.extra == {"note": "run a"}


def test_config_is_stored_as_plain_data(tmp_path):
    """A checkpoint should say what produced it, without needing the exact library
    version that wrote it in order to be readable."""
    from nib.config import load_config

    cfg = load_config()
    path = tmp_path / "ck.pt"
    ckpt.save(path, models={"net": make_model()}, state=ckpt.TrainingState(), config=cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload["config"], dict)
    assert payload["config"]["data"]["image_height"] == cfg.data.image_height


# --------------------------------------------------------------------------
# the manager
# --------------------------------------------------------------------------


def test_manager_saves_on_the_interval_only():
    manager = ckpt.CheckpointManager("unused", every_n_steps=2000)
    assert not manager.should_save(0), "step 0 is not a checkpoint"
    assert not manager.should_save(1999)
    assert manager.should_save(2000)
    assert manager.should_save(4000)


def test_manager_latest_always_points_at_the_newest(tmp_path):
    manager = ckpt.CheckpointManager(tmp_path, every_n_steps=1)
    for step in (1, 2, 3):
        model = make_model(step)
        manager.save(ckpt.TrainingState(step=step), models={"net": model})
    assert manager.latest_path.is_file()
    target = make_model(99)
    assert ckpt.load(manager.latest_path, models={"net": target}).step == 3


def test_manager_prunes_old_checkpoints_but_keeps_latest(tmp_path):
    """Checkpoints go to Drive, which is not unlimited."""
    manager = ckpt.CheckpointManager(tmp_path, every_n_steps=1, keep_last=2)
    for step in range(1, 6):
        manager.save(ckpt.TrainingState(step=step), models={"net": make_model()})
    numbered = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert numbered == ["step_00000004.pt", "step_00000005.pt"]
    assert manager.latest_path.is_file()


def test_manager_resume_returns_none_for_a_fresh_run(tmp_path):
    assert ckpt.CheckpointManager(tmp_path).resume() is None


def test_manager_resume_round_trips(tmp_path):
    manager = ckpt.CheckpointManager(tmp_path, every_n_steps=1)
    model = make_model(3)
    manager.save(ckpt.TrainingState(step=42, epoch=2), models={"net": model})

    target = make_model(8)
    state = ckpt.CheckpointManager(tmp_path).resume(models={"net": target})
    assert state is not None
    assert (state.step, state.epoch) == (42, 2)
    assert_identical(weights_of(target), weights_of(model), "resumed weights")


def test_manager_rejects_nonsense_settings(tmp_path):
    with pytest.raises(ckpt.CheckpointError, match="every_n_steps"):
        ckpt.CheckpointManager(tmp_path, every_n_steps=0)
    with pytest.raises(ckpt.CheckpointError, match="keep_last"):
        ckpt.CheckpointManager(tmp_path, keep_last=0)


# --------------------------------------------------------------------------
# rng helpers
# --------------------------------------------------------------------------


def test_seed_everything_makes_all_three_generators_reproducible():
    import random as py_random

    import numpy as np

    ckpt.seed_everything(11)
    first = (py_random.random(), np.random.rand(), torch.rand(1).item())
    ckpt.seed_everything(11)
    assert first == (py_random.random(), np.random.rand(), torch.rand(1).item())


def test_rng_state_capture_and_restore_round_trips():
    ckpt.seed_everything(2)
    state = ckpt.capture_rng_state()
    expected = torch.rand(3)
    ckpt.restore_rng_state(state)
    assert torch.equal(torch.rand(3), expected)
