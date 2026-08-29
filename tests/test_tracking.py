"""Tests for experiment tracking.

The property that matters most is the one that is easiest to get wrong: logging
must never kill a training run. A multi-day run lost at hour 30 because a metrics
server was briefly unreachable is a self-inflicted wound, so the failure paths get
as much attention here as the happy path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nib.config import load_config
from nib.engine import tracking
from nib.engine.tracking import (
    NullTracker,
    Tracker,
    WandbTracker,
    make_grid,
    make_tracker,
)


def word_image(width: int = 40, height: int = 20, value: int = 128) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


# --------------------------------------------------------------------------
# scalars
# --------------------------------------------------------------------------


def test_scalars_are_appended_one_json_object_per_line(tmp_path):
    """JSONL rather than a single JSON array: a run killed mid-write leaves a file
    that is still readable up to the last complete line."""
    tracker = Tracker(directory=tmp_path)
    tracker.log({"loss": 1.5}, step=1)
    tracker.log({"loss": 1.2, "lr": 0.001}, step=2)

    rows = [json.loads(line) for line in tracker.scalars_path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["loss"] == 1.2
    assert rows[1]["lr"] == 0.001
    assert all("wall_time" in r for r in rows)


def test_numpy_and_torch_scalars_are_serialisable(tmp_path):
    tracker = Tracker(directory=tmp_path)
    tracker.log({"a": np.float32(0.25), "b": np.int64(7)}, step=1)
    row = json.loads(tracker.scalars_path.read_text().splitlines()[0])
    assert row["a"] == 0.25
    assert row["b"] == 7


def test_a_config_can_be_logged_as_plain_data(tmp_path):
    tracker = Tracker(directory=tmp_path)
    tracker.log({"cfg": load_config()}, step=0)
    row = json.loads(tracker.scalars_path.read_text().splitlines()[0])
    assert isinstance(row["cfg"], dict)
    assert row["cfg"]["data"]["image_height"] == 64


# --------------------------------------------------------------------------
# the visual log -- the actual diagnostic for a generative model
# --------------------------------------------------------------------------


def test_images_are_written_as_one_contact_sheet_per_step(tmp_path):
    tracker = Tracker(directory=tmp_path)
    path = tracker.log_images("samples", [word_image() for _ in range(6)], step=500)
    assert path is not None and path.is_file()
    assert path.name == "samples_00000500.png"


def test_logging_no_images_is_a_no_op_not_a_crash(tmp_path):
    assert Tracker(directory=tmp_path).log_images("samples", [], step=1) is None


def test_grid_pads_rather_than_stretches_mismatched_widths():
    """Handwriting crops differ wildly in width. Resizing them to a common box
    would distort the aspect ratio the samples are being inspected for."""
    images = [word_image(width=w) for w in (10, 80, 30)]
    sheet = make_grid(images, columns=3, pad=4)
    assert sheet.shape[0] == 20 + 2 * 4
    assert sheet.shape[1] == 3 * (80 + 4) + 4


def test_grid_wraps_onto_multiple_rows():
    sheet = make_grid([word_image() for _ in range(7)], columns=3, pad=2)
    assert sheet.shape[0] == 3 * (20 + 2) + 2  # ceil(7/3) == 3 rows


def test_grid_rejects_a_caption_count_mismatch():
    with pytest.raises(ValueError, match="captions"):
        make_grid([word_image(), word_image()], captions=["only one"])


def test_grid_rejects_no_images():
    with pytest.raises(ValueError, match="no images"):
        make_grid([])


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 12), dtype=np.uint8),
        np.zeros((8, 12, 1), dtype=np.uint8),
        np.zeros((8, 12, 3), dtype=np.uint8),
        np.zeros((3, 8, 12), dtype=np.uint8),  # CHW, as a model emits
        np.zeros((8, 12), dtype=np.float32),
        np.full((8, 12), 0.5, dtype=np.float32),
    ],
)
def test_grid_accepts_every_shape_a_model_might_emit(image):
    assert make_grid([image]).ndim == 2


def test_generator_output_in_minus_one_to_one_is_mapped_correctly():
    """A generator with a tanh head emits [-1, 1]. Treating that as [0, 1] would
    clip every dark pixel to black and make samples look wrong."""
    sheet = make_grid([np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)], pad=0)
    assert sheet[0, 0] == 0
    assert 120 <= sheet[0, 1] <= 135
    assert sheet[0, 2] == 255


def test_grid_rejects_an_unrenderable_shape():
    with pytest.raises(ValueError, match="cannot render"):
        make_grid([np.zeros((2, 3, 4, 5))])


# --------------------------------------------------------------------------
# choosing a backend
# --------------------------------------------------------------------------


def test_disabled_mode_records_nothing_and_never_raises(tmp_path):
    cfg = load_config(overrides=["tracking.mode=disabled"])
    tracker = make_tracker(cfg, tmp_path)
    assert isinstance(tracker, NullTracker)
    tracker.log({"loss": 1.0}, step=1)
    assert tracker.log_images("s", [word_image()], step=1) is None
    tracker.finish()
    assert not list(tmp_path.iterdir())


def test_offline_mode_writes_locally_and_touches_no_network(tmp_path):
    cfg = load_config(overrides=["tracking.mode=offline"])
    tracker = make_tracker(cfg, tmp_path)
    assert type(tracker) is Tracker
    tracker.log({"loss": 1.0}, step=1)
    tracker.log_images("s", [word_image()], step=1)
    assert tracker.scalars_path.is_file()
    assert list(tracker.images_dir.glob("*.png"))


def test_unknown_mode_raises_rather_than_silently_disabling(tmp_path):
    with pytest.raises(ValueError, match="unknown tracking mode"):
        make_tracker(_with_mode("nonsense"), tmp_path)


def test_the_configured_entity_is_the_username_not_a_secret():
    """A guard against an API key ever being pasted into the config. Keys are
    long hex strings; usernames are not."""
    cfg = load_config("configs/base.yaml")
    entity = cfg.tracking.entity
    if entity is None:
        pytest.skip("no entity configured yet")
    assert len(entity) < 40, "that looks like an API key, not a username"
    assert not all(c in "0123456789abcdef" for c in entity.lower())


# --------------------------------------------------------------------------
# failure must never kill a run
# --------------------------------------------------------------------------


def test_online_mode_degrades_to_local_when_wandb_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "_HAS_WANDB", False)
    tracker = WandbTracker(directory=tmp_path, project="nib", entity="someone")
    assert not tracker.remote_alive

    tracker.log({"loss": 1.0}, step=1)
    tracker.log_images("s", [word_image()], step=1)
    tracker.finish()

    assert tracker.scalars_path.is_file(), "local records must continue regardless"
    assert list(tracker.images_dir.glob("*.png"))


def test_a_failure_to_start_a_run_does_not_raise(tmp_path, monkeypatch):
    class Boom:
        @staticmethod
        def init(**_kwargs):
            raise RuntimeError("no network")

    monkeypatch.setattr(tracking, "_HAS_WANDB", True)
    monkeypatch.setattr(tracking, "wandb", Boom)

    tracker = WandbTracker(directory=tmp_path)  # must not raise
    assert not tracker.remote_alive
    tracker.log({"loss": 1.0}, step=1)
    assert tracker.scalars_path.is_file()


def test_the_remote_backend_is_abandoned_after_repeated_failures(tmp_path, monkeypatch):
    """Retrying a dead connection every step for the rest of a multi-day run costs
    more than the logging is worth."""
    calls = {"n": 0}

    class FlakyRun:
        def log(self, *_args, **_kwargs):
            calls["n"] += 1
            raise RuntimeError("server gone")

        def finish(self):
            pass

    class FakeWandb:
        @staticmethod
        def init(**_kwargs):
            return FlakyRun()

        @staticmethod
        def Image(path):
            return path

    monkeypatch.setattr(tracking, "_HAS_WANDB", True)
    monkeypatch.setattr(tracking, "wandb", FakeWandb)

    tracker = WandbTracker(directory=tmp_path)
    assert tracker.remote_alive
    for step in range(20):
        tracker.log({"loss": 1.0}, step=step)  # must never raise

    assert not tracker.remote_alive
    assert calls["n"] == tracking.MAX_CONSECUTIVE_FAILURES, "kept retrying a dead backend"
    rows = tracker.scalars_path.read_text().splitlines()
    assert len(rows) == 20, "local logging stopped when the remote one did"


def test_a_recovered_backend_resets_the_failure_count(tmp_path, monkeypatch):
    """One dropped packet must not count towards the give-up threshold forever."""
    state = {"fail": True}

    class Run:
        def log(self, *_args, **_kwargs):
            if state["fail"]:
                raise RuntimeError("transient")

        def finish(self):
            pass

    class FakeWandb:
        @staticmethod
        def init(**_kwargs):
            return Run()

        @staticmethod
        def Image(path):
            return path

    monkeypatch.setattr(tracking, "_HAS_WANDB", True)
    monkeypatch.setattr(tracking, "wandb", FakeWandb)

    tracker = WandbTracker(directory=tmp_path)
    tracker.log({"a": 1}, step=1)
    tracker.log({"a": 1}, step=2)
    state["fail"] = False
    tracker.log({"a": 1}, step=3)
    state["fail"] = True
    tracker.log({"a": 1}, step=4)
    tracker.log({"a": 1}, step=5)

    assert tracker.remote_alive, "gave up despite the connection recovering in between"


def _with_mode(mode: str):
    from omegaconf import OmegaConf

    return OmegaConf.create({"tracking": {"mode": mode, "project": "nib", "entity": None}})
