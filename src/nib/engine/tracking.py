"""Experiment tracking, with visual sample logging from day one.

The project brief is specific about this: save image samples every few hundred
steps and look at them. For a generative model that is not a nicety. Loss curves
for a GAN are famously uninformative -- a run that has collapsed to producing one
image can have a perfectly healthy-looking loss. The pictures are the diagnosis.

The one rule this module is built around:

    **Logging must never kill a training run.**

A run is measured in hours or days. If the tracking server hiccups at hour 30, or
the Colab session loses its network, or an image fails to encode, the correct
behaviour is to note it and carry on -- not to lose the run. Every call here is
therefore wrapped, degrades to writing on local disk, and gives up on the remote
backend permanently after a few consecutive failures rather than retrying forever.

Three backends, chosen by ``cfg.tracking.mode``:

``disabled``  nothing is recorded. For tests and quick debugging.
``offline``   scalars to a JSONL file and images to disk. No network at all.
``online``    Weights & Biases, plus the same local copies as a fallback.

Local copies are written in *every* mode except disabled, deliberately. If the
remote run is lost or the account is unreachable later, the samples are still on
disk next to the checkpoints.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb

    _HAS_WANDB = True
except ImportError:  # pragma: no cover - the common case locally
    wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False

MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class Tracker:
    """Base tracker: writes scalars to JSONL and images to disk.

    Used directly for ``offline`` mode, and as the local half of ``online``.
    """

    directory: Path
    run_name: str = "run"
    _scalars_path: Path = field(init=False)
    _images_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self._images_dir = self.directory / "samples"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._scalars_path = self.directory / "scalars.jsonl"

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, "wall_time": time.time(), **_plain(metrics)}
        with open(self._scalars_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_images(
        self,
        name: str,
        images: list[np.ndarray],
        step: int,
        captions: list[str] | None = None,
    ) -> Path | None:
        """Compose ``images`` into one contact sheet and write it to disk.

        A grid rather than separate files on purpose: the question being asked of
        these samples is "are these all the same picture?", and that is answered
        by seeing them side by side, not by opening sixteen files.
        """
        if not images:
            return None
        sheet = make_grid(images, captions=captions)
        path = self._images_dir / f"{name}_{step:08d}.png"
        _write_png(path, sheet)
        return path

    def finish(self) -> None:
        pass

    @property
    def scalars_path(self) -> Path:
        return self._scalars_path

    @property
    def images_dir(self) -> Path:
        return self._images_dir


class NullTracker(Tracker):
    """Records nothing. Keeps call sites free of ``if tracker is not None``."""

    def __init__(self) -> None:
        self.directory = Path(".")
        self.run_name = "disabled"

    def log(self, metrics: dict[str, Any], step: int) -> None:
        return None

    def log_images(self, name, images, step, captions=None):  # type: ignore[override]
        return None

    def finish(self) -> None:
        return None

    @property
    def scalars_path(self) -> Path:
        raise RuntimeError("NullTracker records nothing")

    @property
    def images_dir(self) -> Path:
        raise RuntimeError("NullTracker records nothing")


class WandbTracker(Tracker):
    """Weights & Biases, with the local tracker underneath as a safety net.

    ``entity`` is a username -- a public identifier, safe in a config file. The
    API key is read by wandb itself from the WANDB_API_KEY environment variable
    and is never passed through here.
    """

    def __init__(
        self,
        directory: Path,
        run_name: str = "run",
        project: str = "nib",
        entity: str | None = None,
        config: Any = None,
    ) -> None:
        super().__init__(directory=directory, run_name=run_name)
        self._failures = 0
        self._remote_alive = False
        self._run = None

        if not _HAS_WANDB:
            self._warn("wandb is not installed; recording locally only. pip install wandb")
            return
        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                dir=str(self.directory),
                config=_plain(config) if config is not None else None,
            )
            self._remote_alive = True
        except Exception as exc:
            self._warn(f"could not start a W&B run ({exc}); recording locally only")

    @property
    def remote_alive(self) -> bool:
        return self._remote_alive

    def log(self, metrics: dict[str, Any], step: int) -> None:
        super().log(metrics, step)
        if not self._remote_alive:
            return
        try:
            self._run.log(_plain(metrics), step=step)
            self._failures = 0
        except Exception as exc:
            self._remote_failed(exc)

    def log_images(self, name, images, step, captions=None):  # type: ignore[override]
        path = super().log_images(name, images, step, captions)
        if not self._remote_alive or path is None:
            return path
        try:
            self._run.log({name: wandb.Image(str(path))}, step=step)
            self._failures = 0
        except Exception as exc:
            self._remote_failed(exc)
        return path

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception as exc:
                self._warn(f"error closing the W&B run: {exc}")

    def _remote_failed(self, exc: Exception) -> None:
        self._failures += 1
        self._warn(f"W&B logging failed ({exc})")
        if self._failures >= MAX_CONSECUTIVE_FAILURES:
            # Stop trying. Retrying a dead connection every step for the rest of a
            # multi-day run wastes more time than the logging is worth.
            self._remote_alive = False
            self._warn(
                f"giving up on W&B after {self._failures} consecutive failures. "
                f"Local records continue in {self.directory}."
            )

    @staticmethod
    def _warn(message: str) -> None:
        print(f"[tracking] {message}")


def make_tracker(
    config: Any,
    directory: Path | str,
    run_name: str = "run",
) -> Tracker:
    """Build the tracker described by ``config.tracking.mode``."""
    mode = str(getattr(config.tracking, "mode", "offline")).lower()
    directory = Path(directory)

    if mode == "disabled":
        return NullTracker()
    if mode == "offline":
        return Tracker(directory=directory, run_name=run_name)
    if mode == "online":
        return WandbTracker(
            directory=directory,
            run_name=run_name,
            project=str(config.tracking.project),
            entity=config.tracking.entity,
            config=config,
        )
    raise ValueError(f"unknown tracking mode {mode!r}; expected disabled, offline or online")


def make_grid(
    images: list[np.ndarray],
    columns: int | None = None,
    pad: int = 6,
    background: int = 255,
    captions: list[str] | None = None,
) -> np.ndarray:
    """Lay images out in a grid, padding rather than stretching them.

    Handwriting crops have wildly different widths. Resizing them to a common box
    would distort exactly the aspect ratio the samples are being inspected for, so
    each cell is padded to the largest instead.
    """
    if not images:
        raise ValueError("no images to lay out")

    grays = [_to_gray(image) for image in images]
    if captions is not None and len(captions) != len(grays):
        raise ValueError(f"got {len(captions)} captions for {len(grays)} images")

    columns = columns or min(4, len(grays))
    rows = (len(grays) + columns - 1) // columns
    cell_h = max(image.shape[0] for image in grays)
    cell_w = max(image.shape[1] for image in grays)

    sheet = np.full(
        (rows * (cell_h + pad) + pad, columns * (cell_w + pad) + pad),
        background,
        dtype=np.uint8,
    )
    for index, image in enumerate(grays):
        row, column = divmod(index, columns)
        top = pad + row * (cell_h + pad)
        left = pad + column * (cell_w + pad)
        sheet[top : top + image.shape[0], left : left + image.shape[1]] = image
    return sheet


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Accept float [0,1], float [-1,1], uint8, grayscale or colour."""
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.moveaxis(array, 0, -1)  # CHW -> HWC
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] == 3:
            array = array.mean(axis=-1)
    if array.ndim != 2:
        raise ValueError(f"cannot render an image of shape {np.asarray(image).shape}")

    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        low = float(array.min())
        high = float(array.max())
        if low < -0.01:  # generator output in [-1, 1]
            array = (array + 1.0) / 2.0
        elif high > 1.01:  # already 0..255 but stored as float
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)
    return array


def _write_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _plain(value: Any) -> Any:
    """Convert configs and tensors to plain data that json and wandb both accept."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:  # pragma: no cover
        pass

    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and hasattr(value, "detach"):  # torch scalar
        try:
            return value.detach().cpu().item()
        except Exception:
            return str(value)
    return value
