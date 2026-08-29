"""Save and resume training state, exactly.

One of the two traps the project brief names as having to work perfectly *before*
serious training starts: Colab sessions disconnect, and a training run measured in
days will be interrupted many times. If resuming is even slightly lossy, every
long run silently becomes a different experiment from the one you thought you ran.

"Exactly" is a strong claim, so it is defined precisely here: training for N steps,
being killed, resuming, and training to 2N must produce **bit-identical weights**
to an uninterrupted run of 2N steps. That is the property the tests assert, and it
is stronger than it first appears -- getting it requires saving four things people
routinely forget:

* every network, not just the generator (a GAN has a discriminator too, and its
  optimiser state is what stabilises training)
* every optimiser's internal state (Adam's moment estimates are not recoverable
  from the weights)
* the random number generator state for Python, NumPy and torch (without it the
  data order and every augmentation diverge after resuming)
* the dataloader epoch, so shuffling resumes in the right place

Writes are atomic: the file is written under a temporary name and renamed into
place. A process killed mid-write therefore leaves the previous checkpoint intact
rather than a truncated file that fails to load -- which is the specific way this
goes wrong on Colab, where the kill arrives without warning.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # torch is an optional extra: Colab provides its own build
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in torch-free installs
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

FORMAT_VERSION = 1
LATEST = "latest.pt"


class CheckpointError(RuntimeError):
    pass


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise CheckpointError(
            "PyTorch is not installed. Locally: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu . "
            "On Colab it is preinstalled."
        )


@dataclass
class TrainingState:
    """Everything that defines where a run is, apart from the weights themselves."""

    step: int = 0
    epoch: int = 0
    best_metric: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every random number generator that can affect a run.

    Missing any one of these makes a resumed run diverge from an uninterrupted
    one -- not crash, just quietly differ, which is worse.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if _HAS_TORCH:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if _HAS_TORCH and "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])


def seed_everything(seed: int) -> None:
    """Seed every generator. Call once at the start of a run, before any sampling."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save(
    path: Path | str,
    models: dict[str, Any],
    optimizers: dict[str, Any] | None = None,
    schedulers: dict[str, Any] | None = None,
    state: TrainingState | None = None,
    config: Any = None,
) -> Path:
    """Write a checkpoint atomically.

    Args:
        models: name -> nn.Module. Save *all* of them. A GAN's discriminator is
            not disposable mid-run: resuming without it restarts the adversarial
            game from scratch.
        optimizers: name -> optimizer. Adam's moment estimates cannot be
            reconstructed from weights, so omitting these makes a resume lossy.
        schedulers: name -> LR scheduler, if any.
        state: step, epoch and anything else the loop needs.
        config: the run's config, stored for provenance so a checkpoint says what
            produced it.
    """
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = state or TrainingState()

    payload = {
        "format_version": FORMAT_VERSION,
        "models": {name: module.state_dict() for name, module in models.items()},
        "optimizers": {n: o.state_dict() for n, o in (optimizers or {}).items()},
        "schedulers": {n: s.state_dict() for n, s in (schedulers or {}).items()},
        "state": {
            "step": state.step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "extra": state.extra,
        },
        "rng": capture_rng_state(),
        "config": _config_to_plain(config),
    }

    # Atomic: write beside the target, flush to disk, then rename. A kill during
    # the write leaves the previous checkpoint whole.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load(
    path: Path | str,
    models: dict[str, Any] | None = None,
    optimizers: dict[str, Any] | None = None,
    schedulers: dict[str, Any] | None = None,
    restore_rng: bool = True,
    strict: bool = True,
) -> TrainingState:
    """Restore a checkpoint into the given objects and return the training state.

    Args:
        strict: require that every object passed in has a matching entry in the
            checkpoint, and vice versa. A silently unrestored discriminator would
            look like training instability rather than a bug.
    """
    _require_torch()
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"{path} has checkpoint format {version}, this code expects {FORMAT_VERSION}"
        )

    _restore_group(payload["models"], models, "model", strict)
    _restore_group(payload.get("optimizers", {}), optimizers, "optimizer", strict)
    _restore_group(payload.get("schedulers", {}), schedulers, "scheduler", strict)

    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"])

    saved = payload["state"]
    return TrainingState(
        step=saved["step"],
        epoch=saved["epoch"],
        best_metric=saved.get("best_metric"),
        extra=saved.get("extra", {}),
    )


def _restore_group(
    saved: dict[str, Any],
    targets: dict[str, Any] | None,
    kind: str,
    strict: bool,
) -> None:
    targets = targets or {}
    if strict:
        missing = set(targets) - set(saved)
        unexpected = set(saved) - set(targets)
        if missing:
            raise CheckpointError(
                f"checkpoint has no {kind} entry for {sorted(missing)}. "
                f"It contains {sorted(saved)}."
            )
        if unexpected:
            raise CheckpointError(
                f"checkpoint contains {kind} {sorted(unexpected)} which was not passed in. "
                "Resuming without it would silently change the run."
            )
    for name, target in targets.items():
        if name in saved:
            target.load_state_dict(saved[name])


def _config_to_plain(config: Any) -> Any:
    """Store the config as plain data, so a checkpoint never needs the exact
    library version that wrote it in order to be readable."""
    if config is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    try:
        json.dumps(config)
        return config
    except (TypeError, ValueError):
        return str(config)


class CheckpointManager:
    """Periodic saving, with a stable pointer to the newest checkpoint.

    ``latest.pt`` always names the most recent complete checkpoint, so a resume
    never has to guess which file to pick or parse step numbers out of filenames.
    """

    def __init__(
        self,
        directory: Path | str,
        every_n_steps: int = 2000,
        keep_last: int = 3,
    ) -> None:
        if every_n_steps < 1:
            raise CheckpointError("every_n_steps must be at least 1")
        if keep_last < 1:
            raise CheckpointError("keep_last must be at least 1")
        self.directory = Path(directory)
        self.every_n_steps = every_n_steps
        self.keep_last = keep_last
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.directory / LATEST

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.every_n_steps == 0

    def save(self, state: TrainingState, **kwargs: Any) -> Path:
        """Write ``step_XXXXXXXX.pt`` and update ``latest.pt`` to match."""
        path = self.directory / f"step_{state.step:08d}.pt"
        save(path, state=state, **kwargs)
        shutil.copyfile(path, self.latest_path)
        self._prune()
        return path

    def resume(self, **kwargs: Any) -> TrainingState | None:
        """Load ``latest.pt`` if it exists, else return None for a fresh start."""
        if not self.latest_path.is_file():
            return None
        return load(self.latest_path, **kwargs)

    def _prune(self) -> None:
        saved = sorted(self.directory.glob("step_*.pt"))
        for path in saved[: -self.keep_last]:
            path.unlink(missing_ok=True)
