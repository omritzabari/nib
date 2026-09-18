"""Saving a trained adapter, and putting it back.

An adaptation is trained once and used for ever: a general one, taught on many
writers, ships with the system, and a per-writer one -- if one is ever shown to
help -- belongs to that user. Either way it is a few megabytes beside a model of
gigabytes, and it has to survive leaving the process.

Only the adapter's own tensors are written. The frozen model is not copied, so a
file here is meaningless without the checkpoint it was trained against, and
:func:`load` refuses one that does not fit rather than leaving half a model.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "lora_"
"""What names an adapter's parameters, as peft writes them."""


def _adapter_parameters(model) -> dict:
    return {name: p for name, p in model.named_parameters() if MARKER in name}


def save(model, path: Path | str) -> Path:
    """Write the adapter's weights, and only those. Returns the path."""
    import torch

    found = _adapter_parameters(model)
    if not found:
        raise ValueError("this model has no adapter to save; attach one first")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: p.detach().cpu().clone() for name, p in found.items()}, path)
    return path


def load(model, path: Path | str) -> int:
    """Put a saved adapter back on a model that already has one attached.

    Returns how many tensors were restored. The model must carry an adapter of the
    same shape -- attach it exactly as it was attached for training -- because the
    weights are meaningless anywhere else.
    """
    import torch

    stored = torch.load(Path(path), map_location="cpu", weights_only=True)
    found = _adapter_parameters(model)
    if not found:
        raise ValueError(f"{path}: this model has no adapter attached to load into")
    missing = set(stored) - set(found)
    if missing or set(found) - set(stored):
        raise ValueError(
            f"{path} does not fit this model: {len(missing)} of its {len(stored)} tensors "
            f"have no place here, e.g. {sorted(missing)[:2]}"
        )
    with torch.no_grad():
        for name, parameter in found.items():
            value = stored[name]
            if value.shape != parameter.shape:
                raise ValueError(
                    f"{path} does not fit this model: {name} is {tuple(value.shape)}, "
                    f"the model wants {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(parameter.device))
    return len(stored)
