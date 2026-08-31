"""Configuration: one typed schema, one path root, no hardcoded paths anywhere else.

The project runs in two places -- a Windows laptop and a Colab VM -- and the whole
point of this module is that the same code runs in both without edits. Everything
that differs between the two environments is a value in a config file, and almost
all of it derives from a single value: ``paths.root``.

Why OmegaConf and not Hydra: Hydra also changes the working directory, creates its
own output tree, and installs a CLI wrapper around your entry point. On Colab that
machinery is a liability. OmegaConf alone gives the three things actually needed --
YAML loading, variable interpolation, and typed validation -- and nothing else.

The schema below is a set of dataclasses rather than a free-form dict, which buys
one thing that matters: a typo in a YAML key raises an error instead of being
silently ignored. A misspelled ``image_hieght`` that quietly keeps the default is
the kind of bug that costs a day of training to notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class PathsConfig:
    """Every path in the project, derived from ``root``.

    The ``${...}`` strings are OmegaConf interpolations: they are resolved after
    merging, so overriding ``paths.root`` moves everything else with it. Set
    ``paths.root=/content/nib`` on Colab and nothing else needs to change.
    """

    root: str = "???"  # filled in by load_config; see _default_root
    data: str = "${paths.root}/data"
    raw: str = "${paths.data}/raw"
    processed: str = "${paths.data}/processed"
    iam: str = "${paths.raw}/iam"
    personal: str = "${paths.raw}/personal"
    fixture: str = "${paths.processed}/fixture"
    checkpoints: str = "${paths.root}/checkpoints"
    outputs: str = "${paths.root}/outputs"

    references: str = "${paths.root}/references"
    """Measured baselines, and the one output directory that is *committed*.

    Everything else under ``outputs`` is a by-product of a run and is ignored by
    git. These are different in kind: they are what real handwriting scores on a
    given pack, they are what every generated result is reported against, and
    they have to survive the trip from the machine that measured them to the one
    that generates. Same reasoning as the committed writer split -- an artefact
    someone else needs in order to reproduce a number you published."""


@dataclass
class DataConfig:
    """Properties of the image data itself."""

    image_height: int = 64
    """Text height in pixels. 64 because every released checkpoint in the field
    operates at 32-64px; 96-128px is off-distribution for all of them."""

    charset: str = "english"


@dataclass
class FixtureConfig:
    """The synthetic stand-in dataset used by the test suite (T3).

    Small on purpose: the whole suite must run in seconds, or it stops being run.
    """

    num_writers: int = 20
    words_per_writer: int = 50
    seed: int = 0


@dataclass
class TrackingConfig:
    """Weights & Biases. ``entity`` is a username, never a secret -- the API key
    belongs in the WANDB_API_KEY environment variable and never in a config file."""

    entity: str | None = None
    project: str = "nib"
    mode: str = "online"  # online | offline | disabled


@dataclass
class Config:
    seed: int = 1337
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    fixture: FixtureConfig = field(default_factory=FixtureConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward looking for pyproject.toml.

    Without this, ``paths.root: "."`` means "wherever you happened to run from",
    which breaks the moment a script is invoked from another directory -- and on
    Colab it is always invoked from somewhere else.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {here}. Pass paths.root explicitly.")


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Build the config: schema defaults, then YAML, then command-line overrides.

    Args:
        config_path: YAML file to merge over the defaults. None uses defaults only.
        overrides: dotted assignments, e.g. ``["data.image_height=128", "seed=7"]``.
            These win over the file, which wins over the schema defaults.

    Returns:
        A resolved, read-only config. Unknown keys raise; type mismatches raise.
    """
    cfg = OmegaConf.structured(Config)

    if config_path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(config_path)))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # Resolve root last, so an override or file value wins over the discovered default.
    if OmegaConf.is_missing(cfg.paths, "root"):
        cfg.paths.root = str(find_repo_root())
    cfg.paths.root = str(Path(cfg.paths.root).expanduser().resolve())

    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def get_path(cfg: DictConfig, name: str) -> Path:
    """Return ``cfg.paths.<name>`` as a Path. Use this rather than reading the
    string directly, so path handling stays in one place."""
    return Path(str(cfg.paths[name]))


def ensure_dirs(cfg: DictConfig, *names: str) -> list[Path]:
    """Create the named path entries if they do not exist, and return them.

    Called explicitly by whoever needs a directory. Creating directories as a side
    effect of loading a config makes ``load_config`` unsafe to call in a test.
    """
    made = []
    for name in names:
        p = get_path(cfg, name)
        p.mkdir(parents=True, exist_ok=True)
        made.append(p)
    return made
