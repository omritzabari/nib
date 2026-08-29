"""Tests for the config system.

The risks worth testing here are not "does it load a YAML file" but the three
things that would silently corrupt a training run: an unresolved interpolation,
a typo'd key being ignored, and a relative root that means something different
depending on where the script was launched from.
"""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import (
    ConfigAttributeError,
    ConfigKeyError,
    ReadonlyConfigError,
    ValidationError,
)

from nib.config import ensure_dirs, find_repo_root, get_path, load_config

REPO_ROOT = find_repo_root()
BASE_YAML = REPO_ROOT / "configs" / "base.yaml"


def test_defaults_load():
    cfg = load_config()
    assert cfg.seed == 1337
    assert cfg.data.image_height == 64


def test_root_is_absolute_and_is_the_repo():
    """A relative root means "wherever you ran from", which differs between the
    laptop and Colab. It must always resolve to an absolute path."""
    cfg = load_config()
    root = get_path(cfg, "root")
    assert root.is_absolute()
    assert (root / "pyproject.toml").is_file()


def test_interpolations_resolve_and_nest():
    """paths.iam derives from paths.raw derives from paths.data derives from root.
    If interpolation silently failed we would get the literal '${paths.raw}/iam'."""
    cfg = load_config()
    root = str(get_path(cfg, "root"))
    for name in ["data", "raw", "processed", "iam", "personal", "checkpoints"]:
        value = str(get_path(cfg, name))
        assert "${" not in value, f"paths.{name} left unresolved"
        assert value.startswith(root), f"paths.{name} escaped the root"


def test_moving_root_moves_everything():
    """The whole point of a single path root: change one value, all paths follow."""
    cfg = load_config(overrides=["paths.root=/tmp/elsewhere"])
    assert str(get_path(cfg, "iam")).replace("\\", "/").endswith("/data/raw/iam")
    assert "elsewhere" in str(get_path(cfg, "iam"))


def test_base_yaml_is_valid_against_the_schema():
    """configs/base.yaml is checked in. If it drifts from the schema, fail here
    rather than at the start of a Colab run."""
    cfg = load_config(BASE_YAML)
    assert cfg.data.charset == "english"
    assert cfg.fixture.num_writers == 20


def test_override_beats_file_beats_default():
    cfg = load_config(BASE_YAML, overrides=["data.image_height=128"])
    assert cfg.data.image_height == 128


def test_unknown_key_raises():
    """A typo'd key must fail loudly. Silently keeping the default is the bug that
    costs a training run to discover."""
    with pytest.raises((ConfigKeyError, ConfigAttributeError, ValidationError)):
        load_config(overrides=["data.image_hieght=128"])


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        load_config(overrides=["data.image_height=tall"])


def test_config_is_readonly():
    """Mutating config mid-run makes a logged run unreproducible."""
    cfg = load_config()
    with pytest.raises(ReadonlyConfigError):
        cfg.seed = 1


def test_ensure_dirs_creates_and_is_idempotent(tmp_path):
    cfg = load_config(overrides=[f"paths.root={tmp_path.as_posix()}"])
    made = ensure_dirs(cfg, "processed", "checkpoints")
    assert all(p.is_dir() for p in made)
    ensure_dirs(cfg, "processed")  # second call must not raise
    assert get_path(cfg, "processed").is_dir()


def test_loading_does_not_create_directories(tmp_path):
    """load_config must be free of side effects, or tests that merely load a
    config start littering the filesystem."""
    target = tmp_path / "nothing_here"
    cfg = load_config(overrides=[f"paths.root={target.as_posix()}"])
    assert not target.exists()
    assert OmegaConf.is_readonly(cfg)
