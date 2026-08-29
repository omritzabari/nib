"""Mechanically enforce rule 1: no file under src/ contains a hardcoded path.

This is the rule the project brief cares most about, because breaking it is what
makes code work on the laptop and fail on Colab. A rule nobody checks is a
suggestion, so it is checked here.

Implementation note: this parses the AST and inspects string literals only.
Comments never reach the AST, and docstrings are skipped deliberately -- prose
explaining "on Colab, pass paths.root=/content/nib" is documentation, not a
hardcoded path.
"""

import ast
import re

from nib.config import find_repo_root

SRC = find_repo_root() / "src"

FORBIDDEN = [
    (re.compile(r"^[A-Za-z]:[\/]"), "Windows absolute path"),
    (re.compile(r"^/(content|home|Users|mnt|media|opt|var)/"), "POSIX absolute path"),
    (re.compile(r"drive/MyDrive", re.IGNORECASE), "hardcoded Google Drive mount"),
]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that are docstrings, so they can be skipped."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _offences_in(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        for pattern, label in FORBIDDEN:
            if pattern.search(node.value):
                found.append(f"{path}:{node.lineno}: {label} -- {node.value!r}")
    return found


def test_no_hardcoded_paths_under_src():
    files = sorted(SRC.rglob("*.py"))
    assert files, "found no Python files under src/ -- is the layout right?"

    offences = [o for f in files for o in _offences_in(f)]
    assert not offences, "Hardcoded paths must come from the config instead:\n" + "\n".join(
        offences
    )


def test_the_detector_actually_detects(tmp_path):
    """A guard that never fires is indistinguishable from a broken guard.

    One string per rule: '/content/drive/MyDrive/x' would trip two rules at once,
    which makes the count meaningless as a test of each rule individually.
    """
    bad = tmp_path / "bad.py"
    bad.write_text(
        'DATA = "C:/Users/someone/data"\n'
        'OUT = "/content/outputs"\n'
        'CKPT = "drive/MyDrive/checkpoints"\n'
    )
    offences = _offences_in(bad)
    assert len(offences) == 3, offences
    joined = " ".join(offences)
    for label in ["Windows absolute path", "POSIX absolute path", "Google Drive mount"]:
        assert label in joined, f"rule not firing: {label}"


def test_docstrings_and_config_interpolations_are_not_offences(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        '"""Docs may mention C:/Users or /content/nib freely."""\n'
        'P = "${paths.root}/data"\n'
        'REL = "data/raw/iam"\n'
    )
    assert _offences_in(good) == []
