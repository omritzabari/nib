"""Smoke test: proves the install -> import -> run chain is closed.

Trivial on purpose. Its value is not what it asserts but when it fails: if the
package is not installed, the src layout is wrong, or pytest cannot find the
package, this is where you find out -- in five seconds, locally, rather than
twenty minutes into a Colab session.
"""

import nib


def test_package_imports():
    assert nib.__version__


def test_subpackages_import():
    """Every subpackage is reachable. Catches a missing __init__.py early."""
    import importlib

    for name in [
        "nib.data",
        "nib.models",
        "nib.losses",
        "nib.engine",
        "nib.inference",
        "nib.api",
        "nib.utils",
    ]:
        assert importlib.import_module(name) is not None
