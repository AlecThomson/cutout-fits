from __future__ import annotations

import importlib.metadata

import cutout_fits as m


def test_version():
    assert importlib.metadata.version("cutout_fits") == m.__version__
