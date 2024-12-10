"""
Copyright (c) 2024 Alec Thomson. All rights reserved.

cutout-fits: A package to produce cutouts of (remote) FITS files.
"""

from __future__ import annotations

from cutout_fits.cutout import make_cutout

from ._version import version as __version__

__all__ = ["__version__", "make_cutout"]
