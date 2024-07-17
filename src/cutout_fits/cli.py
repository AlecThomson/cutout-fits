#!/usr/bin/env python3
from __future__ import annotations

import argparse

from cutout_fits.cutout import make_cutout
from cutout_fits.logging import set_verbosity, setup_logger

logger = setup_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a cutout of a FITS file")
    parser.add_argument("infile", help="Path to input FITS file - can be a remote URL")
    parser.add_argument("outfile", help="Path to output FITS file")
    parser.add_argument("ra_deg", type=float, help="Centre RA in degrees")
    parser.add_argument("dec_deg", type=float, help="Centre Dec in degrees")
    parser.add_argument("radius_arcmin", type=float, help="Cutout radius in arcminutes")
    parser.add_argument(
        "--freq-start",
        type=float,
        help="Start frequency in Hz",
        default=None,
    )
    parser.add_argument(
        "--freq-end",
        type=float,
        help="End frequency in Hz",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    args = parser.parse_args()

    set_verbosity(
        logger=logger,
        verbosity=args.verbosity,
    )

    _ = make_cutout(
        infile=args.infile,
        outfile=args.outfile,
        ra_deg=args.ra_deg,
        dec_deg=args.dec_deg,
        radius_arcmin=args.radius_arcmin,
        freq_start_hz=args.freq_start,
        freq_end_hz=args.freq_end,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
