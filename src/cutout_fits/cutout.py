#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from time import time
from typing import NamedTuple

import fsspec.core
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from dotenv import load_dotenv

from cutout_fits.logging import set_verbosity, setup_logger

logger = setup_logger()

_ = load_dotenv()


class SpatialIndex(NamedTuple):
    """
    Start and end indices for spatial axes.

    Attributes:
        start_ra_index: int | None
        end_ra_index: int | None
        start_dec_index: int | None
        end_dec_index: int | None
    """

    start_ra_index: int | None
    """Start index for RA axis"""
    end_ra_index: int | None
    """End index for RA axis"""
    start_dec_index: int | None
    """Start index for Dec axis"""
    end_dec_index: int | None
    """End index for Dec axis"""


class SpectralIndex(NamedTuple):
    """
    Start and end indices for spectral axes.

    Attributes:
        start_freq_index: int | None
        end_freq_index: int | None
    """

    start_freq_index: int | None
    """Start index for frequency axis"""
    end_freq_index: int | None
    """End index for frequency axis"""


class StokesIndex(NamedTuple):
    """
    Start and end indices for Stokes axes.

    Attributes:
        start_stokes_index: int | None
        end_stokes_index: int | None
    """

    start_stokes_index: int | None
    """Start index for Stokes axis"""
    end_stokes_index: int | None
    """End index for Stokes axis"""


def get_spatial_indices(
    wcs: WCS,
    centre: SkyCoord,
    radius: u.Quantity,
) -> SpatialIndex:
    """Get the start and end indices for spatial axes

    Args:
        wcs (WCS): WCS for HDU
        centre (SkyCoord): Centre of cutout
        radius (u.Quantity): Radius of cutout

    Returns:
        SpatialIndex: start_ra_index, end_ra_index, start_dec_index, end_dec_index
    """
    assert (
        wcs.has_celestial
    ), "WCS does not have celestial coordinates - cannot make spatial cutout"

    # Offset each corner by the radius in the diagonal direction
    # sqrt(2) is used to get the diagonal distance - i.e. a square cutout
    top_right = centre.directional_offset_by(
        position_angle=-45 * u.deg,
        separation=radius * np.sqrt(2),
    )
    bottom_left = centre.directional_offset_by(
        position_angle=135 * u.deg,
        separation=radius * np.sqrt(2),
    )
    top_left = centre.directional_offset_by(
        position_angle=45 * u.deg,
        separation=radius * np.sqrt(2),
    )
    bottom_right = centre.directional_offset_by(
        position_angle=-135 * u.deg,
        separation=radius * np.sqrt(2),
    )

    x_left, y_bottom = skycoord_to_pixel(bottom_left, wcs.celestial)
    x_right, y_top = skycoord_to_pixel(top_right, wcs.celestial)
    _x_left, _y_top = skycoord_to_pixel(top_left, wcs.celestial)
    _x_right, _y_bottom = skycoord_to_pixel(bottom_right, wcs.celestial)

    # Check for NaNs - for left/bottom NaN -> 0, for right/top NaN -> image size
    # This is needed when a stupidly big cutout is requested
    x_left = 0 if np.isnan(x_left) else x_left
    y_bottom = 0 if np.isnan(y_bottom) else y_bottom
    x_right = wcs.celestial.pixel_shape[0] if np.isnan(x_right) else x_right
    y_top = wcs.celestial.pixel_shape[1] if np.isnan(y_top) else y_top
    _x_left = 0 if np.isnan(_x_left) else _x_left
    _y_top = wcs.celestial.pixel_shape[1] if np.isnan(_y_top) else _y_top
    _x_right = wcs.celestial.pixel_shape[0] if np.isnan(_x_right) else _x_right
    _y_bottom = 0 if np.isnan(_y_bottom) else _y_bottom

    # Compare all points in case of insanity at the poles
    start_dec_index = int(np.floor(min(y_bottom, _y_bottom, y_top, _y_top)))
    end_dec_index = int(np.ceil(max(y_bottom, _y_bottom, y_top, _y_top)))
    start_ra_index = int(np.floor(min(x_left, x_right, _x_left, _x_right)))
    end_ra_index = int(np.ceil(max(x_left, x_right, _x_left, _x_right)))

    return SpatialIndex(
        start_ra_index=start_ra_index,
        end_ra_index=end_ra_index,
        start_dec_index=start_dec_index,
        end_dec_index=end_dec_index,
    )


def get_spectral_indices(
    wcs: WCS,
    start_freq: u.Quantity | None = None,
    end_freq: u.Quantity | None = None,
) -> SpectralIndex:
    """Get the start and end indices for spectral axes

    Args:
        wcs (WCS): WCS for HDU
        start_freq (u.Quantify | None, optional): Starting frequency. Defaults to None.
        end_freq (u.Quantify | None, optional): End frequency. Defaults to None.

    Returns:
        SpectralIndex: start_freq_index, end_freq_index
    """
    assert (
        wcs.has_spectral
    ), "WCS does not have spectral coordinates - cannot make spectral cutout"
    if wcs.wcs.specsys == "":
        logger.warning("No spectral system defined in WCS - assuming TOPOCENT")
        wcs.wcs.specsys = "TOPOCENT"

    start_freq_index = (
        wcs.spectral.world_to_array_index(start_freq)
        if start_freq is not None
        else None
    )
    end_freq_index = (
        wcs.spectral.world_to_array_index(end_freq) if end_freq is not None else None
    )

    return SpectralIndex(
        start_freq_index=start_freq_index,
        end_freq_index=end_freq_index,
    )


def get_stokes_indices(wcs: WCS) -> StokesIndex:
    # TODO: Stokes selection
    """Get the start and end indices for Stokes axes

    Args:
        wcs (WCS): WCS for HDU

    Returns:
        StokesIndex: start_stokes_index, end_stokes_index
    """
    logger.debug("wcs: %s", wcs)
    return StokesIndex(
        start_stokes_index=None,
        end_stokes_index=None,
    )


def make_slicer(
    wcs: WCS,
    centre: SkyCoord,
    radius: u.Quantity,
    start_freq: u.Quantity | None = None,
    end_freq: u.Quantity | None = None,
) -> tuple[slice, ...]:
    """Create a slicer for a given WCS, centre, radius, and frequency range

    Args:
        wcs (WCS): WCS for HDU
        centre (SkyCoord): Centre of cutout
        radius (u.Quantity): Radius of cutout
        start_freq (u.Quantity | None, optional): Starting frequency. Defaults to None.
        end_freq (u.Quantity | None, optional): End frequnecy. Defaults to None.

    Returns:
        Tuple[slice,...]: Tuple of slices for each axis - in numpy order
    """
    array_ordering = tuple(wcs.wcs.ctype)[::-1]

    spatial_indices = get_spatial_indices(wcs, centre, radius)
    logger.debug("Spatial indices: %s", spatial_indices)
    spectral_indices = get_spectral_indices(wcs, start_freq, end_freq)
    logger.debug("Spectral indices: %s", spectral_indices)
    stokes_indices = get_stokes_indices(wcs)
    logger.debug("Stokes indices: %s", stokes_indices)

    # we only care about the spatial, spectral, and stokes axes
    slice_mapping = {
        "ra": slice(spatial_indices.start_ra_index, spatial_indices.end_ra_index),
        "dec": slice(spatial_indices.start_dec_index, spatial_indices.end_dec_index),
        "freq": slice(
            spectral_indices.start_freq_index, spectral_indices.end_freq_index
        ),
        "stokes": slice(
            stokes_indices.start_stokes_index, stokes_indices.end_stokes_index
        ),
    }

    slicer = []
    for key in array_ordering:
        map_key = key.lower().split("-")[0]
        if map_key not in slice_mapping:
            slicer.append(slice(None))
            continue
        slicer.append(slice_mapping[map_key])
    return tuple(slicer)


def get_cutout_shape(wcs: WCS, slicer: tuple[slice, ...]) -> tuple[int, ...]:
    """Get the shape of a cutout

    Args:
        wcs (WCS): WCS of the cutout
        slicer (tuple[slice, ...]): Slicer for the cutout

    Returns:
        tuple[int, ...]: Shape of the cutout
    """
    shape = []
    for i, s in enumerate(slicer):
        logger.debug("slice: %s", s)
        if s.start is None or s.stop is None:
            shape.append(wcs.array_shape[i])
            continue

        start = max(s.start, 0)
        stop = min(s.stop, wcs.array_shape[i])
        shape.append(stop - start)

    return tuple(shape)


def format_shape(wcs: WCS, shape: tuple[int, ...]) -> str:
    """Produce a string representation of the shape of an cube

    Args:
        wcs (WCS): Cube WCS
        shape (tuple[int,...]): Shape of the cube

    Returns:
        str: String representation of the shape
    """
    crtypes = list(wcs.wcs.ctype)[::-1]
    shape_int = [int(s) for s in shape]
    shape_dict = dict(zip(crtypes, shape_int))
    return str(shape_dict)


def update_header(old_header: fits.Header, slicer: tuple[slice, ...]) -> fits.Header:
    """Update the header to reflect the cutout

    Args:
        old_header (fits.Header): Original header
        slicer (tuple[slice, ...]): Slicer for the cutout

    Returns:
        fits.Header: Cutout header
    """
    new_header = old_header.copy()
    # FITS ordering is reversed
    slicer_fits = slicer[::-1]
    for i, s in enumerate(slicer_fits):
        ax_idx = i + 1
        if s.start is not None:
            new_header[f"CRPIX{ax_idx}"] -= s.start
        if s.stop is not None and s.start is not None:
            new_header[f"NAXIS{ax_idx}"] = s.stop - s.start

    return new_header


def cutout_beamtable(
    hdu: fits.BinTableHDU,
    image_wcs: WCS,
    freq_start_hz: float | None = None,
    freq_end_hz: float | None = None,
) -> fits.BinTableHDU:
    """Cut out a beam table

    Args:
        hdu (fits.BinTableHDU): HDU with the beam table
        image_wcs (WCS): WCS of the image that needs beams
        freq_start_hz (float | None, optional): Starting frequency cut in Hz. Defaults to None.
        freq_end_hz (float | None, optional): End frequnecy cut in Hz. Defaults to None.

    Returns:
        fits.BinTableHDU: Cutout beam table
    """
    spectral_indices = get_spectral_indices(
        wcs=image_wcs,
        start_freq=freq_start_hz * u.Hz if freq_start_hz is not None else None,
        end_freq=freq_end_hz * u.Hz if freq_end_hz is not None else None,
    )
    beam_table = Table.read(hdu)
    beam_slicer = slice(
        spectral_indices.start_freq_index, spectral_indices.end_freq_index
    )
    beam_table_cut = beam_table[beam_slicer]
    beam_table_cut["CHAN"] = np.arange(len(beam_table_cut))
    cut_hdu = fits.BinTableHDU(beam_table_cut, name="BEAMS")
    cut_hdu.header["NCHAN"] = len(beam_table_cut)

    return cut_hdu


def make_cutout(
    infile: str,
    outfile: str,
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    freq_start_hz: float | None = None,
    freq_end_hz: float | None = None,
    overwrite: bool = False,
) -> fits.HDUList:
    """Make a cutout of a FITS file

    Args:
        infile (str): Path to FITS file - can be a remote URL
        outfile (str): Path to output file - must be local
        ra_deg (float): Centre RA in degrees
        dec_deg (float): Centre Dec in degrees
        radius_arcmin (float): Cutout radius in arcminutes
        freq_start_hz (float | None, optional): Start frequency in Hz. Defaults to None.
        freq_end_hz (float | None, optional): End frequency in Hz. Defaults to None.
        overwrite (bool): Whether to overwrite the output file.

    Returns:
        fits.HDUList: Cutout HDUList
    """
    fsspec_kwargs = {}
    if infile.startswith("s3://"):
        logger.info("Attempting to access S3 file...")
        logger.info("Using S3 credentials...")
        fsspec_kwargs["key"] = os.getenv("FSSPEC_S3_KEY")
        fsspec_kwargs["secret"] = os.getenv("FSSPEC_S3_SECRET")
        fsspec_kwargs["endpoint_url"] = os.getenv("FSSPEC_S3_ENDPOINT_URL")

    elif infile.startswith(("https://", "http://")):
        logger.info("Attempting to access HTTP(S) file...")
        fsspec.core.DEFAULT_EXPAND = True

    cutout_hdulist = fits.HDUList()
    with fits.open(
        infile,
        fsspec_kwargs=fsspec_kwargs,
        use_fsspec=True,
        memmap=True,
        mode="denywrite",
        output_verify="silentfix",
    ) as hdul:
        # First check if beams tables are reported
        needs_beamtable = np.zeros(len(hdul), dtype=bool)
        is_beamtable = np.zeros(len(hdul), dtype=bool)
        for ext, hdu in enumerate(hdul):
            if hdu.header.get("CASAMBM", False):
                needs_beamtable[ext] = True
            if hdu.name == "BEAMS":
                is_beamtable[ext] = True

        # Sanity checks
        # - If beams are reported, there must be a beam table
        # - If there is a beam table, beams must be reported
        # - We only support one beam table
        if np.any(needs_beamtable) and not np.any(is_beamtable):
            msg = "Beam table required in header, but no beam table extension found!"
            raise ValueError(msg)
        if np.any(is_beamtable) and not np.any(needs_beamtable):
            msg = (
                "Beam table extension found, but no beam table required in any header!"
            )
            raise ValueError(msg)
        if np.sum(is_beamtable) > 1 or np.sum(needs_beamtable) > 1:
            msg = "Multiple beam tables found! We cannot figure out which one to use!"
            raise ValueError(msg)

        for ext, hdu in enumerate(hdul):
            if hdu.name == "BEAMS":
                needs_beamtable_ext = int(np.where(needs_beamtable)[0])
                is_beamtable_ext = int(np.where(is_beamtable)[0])
                assert (
                    is_beamtable_ext == ext
                ), "Beam table extension is not where we think it is!"
                image_hdu = hdul[needs_beamtable_ext]
                image_wcs = WCS(image_hdu.header)
                cutout_beam_hdu = cutout_beamtable(
                    hdu=hdu,
                    image_wcs=image_wcs,
                    freq_start_hz=freq_start_hz,
                    freq_end_hz=freq_end_hz,
                )
                cutout_hdulist.append(cutout_beam_hdu)
                continue

            # Only cutout primary and image HDUs
            # All other HDUs are passed through
            if not isinstance(hdu, fits.PrimaryHDU) and not isinstance(
                hdu, fits.ImageHDU
            ):
                logger.debug("Skipping HDU: %s", hdu)
                cutout_hdulist.append(hdu)
                continue

            # If a primary HDU or image HDU, make a cutout
            header = hdu.header
            wcs = WCS(header)
            slicer = make_slicer(
                wcs=wcs,
                centre=SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg),
                radius=radius_arcmin * u.arcmin,
                start_freq=freq_start_hz * u.Hz if freq_start_hz is not None else None,
                end_freq=freq_end_hz * u.Hz if freq_end_hz is not None else None,
            )

            # Report info to user
            logger.debug("Slicer: %s", slicer)
            logger.info("Producing cutout...")
            orginal_shape_str = format_shape(wcs, wcs.array_shape)
            logger.info("Original shape: %s", orginal_shape_str)
            cutout_shape = get_cutout_shape(wcs, slicer)
            cutout_shape_str = format_shape(wcs, cutout_shape)
            logger.info("Cutout shape: %s", cutout_shape_str)

            # Download will start here
            tick = time()
            cutout_data = hdu.section[slicer]
            tock = time()
            logger.info("Cutout obtained! Took %.2f seconds", tock - tick)
            # Make sure the header is updated to reflect the cutout
            logger.info("Updating header...")
            cutout_header = update_header(header, slicer)
            cutout_hdulist.append(type(hdu)(cutout_data, cutout_header))

        logger.info("Writing cutout to %s", outfile)
        cutout_hdulist.writeto(outfile, overwrite=overwrite)

    return cutout_hdulist


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
