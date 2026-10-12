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
from fsspec.spec import AbstractFileSystem
from fsspec.utils import merge_offset_ranges

from cutout_fits.logger import logger, set_verbosity

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
    if not wcs.has_celestial:
        msg = "WCS does not have celestial coordinates - cannot make spatial cutout"
        raise ValueError(msg)

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
    # Clip to the image - negative indices wrap around in numpy
    n_ra, n_dec = wcs.celestial.pixel_shape
    start_dec_index = max(int(np.floor(min(y_bottom, _y_bottom, y_top, _y_top))), 0)
    end_dec_index = min(int(np.ceil(max(y_bottom, _y_bottom, y_top, _y_top))), n_dec)
    start_ra_index = max(int(np.floor(min(x_left, x_right, _x_left, _x_right))), 0)
    end_ra_index = min(int(np.ceil(max(x_left, x_right, _x_left, _x_right))), n_ra)

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
    if not wcs.has_spectral:
        msg = "WCS does not have spectral coordinates - cannot make spectral cutout"
        raise ValueError(msg)
    if wcs.wcs.specsys == "":
        logger.warning("No spectral system defined in WCS - assuming TOPOCENT")
        wcs.wcs.specsys = "TOPOCENT"

    if start_freq is None and end_freq is None:
        return SpectralIndex(start_freq_index=None, end_freq_index=None)

    # Default to the cube's frequency limits - works for either channel ordering
    n_chan = wcs.spectral.pixel_shape[0]
    edges = wcs.spectral.array_index_to_world([0, n_chan - 1])
    start = int(
        wcs.spectral.world_to_array_index(
            edges.min() if start_freq is None else start_freq
        )
    )
    end = int(
        wcs.spectral.world_to_array_index(edges.max() if end_freq is None else end_freq)
    )
    start, end = sorted((start, end))

    # Include the end channel
    return SpectralIndex(
        start_freq_index=max(start, 0),
        end_freq_index=min(end + 1, n_chan),
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
    spectral_indices = (
        get_spectral_indices(wcs, start_freq, end_freq)
        if wcs.has_spectral
        else SpectralIndex(start_freq_index=None, end_freq_index=None)
    )
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

    for key, s in zip(array_ordering, slicer, strict=True):
        if s.start is not None and s.stop is not None and s.start >= s.stop:
            msg = f"Cutout does not overlap the image along {key}"
            raise ValueError(msg)
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
    shape_dict = dict(zip(crtypes, shape_int, strict=False))
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


def fetch_section(
    hdu: fits.PrimaryHDU | fits.ImageHDU,
    slicer: tuple[slice, ...],
    fs: AbstractFileSystem,
    path: str,
    join_within_bytes: int = 128 * 1024,
    max_request_bytes: int = 16 * 1024**2,
    max_parallel_requests: int = 64,
    retries: int = 2,
) -> np.ndarray:
    """Get a cutout, requesting all rows from the server at once

    Same result as ``hdu.section[slicer]``, which reads one row at a time.
    Falls back to that for compressed or scaled data.

    Args:
        hdu (fits.PrimaryHDU | fits.ImageHDU): Image HDU opened from ``path``
        slicer (tuple[slice, ...]): Slicer for the cutout - in numpy order
        fs (AbstractFileSystem): Filesystem holding ``path``
        path (str): Path to the FITS file on ``fs``
        join_within_bytes (int, optional): Download two rows in one request (including the pixels between them) if they are fewer than this many bytes apart in the file. Defaults to 128 KiB.
        max_request_bytes (int, optional): Maximum bytes per request. Defaults to 16 MiB.
        max_parallel_requests (int, optional): Requests sent at once. Defaults to 64.
        retries (int, optional): Retries for failed requests. Defaults to 2.

    Returns:
        np.ndarray: Cutout data
    """
    fileinfo = hdu.fileinfo()
    header = hdu.header
    if (
        fileinfo is None
        or fileinfo["file"].compression is not None
        or isinstance(hdu, fits.CompImageHDU)
        or header.get("BSCALE", 1) != 1
        or header.get("BZERO", 0) != 0
        or len(hdu.shape) < 2
    ):
        logger.debug("Falling back to astropy section")
        section: np.ndarray = hdu.section[slicer]
        return section

    shape = hdu.shape
    dtype = hdu.section.dtype.newbyteorder(">")
    itemsize = dtype.itemsize
    *outer, inner = (
        slice(*s.indices(n)[:2]) for s, n in zip(slicer, shape, strict=True)
    )
    out_shape = (*(s.stop - s.start for s in outer), inner.stop - inner.start)
    row_len = out_shape[-1]

    # Byte range of each row in the file
    grid = np.indices(out_shape[:-1]).reshape(len(outer), -1)
    grid += np.array([s.start for s in outer])[:, None]
    rows = np.ravel_multi_index(tuple(grid), shape[:-1])
    starts = fileinfo["datLoc"] + (rows * shape[-1] + inner.start) * itemsize
    ends = starts + row_len * itemsize

    n_rows = len(starts)
    _, block_starts, block_ends = merge_offset_ranges(
        [path] * n_rows,
        starts.tolist(),
        ends.tolist(),
        max_gap=join_within_bytes,
        max_block=max_request_bytes,
        sort=False,
    )
    n_blocks = len(block_starts)
    logger.info("Fetching %d rows in %d requests", n_rows, n_blocks)
    kwargs = {"batch_size": max_parallel_requests} if fs.async_impl else {}
    blocks = list(
        fs.cat_ranges(
            [path] * n_blocks, block_starts, block_ends, on_error="return", **kwargs
        )
    )
    for _ in range(retries):
        failed = [i for i, b in enumerate(blocks) if isinstance(b, Exception)]
        if not failed:
            break
        logger.warning("Retrying %d of %d failed requests", len(failed), n_blocks)
        retried = fs.cat_ranges(
            [path] * len(failed),
            [block_starts[i] for i in failed],
            [block_ends[i] for i in failed],
            on_error="return",
            **kwargs,
        )
        for i, block in zip(failed, retried, strict=True):
            blocks[i] = block
    for block, start, end in zip(blocks, block_starts, block_ends, strict=True):
        if isinstance(block, Exception):
            raise block
        # Some servers ignore the range and send the whole file
        if len(block) != end - start:
            msg = f"Expected {end - start} bytes from {path}, got {len(block)}"
            raise OSError(msg)

    # Pick each row out of the downloaded blocks
    block_starts_arr = np.array(block_starts)
    block_offsets = np.cumsum([0, *(len(b) for b in blocks[:-1])])
    owner = np.searchsorted(block_starts_arr, starts, side="right") - 1
    row_offsets = (block_offsets[owner] + starts - block_starts_arr[owner]) // itemsize
    flat = np.frombuffer(b"".join(blocks), dtype=dtype)
    data: np.ndarray = flat[row_offsets[:, None] + np.arange(row_len)]
    return data.reshape(out_shape).astype(dtype.newbyteorder("="))


def make_cutout(
    infile: str,
    outfile: str,
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    freq_start_hz: float | None = None,
    freq_end_hz: float | None = None,
    overwrite: bool = False,
    **fetch_kwargs: int,
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
        **fetch_kwargs (int): Download settings passed to `fetch_section`.

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

    fs, path = fsspec.core.url_to_fs(infile, **fsspec_kwargs)
    cutout_hdulist = fits.HDUList()
    with (
        # Mostly reads headers - fsspec's default 5 MB block is overkill
        fs.open(path, "rb", block_size=1024**2) as fobj,
        fits.open(fobj, output_verify="silentfix") as hdul,
    ):
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
                logger.info("Found BEAMS ext! Cutting out beam table...")
                needs_beamtable_indices = np.where(needs_beamtable)[0]
                is_beamtable_indices = np.where(is_beamtable)[0]
                needs_beamtable_count = len(needs_beamtable_indices)
                is_beamtable_count = len(is_beamtable_indices)
                if needs_beamtable_count != 1 or is_beamtable_count != 1:
                    msg = f"We are expecting exactly one BEAMS extension and one header that requires a beam table, but we found {is_beamtable_count} and {needs_beamtable_count}!"
                    raise ValueError(msg)
                needs_beamtable_ext = int(needs_beamtable_indices[0])
                is_beamtable_ext = int(is_beamtable_indices[0])
                assert is_beamtable_ext == ext, (
                    "Beam table extension is not where we think it is!"
                )
                logger.debug("Original BEAMS ext: \n%s", hdu.header.tostring(sep="\n"))
                image_hdu: fits.PrimaryHDU | fits.ImageHDU = hdul[needs_beamtable_ext]
                image_wcs = WCS(image_hdu.header)  # pylint: disable=no-member
                cutout_beam_hdu = cutout_beamtable(
                    hdu=hdu,
                    image_wcs=image_wcs,
                    freq_start_hz=freq_start_hz,
                    freq_end_hz=freq_end_hz,
                )
                cutout_hdulist.append(cutout_beam_hdu)
                logger.debug(
                    "Cutout BEAMS ext: \n%s", cutout_beam_hdu.header.tostring(sep="\n")
                )
                continue

            # Only cutout primary and image HDUs with data
            # All other HDUs are passed through
            if (
                not isinstance(hdu, fits.PrimaryHDU | fits.ImageHDU)
                or hdu.header.get("NAXIS", 0) == 0
            ):
                logger.debug("Skipping cutting HDU: %s", hdu)
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
            cutout_data = fetch_section(hdu, slicer, fs, path, **fetch_kwargs)
            tock = time()
            logger.info("Cutout obtained! Took %.2f seconds", tock - tick)
            # Make sure the header is updated to reflect the cutout
            logger.info("Updating header...")
            cutout_header = update_header(header, slicer)
            cutout_hdulist.append(type(hdu)(cutout_data, cutout_header))

        logger.info("Writing cutout to %s", outfile)
        cutout_hdulist.writeto(outfile, overwrite=overwrite)

    return cutout_hdulist


def get_file_parser(parent_parser: bool = False) -> argparse.ArgumentParser:
    cutout_parser = argparse.ArgumentParser(
        description="Make a cutout of a FITS file", add_help=not parent_parser
    )
    parser = cutout_parser.add_argument_group("File options")
    parser.add_argument("infile", help="Path to input FITS file - can be a remote URL")
    parser.add_argument("outfile", help="Path to output FITS file")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    return cutout_parser


def get_cutout_parser(parent_parser: bool = False) -> argparse.ArgumentParser:
    cutout_parser = argparse.ArgumentParser(
        description="Make a cutout of a FITS file", add_help=not parent_parser
    )
    parser = cutout_parser.add_argument_group("Cutout options")
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
    download = cutout_parser.add_argument_group("Download options (see README)")
    download.add_argument(
        "--join-within-bytes",
        type=int,
        default=None,
        help="Download two rows in one request if fewer than this many bytes apart",
    )
    download.add_argument(
        "--max-parallel-requests",
        type=int,
        default=None,
        help="Requests sent to the server at once",
    )
    return cutout_parser


def get_fetch_kwargs(args: argparse.Namespace) -> dict[str, int]:
    """Download options set on the command line

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        dict[str, int]: Options to pass to `fetch_section`
    """
    names = ("join_within_bytes", "max_parallel_requests")
    return {n: getattr(args, n) for n in names if getattr(args, n) is not None}


def main() -> None:
    file_parser = get_file_parser(parent_parser=True)
    cutout_parser = get_cutout_parser(parent_parser=True)
    parser = argparse.ArgumentParser(
        parents=[file_parser, cutout_parser],
        description=cutout_parser.description,
    )
    args = parser.parse_args()

    set_verbosity(
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
        **get_fetch_kwargs(args),
    )


if __name__ == "__main__":
    main()
