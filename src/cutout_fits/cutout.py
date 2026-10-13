#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from time import sleep, time
from typing import Any, NamedTuple

import fsspec.core
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from dotenv import load_dotenv
from fsspec.asyn import _run_coros_in_chunks, sync
from fsspec.callbacks import Callback
from fsspec.spec import AbstractFileSystem
from fsspec.utils import merge_offset_ranges
from tqdm import tqdm

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


class DownloadPlan(NamedTuple):
    """Byte ranges to request for a cutout"""

    row_starts: np.ndarray
    """File offset of each cutout row"""
    row_bytes: int
    """Bytes in each cutout row"""
    row_spacing_bytes: int
    """Unwanted bytes between neighbouring rows of a channel"""
    max_request_bytes: int
    """Largest single request"""
    max_parallel_requests: int
    """Requests sent at once"""
    request_starts: list[int]
    """File offset where each request starts"""
    request_ends: list[int]
    """File offset where each request ends"""

    @property
    def join_limit_bytes(self) -> int:
        """Rows closer than this in the file are downloaded in one request"""
        # A quarter of a request - matches the measured 16 kB for 64 kB requests
        return self.max_request_bytes // 4

    @property
    def n_rows(self) -> int:
        """Number of cutout rows"""
        return len(self.row_starts)

    @property
    def n_requests(self) -> int:
        """Number of requests to the server"""
        return len(self.request_starts)

    @property
    def wanted_bytes(self) -> int:
        """Bytes in the cutout"""
        return self.n_rows * self.row_bytes

    @property
    def requested_bytes(self) -> int:
        """Bytes to download, including unwanted pixels between joined rows"""
        return sum(self.request_ends) - sum(self.request_starts)

    @property
    def unwanted_bytes(self) -> int:
        """Bytes downloaded that are not in the cutout"""
        return self.requested_bytes - self.wanted_bytes

    def describe(self) -> list[str]:
        """Plain-language summary of the plan, one line per item"""
        kb, mb = 1024, 1024**2
        rows = (
            f"Cutout is {self.n_rows} rows of {self.row_bytes / kb:.1f} kB, "
            f"{self.row_spacing_bytes / kb:.1f} kB apart in the file"
        )
        join = (
            f"Rows less than {self.max_request_bytes / kb:.3g}/4="
            f"{self.join_limit_bytes / kb:.3g} kB apart are downloaded together"
        )
        requests = (
            f"Plan: {self.n_requests} requests, {self.max_parallel_requests} at a time; "
            f"{self.requested_bytes / mb:.1f} MB to download "
            f"({self.wanted_bytes / mb:.1f} MB cutout + "
            f"{self.unwanted_bytes / mb:.1f} MB between rows)"
        )
        return [rows, join, requests]


def section_fallback_reason(hdu: fits.PrimaryHDU | fits.ImageHDU) -> str | None:
    """Why ``fetch_section`` must fall back to ``hdu.section``, if it must

    Args:
        hdu (fits.PrimaryHDU | fits.ImageHDU): Image HDU

    Returns:
        str | None: Reason, or None if byte ranges can be requested directly
    """
    fileinfo = hdu.fileinfo()
    header = hdu.header
    if fileinfo is None:
        return "HDU is not backed by a file"
    if fileinfo["file"].compression is not None or isinstance(hdu, fits.CompImageHDU):
        return "compressed file"
    if header.get("BSCALE", 1) != 1 or header.get("BZERO", 0) != 0:
        return "scaled data (BSCALE/BZERO)"
    if len(hdu.shape) < 2:
        return "1D data"
    return None


def plan_download(
    hdu: fits.PrimaryHDU | fits.ImageHDU,
    slicer: tuple[slice, ...],
    max_request_kb: int = 64,
    max_parallel_requests: int = 64,
) -> DownloadPlan:
    """Work out which byte ranges to request for a cutout

    Args:
        hdu (fits.PrimaryHDU | fits.ImageHDU): Image HDU opened from a file
        slicer (tuple[slice, ...]): Slicer for the cutout - in numpy order
        max_request_kb (int, optional): Largest single request in kB (1024 bytes). Defaults to 64.
        max_parallel_requests (int, optional): Requests sent at once. Defaults to 64.

    Returns:
        DownloadPlan: Requests to make
    """
    shape = hdu.shape
    itemsize = hdu.section.dtype.itemsize
    *outer, inner = (
        slice(*s.indices(n)[:2]) for s, n in zip(slicer, shape, strict=True)
    )
    out_shape = (*(s.stop - s.start for s in outer), inner.stop - inner.start)
    row_len = out_shape[-1]

    # Byte range of each row in the file
    grid = np.indices(out_shape[:-1]).reshape(len(outer), -1)
    grid += np.array([s.start for s in outer])[:, None]
    rows = np.ravel_multi_index(tuple(grid), shape[:-1])
    row_starts = hdu.fileinfo()["datLoc"] + (rows * shape[-1] + inner.start) * itemsize
    row_bytes = row_len * itemsize

    plan = DownloadPlan(
        row_starts=row_starts,
        row_bytes=row_bytes,
        row_spacing_bytes=(shape[-1] - row_len) * itemsize,
        max_request_bytes=max_request_kb * 1024,
        max_parallel_requests=max_parallel_requests,
        request_starts=[],
        request_ends=[],
    )
    _, request_starts, request_ends = merge_offset_ranges(
        [""] * len(row_starts),
        row_starts.tolist(),
        (row_starts + row_bytes).tolist(),
        max_gap=plan.join_limit_bytes,
        max_block=plan.max_request_bytes,
        sort=False,
    )
    return plan._replace(request_starts=request_starts, request_ends=request_ends)


def cat_ranges_with_progress(
    fs: AbstractFileSystem,
    path: str,
    starts: list[int],
    ends: list[int],
    max_parallel_requests: int,
    desc: str,
    disable_progress: bool = False,
) -> list[Any]:
    """Download byte ranges, showing a progress bar

    Args:
        fs (AbstractFileSystem): Filesystem holding ``path``
        path (str): Path to the file on ``fs``
        starts (list[int]): Start of each range
        ends (list[int]): End of each range
        max_parallel_requests (int): Requests sent at once
        desc (str): Progress bar label
        disable_progress (bool, optional): Hide the progress bar. Defaults to False.

    Returns:
        list[Any]: Bytes of each range, or the error it raised
    """
    # disable=None hides the bar when not writing to a terminal (e.g. batch jobs)
    with tqdm(
        total=len(starts), desc=desc, unit=" requests", disable=disable_progress or None
    ) as pbar:
        if fs.async_impl:
            # cat_ranges can't report progress, so run its coroutines here instead
            callback = Callback(hooks={"tqdm": lambda *_, **__: pbar.update()})
            coros = [
                fs._cat_file(path, start=start, end=end)  # pylint: disable=protected-access
                for start, end in zip(starts, ends, strict=True)
            ]
            out = sync(
                fs.loop,
                _run_coros_in_chunks,
                coros,
                batch_size=max_parallel_requests,
                callback=callback,
                return_exceptions=True,
                nofiles=True,
            )
        else:
            out = fs.cat_ranges([path] * len(starts), starts, ends, on_error="return")
        pbar.update(len(starts) - pbar.n)
    return list(out)


def fetch_section(
    hdu: fits.PrimaryHDU | fits.ImageHDU,
    slicer: tuple[slice, ...],
    fs: AbstractFileSystem,
    path: str,
    max_request_kb: int = 64,
    max_parallel_requests: int = 64,
    retries: int = 4,
    disable_progress: bool = False,
) -> np.ndarray:
    """Get a cutout, requesting all rows from the server at once

    Same result as ``hdu.section[slicer]``, which reads one row at a time.
    Falls back to that for compressed or scaled data.

    Args:
        hdu (fits.PrimaryHDU | fits.ImageHDU): Image HDU opened from ``path``
        slicer (tuple[slice, ...]): Slicer for the cutout - in numpy order
        fs (AbstractFileSystem): Filesystem holding ``path``
        path (str): Path to the FITS file on ``fs``
        max_request_kb (int, optional): Largest single request in kB (1024 bytes). Defaults to 64.
        max_parallel_requests (int, optional): Requests sent at once. Defaults to 64.
        retries (int, optional): Retries for failed requests, each after a longer wait (1, 2, 4, ... s) with half as many requests at once. Defaults to 4.
        disable_progress (bool, optional): Hide the progress bar. Defaults to False.

    Returns:
        np.ndarray: Cutout data
    """
    reason = section_fallback_reason(hdu)
    if reason is not None:
        logger.info("Reading one row at a time with astropy section: %s", reason)
        section: np.ndarray = hdu.section[slicer]
        return section

    plan = plan_download(hdu, slicer, max_request_kb, max_parallel_requests)
    for line in plan.describe():
        logger.info(line)

    tick = time()
    starts, ends = plan.request_starts, plan.request_ends
    blocks = cat_ranges_with_progress(
        fs,
        path,
        starts,
        ends,
        max_parallel_requests,
        f"Downloading {hdu.name}",
        disable_progress,
    )
    n_retried = 0
    parallel = max_parallel_requests
    for attempt in range(retries):
        failed = [i for i, b in enumerate(blocks) if isinstance(b, Exception)]
        if not failed:
            break
        n_retried += len(failed)
        # Busy (504) or rate-limiting (429) servers need a pause and fewer requests
        parallel = max(parallel // 2, 1)
        error = blocks[failed[0]]
        logger.warning(
            "Retrying %d of %d failed requests in %d s, %d at a time (first error: %s %s)",
            len(failed),
            len(starts),
            2**attempt,
            parallel,
            type(error).__name__,
            getattr(error, "status", ""),
        )
        sleep(2**attempt)
        retried = cat_ranges_with_progress(
            fs,
            path,
            [starts[i] for i in failed],
            [ends[i] for i in failed],
            parallel,
            f"Retrying {hdu.name}",
            disable_progress,
        )
        for i, block in zip(failed, retried, strict=True):
            blocks[i] = block
    for block, start, end in zip(blocks, starts, ends, strict=True):
        if isinstance(block, Exception):
            raise block
        # Some servers ignore the range and send the whole file
        if len(block) != end - start:
            msg = f"Expected {end - start} bytes from {path}, got {len(block)}"
            raise OSError(msg)
    took = time() - tick
    logger.info(
        "Downloaded %.1f MB in %.1f s (%.1f MB/s), %d retried",
        plan.requested_bytes / 1024**2,
        took,
        plan.requested_bytes / 1024**2 / took,
        n_retried,
    )

    # Pick each row out of the downloaded blocks
    dtype = hdu.section.dtype.newbyteorder(">")
    starts_arr = np.array(starts)
    block_offsets = np.cumsum([0, *(len(b) for b in blocks[:-1])])
    owner = np.searchsorted(starts_arr, plan.row_starts, side="right") - 1
    row_offsets = (
        block_offsets[owner] + plan.row_starts - starts_arr[owner]
    ) // dtype.itemsize
    flat = np.frombuffer(b"".join(blocks), dtype=dtype)
    row_len = plan.row_bytes // dtype.itemsize
    data: np.ndarray = flat[row_offsets[:, None] + np.arange(row_len)]
    out_shape = tuple(
        stop - start
        for start, stop, _ in (
            s.indices(n) for s, n in zip(slicer, hdu.shape, strict=True)
        )
    )
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
    dry_run: bool = False,
    disable_progress: bool = False,
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
        dry_run (bool, optional): Log the download plan without downloading or writing. Defaults to False.
        disable_progress (bool, optional): Hide progress bars. Defaults to False.
        **fetch_kwargs (int): Download settings passed to `fetch_section`.

    Returns:
        fits.HDUList: Cutout HDUList
    """
    fsspec_kwargs: dict[str, Any] = {}
    if infile.startswith("s3://"):
        logger.info("Attempting to access S3 file...")
        key = os.getenv("FSSPEC_S3_KEY")
        if key is None:
            logger.info("No S3 credentials - accessing anonymously")
            fsspec_kwargs["anon"] = True
        else:
            logger.info("Using S3 credentials...")
            fsspec_kwargs["key"] = key
            fsspec_kwargs["secret"] = os.getenv("FSSPEC_S3_SECRET")
        if endpoint_url := os.getenv("FSSPEC_S3_ENDPOINT_URL"):
            fsspec_kwargs["endpoint_url"] = endpoint_url
        # botocore allows only 10 connections by default
        fsspec_kwargs["config_kwargs"] = {"max_pool_connections": 256}

    fs, path = fsspec.core.url_to_fs(infile, **fsspec_kwargs)
    cutout_hdulist = fits.HDUList()
    with (
        # Mostly reads headers - fsspec's default 5 MB block is overkill
        fs.open(path, "rb", block_size=256 * 1024) as fobj,
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
            logger.info("Cutting HDU %d (%s)", ext, hdu.name)
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

            if dry_run:
                reason = section_fallback_reason(hdu)
                if reason is not None:
                    logger.info(
                        "Would read one row at a time with astropy section: %s", reason
                    )
                    continue
                plan_kwargs = {
                    k: v
                    for k, v in fetch_kwargs.items()
                    if k in ("max_request_kb", "max_parallel_requests")
                }
                for line in plan_download(hdu, slicer, **plan_kwargs).describe():
                    logger.info(line)
                continue

            # Download will start here
            tick = time()
            cutout_data = fetch_section(
                hdu, slicer, fs, path, disable_progress=disable_progress, **fetch_kwargs
            )
            tock = time()
            logger.info("Cutout obtained! Took %.2f seconds", tock - tick)
            # Make sure the header is updated to reflect the cutout
            logger.info("Updating header...")
            cutout_header = update_header(header, slicer)
            cutout_hdulist.append(type(hdu)(cutout_data, cutout_header))

        if dry_run:
            return cutout_hdulist
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


def get_cutout_parser(
    parent_parser: bool = False, max_request_kb: int = 64
) -> argparse.ArgumentParser:
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
    download = cutout_parser.add_argument_group("Download options")
    download.add_argument(
        "--max-request-kb",
        type=int,
        default=max_request_kb,
        help="Largest single request in kB. Rows less than a quarter of this apart "
        "are downloaded together (default %(default)s)",
    )
    download.add_argument(
        "--max-parallel-requests",
        type=int,
        default=None,
        help="Requests sent to the server at once (default 64)",
    )
    download.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print the download plan without downloading",
    )
    download.add_argument(
        "--disable-progress",
        action="store_true",
        help="Hide progress bars",
    )
    download.add_argument(
        "--quiet",
        action="store_true",
        help="Hide logged output and progress bars",
    )
    return cutout_parser


def get_fetch_kwargs(args: argparse.Namespace) -> dict[str, int]:
    """Download options set on the command line

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        dict[str, int]: Options to pass to `fetch_section`
    """
    names = ("max_request_kb", "max_parallel_requests")
    return {n: getattr(args, n) for n in names if getattr(args, n) is not None}


def get_verbosity(args: argparse.Namespace) -> int:
    """Verbosity from the command line, allowing for --quiet and --dry-run

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        int: Verbosity to pass to `set_verbosity`
    """
    if args.quiet:
        return -1
    return max(args.verbosity, 1) if args.dry_run else int(args.verbosity)


def main() -> None:
    file_parser = get_file_parser(parent_parser=True)
    cutout_parser = get_cutout_parser(parent_parser=True)
    parser = argparse.ArgumentParser(
        parents=[file_parser, cutout_parser],
        description=cutout_parser.description,
    )
    args = parser.parse_args()

    set_verbosity(verbosity=get_verbosity(args))

    _ = make_cutout(
        infile=args.infile,
        outfile=args.outfile,
        ra_deg=args.ra_deg,
        dec_deg=args.dec_deg,
        radius_arcmin=args.radius_arcmin,
        freq_start_hz=args.freq_start,
        freq_end_hz=args.freq_end,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        disable_progress=args.quiet or args.disable_progress,
        **get_fetch_kwargs(args),
    )


if __name__ == "__main__":
    main()
