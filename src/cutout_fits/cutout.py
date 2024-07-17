#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from dotenv import load_dotenv

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


_ = load_dotenv()


class SpatialIndex(NamedTuple):
    start_ra_index: Optional[int]
    end_ra_index: Optional[int]
    start_dec_index: Optional[int]
    end_dec_index: Optional[int]


class SpectralIndex(NamedTuple):
    start_freq_index: Optional[int]
    end_freq_index: Optional[int]


class StokesIndex(NamedTuple):
    start_stokes_index: Optional[int]
    end_stokes_index: Optional[int]


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
    start_freq: Optional[u.Quantity] = None,
    end_freq: Optional[u.Quantity] = None,
) -> SpectralIndex:
    """Get the start and end indices for spectral axes

    Args:
        wcs (WCS): WCS for HDU
        start_freq (Optional[u.Quantity], optional): Starting frequency. Defaults to None.
        end_freq (Optional[u.Quantity], optional): End frequency. Defaults to None.

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
    return StokesIndex(
        start_stokes_index=None,
        end_stokes_index=None,
    )


def make_slicer(
    wcs: WCS,
    centre: SkyCoord,
    radius: u.Quantity,
    start_freq: Optional[u.Quantity] = None,
    end_freq: Optional[u.Quantity] = None,
) -> Tuple[slice]:
    """Create a slicer for a given WCS, centre, radius, and frequency range

    Args:
        wcs (WCS): WCS for HDU
        centre (SkyCoord): Centre of cutout
        radius (u.Quantity): Radius of cutout
        start_freq (Optional[u.Quantity], optional): Starting frequency. Defaults to None.
        end_freq (Optional[u.Quantity], optional): End frequnecy. Defaults to None.

    Returns:
        Tuple[slice]: Tuple of slices for each axis - in numpy order
    """
    array_ordering = tuple(wcs.wcs.ctype)[::-1]

    spatial_indices = get_spatial_indices(wcs, centre, radius)
    logger.debug(f"Spatial indices: {spatial_indices}")
    spectral_indices = get_spectral_indices(wcs, start_freq, end_freq)
    logger.debug(f"Spectral indices: {spectral_indices}")
    stokes_indices = get_stokes_indices(wcs)
    logger.debug(f"Stokes indices: {stokes_indices}")

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


def update_header(old_header: fits.Header, slicer: Tuple[slice]) -> fits.Header:
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


def make_cutout(
    filepath: str,
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    freq_start_hz: Optional[float] = None,
    freq_end_hz: Optional[float] = None,
) -> fits.HDUList:
    """Make a cutout of a FITS file

    Args:
        filepath (str): Path to FITS file - can be a remote URL
        ra_deg (float): Centre RA in degrees
        dec_deg (float): Centre Dec in degrees
        radius_arcmin (float): Cutout radius in arcminutes
        freq_start_hz (Optional[float], optional): Start frequency in Hz. Defaults to None.
        freq_end_hz (Optional[float], optional): End frequency in Hz. Defaults to None.

    Returns:
        fits.HDUList: Cutout HDUList
    """
    fsspec_kwargs = {
        "key": os.getenv("FSSPEC_S3_KEY"),
        "secret": os.getenv("FSSPEC_S3_SECRET"),
        "endpoint_url": os.getenv("FSSPEC_S3_ENDPOINT_URL"),
    }

    cutout_hdulist = fits.HDUList()
    with fits.open(
        filepath,
        fsspec_kwargs=fsspec_kwargs,
        use_fsspec=True,
        memmap=True,
        mode="denywrite",
        output_verify="silentfix",
    ) as hdul:
        for hdu in hdul:
            # Only cutout primary and image HDUs
            # All other HDUs are passed through
            if not isinstance(hdu, fits.PrimaryHDU) or not isinstance(
                hdu, fits.ImageHDU
            ):
                cutout_hdulist.append(hdu)
                continue

            # If a primary HDU or image HDU, make a cutout
            header = hdu.header
            wcs = WCS(header)
            slicer = make_slicer(
                wcs=wcs,
                coord=SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg),
                radius=radius_arcmin * u.arcmin,
                start_freq=freq_start_hz * u.Hz if freq_start_hz is not None else None,
                end_freq=freq_end_hz * u.Hz if freq_end_hz is not None else None,
            )
            logger.debug(f"Slicer: {slicer}")
            cutout_data = hdu.section[slicer]
            # Make sure the header is updated to reflect the cutout
            cutout_header = update_header(header, slicer)
            cutout_hdulist.append(type(hdu)(cutout_data, cutout_header))
    return cutout_hdulist


def main() -> None:
    pass


if __name__ == "__main__":
    main()
