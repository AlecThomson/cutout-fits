from __future__ import annotations

try:
    # Python < 3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from cutout_fits import cutout
from cutout_fits.logger import logger


def get_packaged_resource_path(package: str, filename: str) -> Path:
    """Load in the path of a package sources.

    The `package` argument is passed as a though the module
    is being specified as an import statement.

    Args:
        package (str): The module path to the resources
        filename (str): Filename of the datafile to load

    Returns:
        Path: The absolute path to the packaged resource file
    """
    logger.info(f"Loading {package=} for {filename=}")

    dir_path = Path(resources.files(package))
    return dir_path / filename


class SampleData(NamedTuple):
    """Sample data for testing"""

    n_ra_pix: int
    n_dec_pix: int
    n_freq: int
    n_pol: int
    n_time: int
    fov: float
    ra_center: float
    dec_center: float
    freq_start: float
    freq_stop: float
    data_path: Path


@pytest.fixture()
def sample_data(tmpdir) -> np.Generator[SampleData, np.Any, None]:
    n_ra_pix = 128
    n_dec_pix = 128
    n_freq = 8
    n_pol = 4
    n_time = 1

    fov = 60  # arcmin
    ra_center = 0  # degrees
    dec_center = 0  # degrees
    freq_start = 1  # GHz
    freq_stop = 2  # GHz

    shape = (n_ra_pix, n_dec_pix, n_freq, n_pol, n_time)
    data = np.zeros(shape[::-1], dtype=np.float32)
    header_string = f"""
SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    {len(shape)} / number of data axes
NAXIS1  =                  {shape[0]}
NAXIS2  =                  {shape[1]}
NAXIS3  =                  {shape[2]}
NAXIS4  =                  {shape[3]}
NAXIS5  =                  {shape[4]}
CTYPE1  = 'RA---SIN'           / Right ascension
CRVAL1  =                  {ra_center} / [deg] RA at CRPIX1
CDELT1  =                  {-fov/60 / n_ra_pix} / [deg] RA increment
CRPIX1  =                  {n_ra_pix / 2} / Reference pixel in RA
CTYPE2  = 'DEC--SIN'           / Declination
CRVAL2  =                  {dec_center} / [deg] DEC at CRPIX2
CDELT2  =                  {fov/60 / n_dec_pix} / [deg] DEC increment
CRPIX2  =                  {n_dec_pix / 2} / Reference pixel in DEC
CTYPE3  = 'FREQ    '           / Frequency
CRVAL3  =                  {freq_start} / [GHz] FREQ at CRPIX3
CDELT3  =                  {(freq_stop - freq_start) / n_freq} / [GHz] FREQ increment
CRPIX3  =                  1 / Reference pixel in FREQ
CTYPE4  = 'STOKES  '           / Stokes
CRVAL4  =                  1 / STOKES at CRPIX4
CDELT4  =                  1 / STOKES increment
CRPIX4  =                  1 / Reference pixel in STOKES
CTYPE5  = 'TIME    '           / Time
CRVAL5  =                  0 / TIME at CRPIX5
CDELT5  =                  1 / TIME increment
CRPIX5  =                  1 / Reference pixel in TIME
BUNIT   = 'Jy/beam '           / Units
BMAJ    =                  1 / [deg] Beam major axis
BMIN    =                  1 / [deg] Beam minor axis
BPA     =                  0 / [deg] Beam position angle
    """
    header = fits.Header.fromstring(header_string, sep="\n")
    hdu = fits.PrimaryHDU(data, header=header)
    tmp_path = Path(tmpdir) / "test.fits.gz"
    hdu.writeto(tmp_path, overwrite=True)
    yield SampleData(
        n_ra_pix=n_ra_pix,
        n_dec_pix=n_dec_pix,
        n_freq=n_freq,
        n_pol=n_pol,
        n_time=n_time,
        fov=fov,
        ra_center=ra_center,
        dec_center=dec_center,
        freq_start=freq_start,
        freq_stop=freq_stop,
        data_path=tmp_path,
    )
    tmp_path.unlink()


@pytest.fixture()
def center_coord(sample_data: SampleData) -> SkyCoord:
    header = fits.getheader(sample_data.data_path)
    wcs = WCS(header)
    return wcs.celestial.pixel_to_world(*wcs.celestial.wcs.crpix)


@pytest.fixture()
def temp_dir_path(tmpdir) -> Path:
    return Path(tmpdir)


def test_cutout_half(sample_data, center_coord, temp_dir_path):
    out_path = temp_dir_path / "temp.fits"
    cut_hdul = cutout.make_cutout(
        infile=sample_data.data_path.as_posix(),
        outfile=out_path.as_posix(),
        ra_deg=center_coord.ra.deg,
        dec_deg=center_coord.dec.deg,
        radius_arcmin=sample_data.fov / 4,
        overwrite=True,
    )
    out_path.unlink()
    # assert cut_hdu.data.shape ==
    assert cut_hdul is not None
    assert cut_hdul[0].data.shape == (
        sample_data.n_time,
        sample_data.n_pol,
        sample_data.n_freq,
        sample_data.n_dec_pix // 2,
        sample_data.n_ra_pix // 2,
    )


def test_cutout_oversize(sample_data, center_coord, temp_dir_path):
    out_path = temp_dir_path / "temp.fits"
    cut_hdul = cutout.make_cutout(
        infile=sample_data.data_path.as_posix(),
        outfile=out_path.as_posix(),
        ra_deg=center_coord.ra.deg,
        dec_deg=center_coord.dec.deg,
        radius_arcmin=180 * 60,
        overwrite=True,
    )
    out_path.unlink()
    # assert cut_hdu.data.shape ==
    assert cut_hdul is not None
    assert cut_hdul[0].data.shape == (
        sample_data.n_time,
        sample_data.n_pol,
        sample_data.n_freq,
        sample_data.n_dec_pix,
        sample_data.n_ra_pix,
    )


def test_slicer_to_shape(sample_data, center_coord, temp_dir_path):
    out_path = temp_dir_path / "temp.fits"
    cut_hdul = cutout.make_cutout(
        infile=sample_data.data_path.as_posix(),
        outfile=out_path.as_posix(),
        ra_deg=center_coord.ra.deg,
        dec_deg=center_coord.dec.deg,
        radius_arcmin=sample_data.fov * 2,
        overwrite=True,
    )
    out_path.unlink()
    # assert cut_hdu.data.shape ==
    assert cut_hdul is not None
    assert cut_hdul[0].data.shape == (
        sample_data.n_time,
        sample_data.n_pol,
        sample_data.n_freq,
        sample_data.n_dec_pix,
        sample_data.n_ra_pix,
    )

    wcs = WCS(cut_hdul[0].header)
    slicer = cutout.make_slicer(wcs, center_coord, sample_data.fov)

    test_shape = cutout.get_cutout_shape(wcs, slicer)

    assert test_shape == cut_hdul[0].data.shape
    shape_str = cutout.format_shape(wcs, test_shape)
    logger.critical("%s", shape_str)
    known_str = "{'TIME': 1, 'STOKES': 4, 'FREQ': 8, 'DEC--SIN': 128, 'RA---SIN': 128}"
    assert shape_str == known_str
