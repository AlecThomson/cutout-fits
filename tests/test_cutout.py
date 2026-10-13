from __future__ import annotations

import logging

try:
    # Python < 3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources
from pathlib import Path
from typing import NamedTuple

import fsspec.core
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from fsspec.implementations.local import LocalFileSystem

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


@pytest.fixture
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
CDELT1  =                  {-fov / 60 / n_ra_pix} / [deg] RA increment
CRPIX1  =                  {n_ra_pix / 2} / Reference pixel in RA
CTYPE2  = 'DEC--SIN'           / Declination
CRVAL2  =                  {dec_center} / [deg] DEC at CRPIX2
CDELT2  =                  {fov / 60 / n_dec_pix} / [deg] DEC increment
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


@pytest.fixture
def center_coord(sample_data: SampleData) -> SkyCoord:
    header = fits.getheader(sample_data.data_path)
    wcs = WCS(header)
    return wcs.celestial.pixel_to_world(*wcs.celestial.wcs.crpix)


@pytest.fixture
def temp_dir_path(tmpdir) -> Path:
    return Path(tmpdir)


@pytest.fixture
def beamtable_data(sample_data: SampleData, tmpdir) -> np.Generator[Path, np.Any, None]:
    tmp_path = Path(tmpdir) / "beamtable_test.fits"

    with fits.open(sample_data.data_path) as hdul:
        image_hdu = hdul[0].copy()
    image_hdu.header["CASAMBM"] = True

    beams_table = Table(
        {
            "CHAN": np.arange(sample_data.n_freq, dtype=np.int32),
            "BMAJ": np.full(sample_data.n_freq, 1.0, dtype=np.float32),
            "BMIN": np.full(sample_data.n_freq, 1.0, dtype=np.float32),
            "BPA": np.zeros(sample_data.n_freq, dtype=np.float32),
        }
    )
    beams_hdu = fits.BinTableHDU(beams_table, name="BEAMS")
    beams_hdu.header["NCHAN"] = sample_data.n_freq

    fits.HDUList([image_hdu, beams_hdu]).writeto(tmp_path, overwrite=True)
    yield tmp_path
    tmp_path.unlink()


@pytest.fixture
def no_beamtable_data(
    sample_data: SampleData, tmpdir
) -> np.Generator[Path, np.Any, None]:
    tmp_path = Path(tmpdir) / "no_beamtable_test.fits"

    with fits.open(sample_data.data_path) as hdul:
        image_hdu = hdul[0].copy()
    image_hdu.header["CASAMBM"] = True

    fits.HDUList([image_hdu]).writeto(tmp_path, overwrite=True)
    yield tmp_path
    tmp_path.unlink()


@pytest.fixture
def beamtable_without_requirement_data(
    sample_data: SampleData, tmpdir
) -> np.Generator[Path, np.Any, None]:
    tmp_path = Path(tmpdir) / "beamtable_without_requirement_test.fits"

    with fits.open(sample_data.data_path) as hdul:
        image_hdu = hdul[0].copy()
    if "CASAMBM" in image_hdu.header:
        del image_hdu.header["CASAMBM"]

    beams_table = Table(
        {
            "CHAN": np.arange(sample_data.n_freq, dtype=np.int32),
            "BMAJ": np.full(sample_data.n_freq, 1.0, dtype=np.float32),
            "BMIN": np.full(sample_data.n_freq, 1.0, dtype=np.float32),
            "BPA": np.zeros(sample_data.n_freq, dtype=np.float32),
        }
    )
    beams_hdu = fits.BinTableHDU(beams_table, name="BEAMS")
    beams_hdu.header["NCHAN"] = sample_data.n_freq

    fits.HDUList([image_hdu, beams_hdu]).writeto(tmp_path, overwrite=True)
    yield tmp_path
    tmp_path.unlink()


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


def test_cutout_with_single_beamtable_extension(
    sample_data: SampleData,
    center_coord: SkyCoord,
    temp_dir_path: Path,
    beamtable_data: Path,
):
    out_path = temp_dir_path / "beamtable_out.fits"
    cut_hdul = cutout.make_cutout(
        infile=beamtable_data.as_posix(),
        outfile=out_path.as_posix(),
        ra_deg=center_coord.ra.deg,
        dec_deg=center_coord.dec.deg,
        radius_arcmin=sample_data.fov,
        overwrite=True,
    )

    out_path.unlink()
    assert cut_hdul is not None
    assert len(cut_hdul) == 2
    assert cut_hdul[1].name == "BEAMS"
    assert cut_hdul[1].header["NCHAN"] == sample_data.n_freq


def test_cutout_raises_when_beamtable_required_but_missing(
    center_coord: SkyCoord,
    temp_dir_path: Path,
    no_beamtable_data: Path,
):
    out_path = temp_dir_path / "no_beamtable_out.fits"

    with pytest.raises(
        ValueError,
        match="Beam table required in header, but no beam table extension found!",
    ):
        cutout.make_cutout(
            infile=no_beamtable_data.as_posix(),
            outfile=out_path.as_posix(),
            ra_deg=center_coord.ra.deg,
            dec_deg=center_coord.dec.deg,
            radius_arcmin=60,
            overwrite=True,
        )


def test_cutout_raises_when_beamtable_present_but_not_required(
    center_coord: SkyCoord,
    temp_dir_path: Path,
    beamtable_without_requirement_data: Path,
):
    out_path = temp_dir_path / "beamtable_without_requirement_out.fits"

    with pytest.raises(
        ValueError,
        match="Beam table extension found, but no beam table required in any header!",
    ):
        cutout.make_cutout(
            infile=beamtable_without_requirement_data.as_posix(),
            outfile=out_path.as_posix(),
            ra_deg=center_coord.ra.deg,
            dec_deg=center_coord.dec.deg,
            radius_arcmin=60,
            overwrite=True,
        )


@pytest.fixture
def plain_data_path(sample_data: SampleData, tmpdir) -> Path:
    """Uncompressed copy with random data - fetch_section skips compressed files"""
    tmp_path = Path(tmpdir) / "plain.fits"
    header = fits.getheader(sample_data.data_path)
    rng = np.random.default_rng(0)
    data = rng.standard_normal(
        (
            sample_data.n_time,
            sample_data.n_pol,
            sample_data.n_freq,
            sample_data.n_dec_pix,
            sample_data.n_ra_pix,
        )
    ).astype(np.float32)
    fits.PrimaryHDU(data, header=header).writeto(tmp_path)
    return tmp_path


SLICERS = [
    (slice(None), slice(None), slice(None), slice(10, 20), slice(30, 50)),
    (slice(0, 1), slice(1, 3), slice(2, 3), slice(0, 128), slice(0, 128)),
    (slice(None), slice(0, 1), slice(None), slice(120, 128), slice(0, 5)),
    (slice(None), slice(None), slice(None), slice(None), slice(None)),
]


@pytest.mark.parametrize("slicer", SLICERS)
@pytest.mark.parametrize("max_request_kb", [1, 1024])
def test_fetch_section_matches_section(plain_data_path, slicer, max_request_kb):
    fs, path = fsspec.core.url_to_fs(plain_data_path.as_posix())
    with fits.open(plain_data_path) as hdul:
        hdu = hdul[0]
        expected = hdu.section[slicer]
        got = cutout.fetch_section(
            hdu,
            slicer,
            fs,
            path,
            max_request_kb=max_request_kb,
        )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float64])
def test_fetch_section_dtypes_and_extension(tmpdir, dtype):
    tmp_path = Path(tmpdir) / "ext.fits"
    data = np.arange(6 * 7 * 8).reshape(6, 7, 8).astype(dtype)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)]).writeto(tmp_path)
    slicer = (slice(1, 4), slice(2, 5), slice(3, 7))
    fs, path = fsspec.core.url_to_fs(tmp_path.as_posix())
    with fits.open(tmp_path) as hdul:
        got = cutout.fetch_section(hdul[1], slicer, fs, path)
    np.testing.assert_array_equal(got, data[slicer])
    assert got.dtype == data.dtype


def test_cutout_at_edge(plain_data_path, temp_dir_path):
    with fits.open(plain_data_path) as hdul:
        data = hdul[0].data.copy()
        wcs_in = WCS(hdul[0].header)
    corner = wcs_in.celestial.pixel_to_world(0, 0)
    cut_hdul = cutout.make_cutout(
        infile=plain_data_path.as_posix(),
        outfile=(temp_dir_path / "edge.fits").as_posix(),
        ra_deg=corner.ra.deg,
        dec_deg=corner.dec.deg,
        radius_arcmin=5,
    )
    cut = cut_hdul[0]
    n_dec, n_ra = cut.data.shape[-2:]
    assert n_dec > 0
    assert n_ra > 0
    np.testing.assert_array_equal(cut.data, data[..., :n_dec, :n_ra])
    wcs_out = WCS(cut.header).celestial
    assert wcs_out.pixel_to_world(0, 0).separation(corner).arcsec < 1e-6


@pytest.mark.parametrize(("crval", "cdelt"), [(1.0, 0.125), (1.875, -0.125)])
def test_spectral_range_inclusive(
    plain_data_path, center_coord, temp_dir_path, crval, cdelt
):
    with fits.open(plain_data_path, mode="update") as hdul:
        hdul[0].header["CRVAL3"] = crval
        hdul[0].header["CDELT3"] = cdelt
    cut_hdul = cutout.make_cutout(
        infile=plain_data_path.as_posix(),
        outfile=(temp_dir_path / "spec.fits").as_posix(),
        ra_deg=center_coord.ra.deg,
        dec_deg=center_coord.dec.deg,
        radius_arcmin=5,
        freq_start_hz=1.125,
        freq_end_hz=1.5,
    )
    spec_wcs = WCS(cut_hdul[0].header).spectral
    freqs = spec_wcs.pixel_to_world(np.arange(spec_wcs.pixel_shape[0])).to_value("Hz")
    np.testing.assert_allclose(sorted(freqs), [1.125, 1.25, 1.375, 1.5])


def test_cutout_outside_image_raises(plain_data_path, temp_dir_path):
    with pytest.raises(ValueError, match="does not overlap"):
        cutout.make_cutout(
            infile=plain_data_path.as_posix(),
            outfile=(temp_dir_path / "outside.fits").as_posix(),
            ra_deg=10,
            dec_deg=0,
            radius_arcmin=5,
        )


def test_fetch_section_retries_failed_ranges(plain_data_path):
    class FlakyFS(LocalFileSystem):  # type: ignore[misc]
        calls = 0

        def cat_ranges(self, *args, **kwargs):
            out = super().cat_ranges(*args, **kwargs)
            FlakyFS.calls += 1
            if FlakyFS.calls == 1:
                out[0] = OSError("dropped")
            return out

    fs = FlakyFS(skip_instance_cache=True)
    slicer = SLICERS[0]
    with fits.open(plain_data_path) as hdul:
        expected = hdul[0].section[slicer]
        got = cutout.fetch_section(
            hdul[0],
            slicer,
            fs,
            plain_data_path.as_posix(),
            max_request_kb=1,
        )
    assert FlakyFS.calls == 2
    np.testing.assert_array_equal(got, expected)


def test_cutout_2d_with_empty_primary(temp_dir_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    wcs.wcs.crval = [10, -30]
    data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    in_path = temp_dir_path / "img2d.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, wcs.to_header())]).writeto(
        in_path
    )
    cut_hdul = cutout.make_cutout(
        infile=in_path.as_posix(),
        outfile=(temp_dir_path / "img2d_out.fits").as_posix(),
        ra_deg=10,
        dec_deg=-30,
        radius_arcmin=0.5,
        max_request_kb=1,
    )
    assert cut_hdul[0].data is None
    assert 0 < cut_hdul[1].data.shape[0] < 100


def test_plan_download(plain_data_path):
    # 128-pixel float32 rows; a 10-pixel cutout leaves 472 unwanted bytes between rows
    slicer = (slice(None), slice(None), slice(None), slice(10, 20), slice(30, 40))
    with fits.open(plain_data_path) as hdul:
        separate = cutout.plan_download(hdul[0], slicer, max_request_kb=1)
        joined = cutout.plan_download(hdul[0], slicer)

    assert separate.row_spacing_bytes == 472
    assert separate.n_requests == separate.n_rows
    assert separate.unwanted_bytes == 0
    assert "1/4=0.25 kB apart" in separate.describe()[1]

    assert joined.n_requests < joined.n_rows
    assert joined.wanted_bytes == joined.n_rows * 40
    assert joined.unwanted_bytes == joined.requested_bytes - joined.wanted_bytes > 0
    assert all(
        end - start <= 64 * 1024
        for start, end in zip(joined.request_starts, joined.request_ends, strict=True)
    )
    assert "64/4=16 kB apart" in joined.describe()[1]


def test_dry_run(plain_data_path, center_coord, temp_dir_path, caplog):
    caplog.set_level(logging.INFO, logger="cutout_fits")
    out_path = temp_dir_path / "dry.fits"
    cutout.make_cutout(
        infile=plain_data_path.as_posix(),
        outfile=out_path.as_posix(),
        ra_deg=center_coord.ra.deg,
        dec_deg=center_coord.dec.deg,
        radius_arcmin=5,
        dry_run=True,
    )
    assert not out_path.exists()
    assert any(r.getMessage().startswith("Plan: ") for r in caplog.records)


@pytest.mark.parametrize("max_request_kb", [1, 1024])
def test_fetch_section_async_with_progress(plain_data_path, max_request_kb):
    asyn_wrapper = pytest.importorskip("fsspec.implementations.asyn_wrapper")
    fs = asyn_wrapper.AsyncFileSystemWrapper(LocalFileSystem())
    slicer = SLICERS[0]
    with fits.open(plain_data_path) as hdul:
        expected = hdul[0].section[slicer]
        got = cutout.fetch_section(
            hdul[0],
            slicer,
            fs,
            plain_data_path.as_posix(),
            max_request_kb=max_request_kb,
        )
    np.testing.assert_array_equal(got, expected)
