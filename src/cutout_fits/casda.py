from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroquery.casda import Casda
from astroquery.casda.core import CasdaClass
from astroquery.utils.tap.core import TapPlus

from cutout_fits import make_cutout
from cutout_fits.logger import logger, set_verbosity


async def get_staging_url(file_name: str) -> pd.DataFrame:
    tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
    query_str = f"SELECT access_url, filename FROM ivoa.obscore WHERE filename='{file_name}' AND dataproduct_type='cube'"
    logger.info(f"Querying CASDA for {file_name}")
    logger.debug(f"Query: {query_str}")
    job = await asyncio.to_thread(tap.launch_job_async, query_str)
    results = job.get_results()

    if results is None:
        msg = "No results found!"
        raise ValueError(msg)

    return results


async def get_download_url(result_table: Table, casda: CasdaClass) -> str:
    assert len(result_table) == 1, "Multiple files found!"
    logger.info("Staging data on CASDA...")
    url_list: list[str] = await asyncio.to_thread(casda.stage_data, result_table)

    url_fits_list = []
    for url in url_list:
        if url.endswith("checksum"):
            continue
        url_fits_list.append(url)

    if len(url_fits_list) == 0:
        msg = "No fits file found!"
        raise ValueError(msg)
    if len(url_fits_list) > 1:
        msg = "Multiple fits files found!"
        raise ValueError(msg)

    url = url_fits_list[0]
    msg = f"Staged data at {url}"
    logger.info(msg)
    return url


def casda_login(
    username: str | None = None,
    store_password: bool = False,
    reenter_password: bool = False,
) -> CasdaClass:
    """Login to CASDA.

    Args:
        username (str | None, optional): CASDA username. Defaults to None.
        store_password (bool, optional): Stores the password securely in your keyring. Defaults to False.
        reenter_password (bool, optional): Asks for the password even if it is already stored in the keyring. This is the way to overwrite an already stored passwork on the keyring. Defaults to False.

    Returns:
        CasdaClass: _description_
    """
    casda: CasdaClass = Casda()
    if username is None:
        username = os.environ.get("CASDA_USERNAME")
    if username is None:
        username = input("Please enter your CASDA username: ")

    casda.login(
        username=username,
        store_password=store_password,
        reenter_password=reenter_password,
    )

    return casda


async def cutout_from_casda(
    casda: CasdaClass,
    file_name: str,
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    output_dir: Path,
    freq_start_hz: float | None = None,
    freq_end_hz: float | None = None,
) -> fits.HDUList:
    result_table: pd.DataFrame = await get_staging_url(file_name)
    url = await get_download_url(result_table, casda)
    outfile = output_dir / file_name.replace(".fits", ".cutout.fits")
    return await asyncio.to_thread(
        make_cutout,
        infile=url,
        outfile=outfile.as_posix(),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcmin=radius_arcmin,
        freq_start_hz=freq_start_hz,
        freq_end_hz=freq_end_hz,
        overwrite=True,
    )


async def get_cutouts_from_casda(
    file_name_list: list[str],
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    output_dir: Path | None = None,
    username: str | None = None,
    freq_start_hz: float | None = None,
    freq_end_hz: float | None = None,
    store_password: bool = False,
    reenter_password: bool = False,
) -> list[fits.HDUList]:
    casda = casda_login(
        username=username,
        store_password=store_password,
        reenter_password=reenter_password,
    )

    if output_dir is None:
        output_dir = Path.cwd()

    tasks = []
    for file_name in file_name_list:
        task = cutout_from_casda(
            casda=casda,
            file_name=file_name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_arcmin=radius_arcmin,
            output_dir=output_dir,
            freq_start_hz=freq_start_hz,
            freq_end_hz=freq_end_hz,
        )
        tasks.append(task)

    return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a cutout of a FITS file on CASDA"
    )
    parser.add_argument("filename", nargs="+", help="FITS file name(s)")
    parser.add_argument("ra_deg", type=float, help="Centre RA in degrees")
    parser.add_argument("dec_deg", type=float, help="Centre Dec in degrees")
    parser.add_argument("radius_arcmin", type=float, help="Cutout radius in arcminutes")
    parser.add_argument(
        "--output", help="Directory to save FITS cutouts", type=Path, default=None
    )
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
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    parser.add_argument(
        "--username",
        help="CASDA username",
        default=None,
    )
    parser.add_argument(
        "--store-password",
        help="Store password in keyring",
        action="store_true",
    )
    parser.add_argument(
        "--reenter-password",
        help="Re-enter password",
        action="store_true",
    )
    args = parser.parse_args()

    set_verbosity(
        verbosity=args.verbosity,
    )

    file_names = args.filename
    if isinstance(file_names, str):
        file_names = [file_names]

    _ = asyncio.run(
        get_cutouts_from_casda(
            file_name_list=file_names,
            output_dir=args.output,
            ra_deg=args.ra_deg,
            dec_deg=args.dec_deg,
            radius_arcmin=args.radius_arcmin,
            freq_start_hz=args.freq_start,
            freq_end_hz=args.freq_end,
            username=args.username,
            store_password=args.store_password,
            reenter_password=args.reenter_password,
        )
    )


if __name__ == "__main__":
    main()
