# cutout-fits

[![Actions Status][actions-badge]][actions-link]
[![Codecov Status][codecov-badge]][codecov-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[codecov-link]:             https://codecov.io/gh/AlecThomson/cutout-fits
[codecov-badge]:            https://codecov.io/gh/AlecThomson/cutout-fits/graph/badge.svg?token=7EARBRN20D
[actions-badge]:            https://github.com/AlecThomson/cutout-fits/workflows/CI/badge.svg
[actions-link]:             https://github.com/AlecThomson/cutout-fits/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/cutout-fits
[conda-link]:               https://github.com/conda-forge/cutout-fits-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/AlecThomson/cutout-fits/discussions
[pypi-link]:                https://pypi.org/project/cutout-fits/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/cutout-fits
[pypi-version]:             https://img.shields.io/pypi/v/cutout-fits
[rtd-badge]:                https://readthedocs.org/projects/cutout-fits/badge/?version=latest
[rtd-link]:                 https://cutout-fits.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

A utility to produce cutouts of FITS images using `astropy`. Remotely-hosted
FITS images are fully supported using `ffspec` and `s3fs`.

## Installation

From PyPI (stable):

```
pip install cutout-fits
```

From git (latest):

```
pip install git+https://github.com/AlecThomson/cutout-fits.git
```

## Usage

### Universal cutouts

The command-line tool can be invoked using `cutout-fits` entry point. Currently,
spatial cutouts are specified using a centre right ascension and declination
along with a cutout radius. Spectral cutouts are specified with a start and end
frequency range.

Any additional cube dimensions, such as time or Stokes, will simply be included
in the cutout. Further, any non-image HDUs present in the FITS file will also be
simply included in the output file.

```
$ cutout-fits -h
usage: cutout-fits [-h] [--freq-start FREQ_START] [--freq-end FREQ_END] [-o] [-v] infile outfile ra_deg dec_deg radius_arcmin

Make a cutout of a FITS file

positional arguments:
  infile                Path to input FITS file - can be a remote URL
  outfile               Path to output FITS file
  ra_deg                Centre RA in degrees
  dec_deg               Centre Dec in degrees
  radius_arcmin         Cutout radius in arcminutes

options:
  -h, --help            show this help message and exit
  --freq-start FREQ_START
                        Start frequency in Hz
  --freq-end FREQ_END   End frequency in Hz
  -o, --overwrite       Overwrite output file if it exists
  -v, --verbosity       Increase output verbosity
```

### CASDA Access

Cutouts from the CSIRO ASKAP Science Data Archive
([CASDA](https://research.csiro.au/casda/)) can be made by invoking the
`cutout-casda` entry point. The same set of cutout arguments as `cutout-fits`
are supported

```
$ cutout-casda -h
usage: cutout-casda [-h] [--output OUTPUT] [--freq-start FREQ_START] [--freq-end FREQ_END] [-v] [--username USERNAME] [--store-password] [--reenter-password]
                    filename [filename ...] ra_deg dec_deg radius_arcmin

Make a cutout of a FITS file on CASDA

positional arguments:
  filename              FITS file name(s)
  ra_deg                Centre RA in degrees
  dec_deg               Centre Dec in degrees
  radius_arcmin         Cutout radius in arcminutes

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Directory to save FITS cutouts
  --freq-start FREQ_START
                        Start frequency in Hz
  --freq-end FREQ_END   End frequency in Hz
  -v, --verbosity       Increase output verbosity
  --username USERNAME   CASDA username
  --store-password      Store password in keyring
  --reenter-password    Re-enter password
```

The
[CASDA Astroquery API](https://astroquery.readthedocs.io/en/latest/casda/casda.html)
requires an [OPAL](https://opal.atnf.csiro.au/) username and password to login.
You can supply your username via the CLI or as the `CASDA_USERNAME` environment
variable. Astroquery supports caching your password, which you will need if
running non-interactively. To cache your password you can interactively run the
helper script `casda-login`:

```
$ casda-login -h
usage: casda-login [-h] username

Login to CASDA and save credentials

positional arguments:
  username    Username for CASDA

options:
  -h, --help  show this help message and exit
```

### Python API

Further API documentation is provided on [read the docs][rtd-link].

## Remote files

If accessing a remote file on S3, you'll need to set your access keys. To do
this, `fsspec` looks for the following environment variables:

```
FSSPEC_S3_ENDPOINT_URL='https://...'
FSSPEC_S3_KEY='...'
FSSPEC_S3_SECRET='...'
```

This project support using a `.env` file to store these variables, if needed.
Simply set these variables in a `.env` files in your current working directory
or set them in your environment if you wish. Be careful not to commit them to
VCS.
