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
usage: cutout-fits [-h] [-o] [-v] [--freq-start FREQ_START] [--freq-end FREQ_END]
                   [--max-request-kb MAX_REQUEST_KB]
                   [--max-parallel-requests MAX_PARALLEL_REQUESTS] [-n] [--disable-progress]
                   [--quiet]
                   infile outfile ra_deg dec_deg radius_arcmin

Make a cutout of a FITS file

options:
  -h, --help            show this help message and exit

File options:
  infile                Path to input FITS file - can be a remote URL
  outfile               Path to output FITS file
  -o, --overwrite       Overwrite output file if it exists
  -v, --verbosity       Increase output verbosity

Cutout options:
  ra_deg                Centre RA in degrees
  dec_deg               Centre Dec in degrees
  radius_arcmin         Cutout radius in arcminutes
  --freq-start FREQ_START
                        Start frequency in Hz
  --freq-end FREQ_END   End frequency in Hz

Download options:
  --max-request-kb MAX_REQUEST_KB
                        Largest single request in kB. Rows less than a quarter of this apart are
                        downloaded together (default 64)
  --max-parallel-requests MAX_PARALLEL_REQUESTS
                        Requests sent to the server at once (default 64)
  -n, --dry-run         Print the download plan without downloading
  --disable-progress    Hide progress bars
  --quiet               Hide logged output and progress bars
```

### CASDA Access

Cutouts from the CSIRO ASKAP Science Data Archive
([CASDA](https://research.csiro.au/casda/)) can be made by invoking the
`cutout-casda` entry point. The same set of cutout arguments as `cutout-fits`
are supported

```
$ cutout-casda -h
usage: cutout-casda [-h] [--freq-start FREQ_START] [--freq-end FREQ_END]
                    [--max-request-kb MAX_REQUEST_KB]
                    [--max-parallel-requests MAX_PARALLEL_REQUESTS] [-n] [--disable-progress]
                    [--quiet] [--output OUTPUT] [--username USERNAME] [--store-password]
                    [--reenter-password] [-v]
                    ra_deg dec_deg radius_arcmin filename [filename ...]

Make a cutout of a FITS file on CASDA

options:
  -h, --help            show this help message and exit

Cutout options:
  ra_deg                Centre RA in degrees
  dec_deg               Centre Dec in degrees
  radius_arcmin         Cutout radius in arcminutes
  --freq-start FREQ_START
                        Start frequency in Hz
  --freq-end FREQ_END   End frequency in Hz

Download options:
  --max-request-kb MAX_REQUEST_KB
                        Largest single request in kB. Rows less than a quarter of this apart are
                        downloaded together (default 256)
  --max-parallel-requests MAX_PARALLEL_REQUESTS
                        Requests sent to the server at once (default 64)
  -n, --dry-run         Print the download plan without downloading
  --disable-progress    Hide progress bars
  --quiet               Hide logged output and progress bars

CASDA options:
  filename              FITS file name(s)
  --output OUTPUT       Output directory
  --username USERNAME   CASDA username
  --store-password      Store password in keyring
  --reenter-password    Re-enter password
  -v, --verbosity       Increase output verbosity
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

If `FSSPEC_S3_KEY` is not set, S3 files are accessed anonymously, which works
for public buckets.

This project support using a `.env` file to store these variables, if needed.
Simply set these variables in a `.env` files in your current working directory
or set them in your environment if you wish. Be careful not to commit them to
VCS.

### Download settings

When cutting out a remote file, `cutout-fits` only downloads the parts of the
file that it needs. FITS images are stored row by row, so each row of your
cutout sits in a different place in the file. Rather than fetching these rows
one at a time (as astropy's `section` does), `cutout-fits` asks the server for
many of them at once.

Each request to a server has some overhead, so if two rows of the cutout are
close together in the file, it is faster to grab both in one request, along with
the pixels in between, than to ask for them separately. Rows are joined like
this if they are less than a quarter of `--max-request-kb` apart (16 kB by
default, 64 kB for `cutout-casda`). `--max-parallel-requests` sets how many
requests are sent at once (64 by default).

To see what would be downloaded without downloading anything, use `--dry-run`.
With `-v`, the same information is printed before each download. For example,
for the HST image used in the astropy docs:

```
$ cutout-fits s3://stpubdata/hst/public/j8pu/j8pu0y010/j8pu0y010_drc.fits out.fits 150.42742 2.42132 0.5 -v
...
Cutout is 1391 rows of 5.4 kB, 11.1 kB apart in the file
Rows less than 64/4=16 kB apart are downloaded together
Plan: 348 requests, 64 at a time; 18.7 MB to download (7.4 MB cutout + 11.3 MB between rows)
Downloading SCI: 100%|██████████████████████| 348/348 [00:03<00:00, 113.38 requests/s]
Downloaded 18.7 MB in 3.1 s (6.0 MB/s), 0 retried
```

Here the rows are 11.1 kB apart, so they are joined into 348 requests, and 11.3
MB is downloaded on top of the 7.4 MB cutout. For a small cutout of a wide CASDA
cube the rows are much further apart (51.3 kB). With `cutout-fits` defaults each
row would be requested on its own, but `cutout-casda` joins them, as this was
about four times faster in our tests even though it downloads far more data than
the cutout.

Progress bars are shown by default, and are hidden automatically when output is
not going to a terminal (e.g. in batch jobs). `--disable-progress` turns them
off, and `--quiet` turns off both the progress bars and the logging. Failed
requests are retried up to four times, waiting longer and sending fewer requests
at once each time.

The defaults should work well in most cases. In our tests they worked well for
AWS S3. On a slow connection to CASDA, try `--max-request-kb 64` to download
only the cutout rows. If you see a lot of retries, the server is either limiting
how fast you can ask (error 429, as MAST does) or is too busy (error 503 or
504), so try a lower `--max-parallel-requests`. If most of the download is data
between rows and it is slow, try a smaller `--max-request-kb`; if you are making
lots of small requests to a distant server, try a larger one.
