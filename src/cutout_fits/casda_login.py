"""Helper to login to CASDA and save credentials"""

from __future__ import annotations

import argparse

from cutout_fits.casda import casda_login


def main() -> None:
    parser = argparse.ArgumentParser(description="Login to CASDA and save credentials")
    parser.add_argument("username", help="Username for CASDA")

    args = parser.parse_args()

    _ = casda_login(username=args.username, store_password=True, reenter_password=True)


if __name__ == "__main__":
    main()
