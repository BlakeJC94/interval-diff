import argparse
import sys

from .benchmark import benchmark


def parse_cli_input():
    parser = argparse.ArgumentParser("interval-diff", add_help=True)
    parser.add_argument(
        "--n-intervals",
        "-n",
        nargs="*",
        type=int,
        help="number of intervals to run algorithms in each sample.",
    )
    parser.add_argument(
        "--n-samples", "-k", nargs="?", type=int, help="number of random samples to run algorithms."
    )

    return parser.parse_args()


def main():
    benchmark(**vars(parse_cli_input()))


if __name__ == "__main__":
    sys.exit(main())
