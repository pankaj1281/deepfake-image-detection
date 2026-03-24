"""
main.py — Unified CLI entry point for the Fake Image Detection project.

Commands:
    train    — Train a model on the dataset.
    predict  — Run prediction on a single image or a directory.

Run `python main.py <command> --help` for command-specific options.
"""

import sys
import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Fake Image Detection — Unified CLI",
    )
    parser.add_argument(
        "command",
        choices=["train", "predict"],
        help="Sub-command to run.",
    )

    # Parse only the first positional argument to determine the sub-command
    args, remaining = parser.parse_known_args()

    if args.command == "train":
        subprocess.run(
            [sys.executable, "train.py"] + remaining,
            check=True,
        )
    elif args.command == "predict":
        subprocess.run(
            [sys.executable, "predict.py"] + remaining,
            check=True,
        )


if __name__ == "__main__":
    main()
