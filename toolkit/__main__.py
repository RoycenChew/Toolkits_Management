"""`python -m toolkit` -> the CLI.

A separate module from `cli.py` so the CLI can be imported and tested without
executing anything, and so `python -m toolkit` works without installing a
console script.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
