"""PardaDB entry point: `python main.py` is the same as the installed `pardadb` command."""

import sys

from pardadb.cli import main

if __name__ == "__main__":
    sys.exit(main())
