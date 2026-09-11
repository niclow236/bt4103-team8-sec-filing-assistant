"""Entry point for ``python -m src.retrieval``.

Python requires this module for a package to be runnable, and that is all it
is for. Every command, its arguments, and what it runs live in ``cli.py``, so
that there is one place to look for the command line rather than a table here
and the parsers there.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
