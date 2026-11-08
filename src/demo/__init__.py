"""The demo's pinned snapshot (#51): one fixed index, model set and configuration.

``python -m src.demo prepare`` makes a machine ready, ``check`` says whether it
is, and ``start`` opens the app on it with nothing rebuilt or downloaded. See
``snapshot.py`` for what is pinned and why.
"""

from .snapshot import PIN, DemoPin, Layout, Report, check, prepare

__all__ = ["PIN", "DemoPin", "Layout", "Report", "check", "prepare"]
