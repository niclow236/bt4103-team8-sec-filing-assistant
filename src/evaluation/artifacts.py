"""Atomic JSON artifacts and one recovery rule for plain/gzipped JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(stream: TextIO, *, recover: bool = False) -> list[dict]:
    """Recover only completed lines before an unfinished row or gzip footer.

    Invalid complete JSON and non-EOF decompression errors always propagate.
    """
    rows = []
    try:
        for line in stream:
            if not line.endswith("\n"):
                if recover:
                    break
                raise EOFError("unfinished JSONL row")
            if line.strip():
                rows.append(json.loads(line))
    except EOFError:
        if not recover:
            raise
    return rows
