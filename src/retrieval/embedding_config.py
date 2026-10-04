"""The encoder settings passed together through index building and retrieval."""

from dataclasses import dataclass
from pathlib import Path

from .constants import CHROMA_DIR, EMBED_DIMENSIONS, EMBED_MODEL, PASSAGE_PREFIX, QUERY_PREFIX


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = EMBED_MODEL
    dimensions: int = EMBED_DIMENSIONS
    index_dir: Path = CHROMA_DIR
    query_prefix: str = QUERY_PREFIX
    passage_prefix: str = PASSAGE_PREFIX
    max_tokens: int | None = None
