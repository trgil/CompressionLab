from __future__ import annotations
from typing import Dict

from .base import Codec


class CodecRegistry:
    """In-memory codec registry."""
    def __init__(self) -> None:
        self._codecs: Dict[str, Codec] = {}

    def register(self, codec: Codec) -> None:
        name = codec.name.strip().lower()
        if not name:
            raise ValueError("codec.name must be non-empty")
        if name in self._codecs:
            raise ValueError(f"Codec '{name}' already registered")
        self._codecs[name] = codec

    def get(self, name: str) -> Codec:
        key = name.strip().lower()
        if key not in self._codecs:
            available = ", ".join(sorted(self._codecs.keys()))
            raise KeyError(f"Unknown codec '{name}'. Available: {available}")
        return self._codecs[key]

    def list(self) -> list[str]:
        return sorted(self._codecs.keys())
