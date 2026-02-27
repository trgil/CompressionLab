"""
Codec registration and lookup utilities.

The CodecRegistry provides a controlled environment for managing
available codecs within CompressionLab.

Design Rationale
----------------

The registry decouples:

    • Codec implementations
    • Benchmark orchestration
    • CLI interfaces
    • Experiment configuration

Without a registry, benchmarking code would need to import and
instantiate codecs directly, leading to tight coupling and
manual maintenance.

With a registry:

    • Codecs register themselves once
    • Benchmarking code performs name-based lookup
    • New codecs can be added without modifying orchestration logic
    • Automated iteration over all registered codecs becomes trivial

This enables scalable experimentation.
"""

from __future__ import annotations
from typing import Dict

from .base import Codec


class CodecRegistry:
    """
    In-memory registry for codec instances.

    Responsibilities
    ----------------
    • Store codec instances indexed by unique name
    • Prevent duplicate registrations
    • Provide safe lookup by name
    • Expose available codec list

    Design Decisions
    ----------------
    • Registry stores instantiated codec objects.
    • Codec names are normalized to lowercase.
    • Duplicate names are disallowed.
    • Lookup failure produces informative error messages.

    This class does not:
        - Perform dynamic module discovery
        - Persist registry state
        - Enforce version compatibility

    Example
    -------
        registry = CodecRegistry()
        registry.register(IdentityCodec())

        codec = registry.get("identity")
        blob = codec.encode(request)
    """

    def __init__(self) -> None:
        self._codecs: Dict[str, Codec] = {}

    def register(self, codec: Codec) -> None:
        """
        Register a codec instance.
        """

        name = codec.name.strip().lower()
        if not name:
            raise ValueError("codec.name must be non-empty")
        if name in self._codecs:
            raise ValueError(f"Codec '{name}' already registered")
        self._codecs[name] = codec

    def get(self, name: str) -> Codec:
        """
        Retrieve a codec by name.
        """

        key = name.strip().lower()
        if key not in self._codecs:
            available = ", ".join(sorted(self._codecs.keys()))
            raise KeyError(f"Unknown codec '{name}'. Available: {available}")
        return self._codecs[key]

    def list(self) -> list[str]:
        """
        Return a sorted list of registered codec names.
        """
        return sorted(self._codecs.keys())
