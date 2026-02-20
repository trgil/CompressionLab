from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from image import RawImage, CompressedImage


@dataclass(frozen=True, slots=True)
class EncodeRequest:
    """
    Request object passed to codecs during encoding.

    `params` is codec-specific but must be JSON-serializable.
    """
    image: RawImage
    params: dict[str, Any]


@runtime_checkable
class Codec(Protocol):
    """
    Codec plugin interface.

    Implementations must be deterministic given the same image+params.
    """

    @property
    def name(self) -> str:
        """Unique codec name used in registry lookups (e.g. 'identity', 'png', 'jpeg')."""

    def encode(self, req: EncodeRequest) -> CompressedImage:
        """Encode a RawImage into a compressed blob plus JSON-serializable sidecar metadata."""

    def decode(self, blob: CompressedImage) -> RawImage:
        """Decode a CompressedImage back into a RawImage."""
