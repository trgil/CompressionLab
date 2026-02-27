"""
Core codec interface definitions for CompressionLab.

This module defines the formal contract that all compression codecs
must follow in order to integrate with the CompressionLab framework.

Design Philosophy
-----------------

A codec in CompressionLab is defined as a deterministic transformation:

    RawImage  --encode-->  CompressedImage
    CompressedImage  --decode-->  RawImage

The framework does not assume:

    - Any particular compression algorithm
    - Any file format
    - Any internal representation
    - Any lossless or lossy behavior

Instead, it enforces a minimal interface to guarantee interoperability.

All codecs must:

    • Provide a unique string identifier (`name`)
    • Implement `encode()`
    • Implement `decode()`
    • Produce valid `CompressedImage` objects
    • Be deterministic given identical inputs and parameters

This abstraction allows the benchmarking pipeline to treat all codecs
uniformly, enabling automated comparison, parameter sweeps, and
rate–distortion analysis without special-case logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from image import RawImage, CompressedImage


@dataclass(frozen=True, slots=True)
class EncodeRequest:
    """
    Container for encoding parameters.

    Attributes
    ----------
    image : RawImage
        The source image to be encoded.

    params : dict[str, Any]
        Codec-specific encoding parameters.

        This dictionary must be JSON-serializable, as it may later
        be stored as part of experiment metadata.

    Notes
    -----
    The framework does not interpret `params`. Each codec is responsible
    for validating and using the parameters it supports.

    Example
    -------
        EncodeRequest(
            image=raw,
            params={"quality": 90}
        )
    """
    image: RawImage
    params: dict[str, Any]


@runtime_checkable
class Codec(Protocol):
    """
    Formal codec interface for CompressionLab.

    Any class implementing this protocol is considered a valid codec.

    Required Properties
    -------------------
    name : str
        A unique, lowercase identifier for the codec.
        Used for registry lookup and experiment reporting.

    Required Methods
    ----------------
    encode(req: EncodeRequest) -> CompressedImage
        Compress a RawImage into an opaque payload plus
        JSON-serializable sidecar metadata.

    decode(blob: CompressedImage) -> RawImage
        Reconstruct a RawImage from the compressed representation.

    Behavioral Requirements
    ------------------------
    • Deterministic behavior given identical input and parameters.
    • No modification of input RawImage.
    • Returned CompressedImage must satisfy framework invariants.
    • decode(encode(image)) must return a valid RawImage.
    • Lossless codecs must produce bit-exact reconstruction.

    Architectural Notes
    --------------------
    The protocol is intentionally minimal. The framework does not impose:

        - State management
        - Versioning
        - Capability flags
        - Performance guarantees

    These may be layered on later without breaking the core contract.
    """

    @property
    def name(self) -> str:
        """Unique codec name used in registry lookups (e.g. 'identity', 'png', 'jpeg')."""

    def encode(self, req: EncodeRequest) -> CompressedImage:
        """Encode a RawImage into a compressed blob plus JSON-serializable sidecar metadata."""

    def decode(self, blob: CompressedImage) -> RawImage:
        """Decode a CompressedImage back into a RawImage."""
