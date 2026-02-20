"""
compressionlab.image
====================

Core image container primitives used throughout CompressionLab.

This module defines two immutable data containers:

    - RawImage
    - CompressedImage

These classes are deliberately minimal and codec-agnostic. They do not
perform encoding, decoding, benchmarking, or file I/O. Their sole
responsibility is to provide validated, immutable containers for:

    • Uncompressed RGB image data (RawImage)
    • Compressed opaque payloads with codec-specific metadata (CompressedImage)

Design Principles
-----------------

1. Immutability
   Both classes are frozen dataclasses. Structural fields cannot be
   reassigned after construction. This guarantees experiment integrity.

2. Strict Invariants
   Objects are validated at creation time to prevent invalid internal state.

3. Codec-Agnosticism
   No assumptions are made about how a codec operates internally.
   The container only stores data — behavior belongs to codecs.

4. Deterministic Size Accounting
   CompressedImage supports canonical, deterministic size calculation,
   including sidecar metadata.

These primitives form the foundation of the CompressionLab benchmarking
pipeline and are designed to be stable, safe, and reproducible.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import json


@dataclass(slots=True, frozen=True)
class RawImage:
    """
    Immutable container for uncompressed RGB image data.

    RawImage represents a validated, canonical in-memory RGB image.
    It is intended to serve as the standard input and output format
    for all codecs within CompressionLab.

    Invariants
    ----------
    - pixels must be a numpy ndarray
    - dtype must be uint8
    - shape must be (H, W, 3)
    - image_name must be a non-empty string
    - pixels are deep-copied during initialization
    - internal pixel array is contiguous (C-order)

    Immutability
    ------------
    The class is frozen. Attribute reassignment is not allowed.
    However, numpy array contents remain mutable by design.

    Parameters
    ----------
    pixels : np.ndarray
        RGB image data with shape (H, W, 3) and dtype uint8.

    image_name : str
        Unique experiment identifier for the image. This value
        must be non-empty and is treated as immutable.

    Properties
    ----------
    height : int
        Image height in pixels.

    width : int
        Image width in pixels.

    shape : tuple[int, int, int]
        Image shape (H, W, 3).

    nbytes : int
        Total memory footprint of the pixel buffer.

    Notes
    -----
    RawImage performs deep copying of input data to ensure that
    upstream modifications do not affect internal state.

    This class does not perform:
        - file loading
        - color space conversion
        - resizing or preprocessing
        - benchmarking

    Those responsibilities belong elsewhere in the pipeline.
    """

    pixels: np.ndarray
    image_name: str

    def __post_init__(self):
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("image_name must be a non-empty string")

        if not isinstance(self.pixels, np.ndarray):
            raise TypeError("pixels must be a numpy ndarray")

        pixels = np.array(self.pixels, copy=True)

        if pixels.dtype != np.uint8:
            raise ValueError("pixels dtype must be uint8")

        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("pixels must have shape (H, W, 3)")

        if not pixels.flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)

        object.__setattr__(self, "pixels", pixels)

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def nbytes(self) -> int:
        return self.pixels.nbytes

    # ---------- Utility ----------

    def copy(self) -> RawImage:
        return RawImage(self.pixels.copy(), self.image_name)


def _is_json_serializable(obj: Any) -> bool:
    try:
        json.dumps(obj, sort_keys=True)
        return True
    except (TypeError, OverflowError):
        return False


@dataclass(slots=True, frozen=True)
class CompressedImage:
    """
    Immutable container for compressed image data.

    CompressedImage represents an opaque compressed payload produced
    by a codec, along with codec-specific metadata required for
    reconstruction.

    The payload is treated as an abstract binary blob. The container
    makes no assumptions about file format, structure, or internal
    encoding.

    Invariants
    ----------
    - image_id must be a non-empty string
    - codec must be a non-empty string
    - payload must be non-empty bytes
    - compression_data must be JSON-serializable
    - structural immutability enforced via frozen dataclass

    Parameters
    ----------
    image_id : str
        Unique experiment identifier corresponding to the source RawImage.

    codec : str
        Identifier of the codec that produced this compressed artifact.

    payload : bytes
        Opaque compressed binary data.

    compression_data : dict[str, Any], optional
        Codec-specific metadata required for decoding or reconstruction.
        This dictionary must be JSON-serializable. Raw binary data is not
        allowed here and must instead be stored within `payload`.

    Size Accounting
    ---------------
    The total compressed size includes both:

        - payload_size_bytes
        - compression_data_size_bytes (JSON canonical encoding)

    This ensures fair and deterministic benchmarking comparisons.

    Properties
    ----------
    payload_size_bytes : int
        Size of the compressed payload in bytes.

    compression_data_size_bytes : int
        Deterministic size of metadata after canonical JSON encoding.

    total_size_bytes : int
        Combined size of payload and metadata.

    Design Philosophy
    -----------------
    This class is purely a data container. It does not:

        - decode itself
        - compute distortion metrics
        - perform benchmarking
        - handle file I/O

    Those behaviors belong to codec implementations and benchmarking modules.
    """

    image_id: str                       # Unique experiment ID from RawImage
    codec: str                          # Codec identifier (e.g. "jpeg", "mycodec")
    payload: bytes                      # Opaque compressed blob
    compression_data: dict[str, Any] = field(default_factory=dict)
    # Codec-specific metadata required for decoding/reconstruction

    def __post_init__(self):
        # ---------- Validate identifiers ----------

        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be a non-empty string")

        if not isinstance(self.codec, str) or not self.codec.strip():
            raise ValueError("codec must be a non-empty string")

        # ---------- Validate payload ----------

        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")

        payload_bytes = bytes(self.payload)

        if len(payload_bytes) == 0:
            raise ValueError("payload must be non-empty")

        object.__setattr__(self, "payload", payload_bytes)

        # ---------- Validate compression_data ----------

        if self.compression_data is None:
            object.__setattr__(self, "compression_data", {})

        if not isinstance(self.compression_data, dict):
            raise TypeError("compression_data must be a dictionary")

        if not _is_json_serializable(self.compression_data):
            raise ValueError(
                "compression_data must be JSON-serializable "
                "(no raw bytes or unsupported objects)"
            )

        # ---------- Size Helpers ----------

    @property
    def payload_size_bytes(self) -> int:
        return len(self.payload)

    @property
    def compression_data_size_bytes(self) -> int:
        # Canonical deterministic encoding
        encoded = json.dumps(
            self.compression_data,
            sort_keys=True,
            separators=(",", ":")
        ).encode("utf-8")
        return len(encoded)

    @property
    def total_size_bytes(self) -> int:
        return self.payload_size_bytes + self.compression_data_size_bytes
