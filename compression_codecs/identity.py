from __future__ import annotations

import numpy as np

from image import RawImage, CompressedImage
from .base import EncodeRequest


class IdentityCodec:
    """
    Baseline codec that performs no compression.

    Stores raw RGB bytes as payload and only the minimal metadata
    required to reconstruct the image.
    """

    @property
    def name(self) -> str:
        return "identity"

    def encode(self, req: EncodeRequest) -> CompressedImage:
        pixels = req.image.pixels

        payload = pixels.tobytes()

        # Absolute minimum required to decode
        compression_data = {
            "shape": list(pixels.shape),
            "dtype": "uint8",
        }

        return CompressedImage(
            image_id=req.image.image_name,
            codec=self.name,
            payload=payload,
            compression_data=compression_data,
        )

    def decode(self, blob: CompressedImage) -> RawImage:
        cd = blob.compression_data

        shape = tuple(cd["shape"])
        dtype = np.dtype(cd["dtype"])

        arr = np.frombuffer(blob.payload, dtype=dtype).reshape(shape)

        return RawImage(
            pixels=arr,
            image_name=f"{blob.image_id}__identity__decoded",
        )
