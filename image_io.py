from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

from image import RawImage


PathLike = Union[str, Path]


def read_image(path: PathLike, *, image_name: str | None = None) -> RawImage:
    """
    Load an image file into a RawImage.

    The image is converted to RGB and normalized to uint8 (H, W, 3).

    Parameters
    ----------
    path : str | Path
        Path to image file.
    image_name : str, optional
        Unique experiment ID. If None, filename stem is used.

    Returns
    -------
    RawImage
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        pixels = np.asarray(img, dtype=np.uint8)

    if image_name is None:
        image_name = path.stem

    return RawImage(pixels=pixels, image_name=image_name)


def write_image(path: PathLike, image: RawImage) -> None:
    """
    Write a RawImage to disk.

    Parameters
    ----------
    path : str | Path
        Output file path.
    image : RawImage
        Image to write.
    """
    path = Path(path)

    img = Image.fromarray(image.pixels, mode="RGB")
    img.save(path)