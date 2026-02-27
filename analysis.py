from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .image import RawImage, CompressedImage


# ----------------------------
# Size / rate helpers
# ----------------------------

def total_size_bytes(blob: CompressedImage) -> int:
    """Total size in bytes, including payload + JSON metadata."""
    return blob.total_size_bytes


def bits_per_pixel(
    blob: CompressedImage,
    *,
    image_shape: Optional[Tuple[int, int, int]] = None,
    raw: Optional[RawImage] = None,
) -> float:
    """
    Compute bits-per-pixel (bpp) for a compressed image.

    You must provide image dimensions either via:
      - raw=RawImage, OR
      - image_shape=(H, W, C)

    As a convenience fallback, if neither is provided, this function will try
    to read compression_data["shape"] if present.

    Notes
    -----
    - Uses total_size_bytes (payload + metadata) for fair accounting.
    - bpp is computed per pixel (H*W), not per channel.
    """
    if raw is not None:
        h, w, _ = raw.shape
    elif image_shape is not None:
        h, w, _ = image_shape
    else:
        # best-effort fallback for codecs that include shape sidecar metadata
        cd = blob.compression_data
        if isinstance(cd, dict) and "shape" in cd:
            s = cd["shape"]
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                h, w = int(s[0]), int(s[1])
            else:
                raise ValueError("compression_data['shape'] exists but is not a valid (H,W,...) sequence")
        else:
            raise ValueError(
                "bits_per_pixel requires raw=RawImage or image_shape=(H,W,C). "
                "No usable shape found in compression_data."
            )

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions for bpp: {(h, w)}")

    return (blob.total_size_bytes * 8.0) / (h * w)


# ----------------------------
# Distortion / fidelity metrics
# ----------------------------

def mse(a: RawImage, b: RawImage) -> float:
    """
    Mean Squared Error between two RawImages (RGB uint8).

    Returns
    -------
    float
        Mean squared error over all pixels and channels.
    """
    _require_same_shape(a, b)
    x = a.pixels.astype(np.float32)
    y = b.pixels.astype(np.float32)
    diff = x - y
    return float(np.mean(diff * diff))


def changed_pixel_ratio(a: RawImage, b: RawImage, *, threshold: float = 0.0) -> float:
    """
    Fraction of pixels whose RGB difference magnitude exceeds threshold.
    threshold=0 counts any change at all.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    err = _pixel_error_l2(a, b)
    changed = err > threshold
    return float(np.mean(changed))


def mse_on_changed_pixels(a: RawImage, b: RawImage, *, threshold: float = 0.0) -> float:
    """
    MSE computed only over pixels whose RGB difference magnitude exceeds threshold.
    Returns 0.0 if no pixels exceed threshold.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    x = a.pixels.astype(np.float32)
    y = b.pixels.astype(np.float32)
    diff = x - y

    err_l2 = np.sqrt(np.sum(diff * diff, axis=2))  # (H, W)
    mask = err_l2 > threshold

    if not np.any(mask):
        return 0.0

    # Compute squared error per pixel across channels, then average over masked pixels
    se = np.sum(diff * diff, axis=2)  # (H, W) sum over channels
    return float(np.mean(se[mask]))


def psnr(a: RawImage, b: RawImage, *, data_range: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.

    For 8-bit RGB images, data_range should be 255.

    Returns
    -------
    float
        PSNR in decibels. Returns +inf when images are identical.
    """
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(m))


def ssim(a: RawImage, b: RawImage) -> float:
    """
    Structural Similarity Index (SSIM).

    Implementation note:
    - Uses scikit-image if available.
    - If not installed, raises a clear error.

    Returns
    -------
    float
        SSIM in [-1, 1], where 1 is perfect match.
    """
    _require_same_shape(a, b)

    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError as e:
        raise ImportError(
            "SSIM requires scikit-image. Install with: pip install scikit-image"
        ) from e

    # skimage expects channel_axis for color images
    x = a.pixels
    y = b.pixels
    return float(_ssim(x, y, channel_axis=2, data_range=255))


def lpips(a: RawImage, b: RawImage, *, net: str = "alex") -> float:
    """
    LPIPS perceptual distance.

    Implementation note:
    - Uses PyTorch + the 'lpips' package if available.
    - If not installed, raises a clear error.

    Returns
    -------
    float
        Lower is better. 0 means identical (approximately, depending on net).
    """
    _require_same_shape(a, b)

    try:
        import torch
        import lpips as lpips_pkg
    except ImportError as e:
        raise ImportError(
            "LPIPS requires torch + lpips. Install with: pip install torch lpips"
        ) from e

    model = _get_lpips_model(lpips_pkg, net=net)

    # Convert uint8 [0,255] to float32 [-1,1], shape [N,C,H,W]
    x = _to_torch_nchw(a.pixels)
    y = _to_torch_nchw(b.pixels)

    with torch.no_grad():
        d = model(x, y)
    return float(d.item())


# ----------------------------
# Internals
# ----------------------------

def _require_same_shape(a: RawImage, b: RawImage) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


def _to_torch_nchw(rgb_uint8: np.ndarray):
    import torch
    x = rgb_uint8.astype(np.float32) / 255.0  # [0,1]
    x = x * 2.0 - 1.0                         # [-1,1]
    x = np.transpose(x, (2, 0, 1))            # CHW
    x = np.expand_dims(x, axis=0)             # NCHW
    return torch.from_numpy(x)


# Cache LPIPS models by net name so repeated calls are cheap
_LPIPS_CACHE = {}


def _get_lpips_model(lpips_pkg, *, net: str):
    import torch
    key = str(net).lower().strip()
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]

    model = lpips_pkg.LPIPS(net=key)
    model.eval()

    # If CUDA is available you *may* choose to move it, but for reproducibility
    # and simplicity, keep CPU by default.
    # if torch.cuda.is_available():
    #     model = model.cuda()

    _LPIPS_CACHE[key] = model
    return model


def _pixel_error_l2(a: RawImage, b: RawImage) -> np.ndarray:
    # returns (H, W) float32 L2 error per pixel across RGB
    x = a.pixels.astype(np.float32)
    y = b.pixels.astype(np.float32)
    d = x - y
    return np.sqrt(np.sum(d * d, axis=2))

