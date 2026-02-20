from __future__ import annotations

from typing import Iterable, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

from image import RawImage  # adjust import to your project layout


def display_images(
    images: Sequence[RawImage],
    *,
    cols: Optional[int] = None,
    figsize: Optional[tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    show_axes: bool = False,
) -> None:
    """
    Display RawImage objects using matplotlib, with image_name shown above each image.

    Parameters
    ----------
    images:
        Sequence of RawImage objects to display.
    cols:
        Number of columns in the grid. If None, uses len(images) (single row).
        If provided, images will wrap to multiple rows.
    figsize:
        Figure size passed to matplotlib. If None, computed automatically.
    suptitle:
        Optional title for the entire figure.
    show_axes:
        If True, show axes ticks/frames. Default False (clean comparison view).
    """
    if not images:
        raise ValueError("display_images(): 'images' must be a non-empty sequence")

    n = len(images)
    if cols is None:
        cols = n
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    # Compute a reasonable default figsize
    if figsize is None:
        # ~4 inches per image horizontally, ~4 vertically per row (tweak as you like)
        figsize = (max(6.0, 4.0 * min(cols, n)), max(4.0, 4.0 * rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # Normalize axes into a flat list
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]

    for ax in axes_list[n:]:
        ax.axis("off")

    for ax, img in zip(axes_list, images):
        ax.imshow(img.pixels)
        ax.set_title(img.image_name, fontsize=10, pad=8)

        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout()
    plt.show()