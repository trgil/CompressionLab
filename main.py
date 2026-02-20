from pathlib import Path

from image_io import read_image
from display import display_images
from image import RawImage


def main() -> None:
    # Path to example image
    image_path = Path("images/exmpl1.bmp")

    # Load image into RawImage
    original = read_image(image_path, image_name="original")

    # Create a copy with a new experiment ID
    copy = RawImage(
        pixels=original.pixels,
        image_name="copy"
    )

    # Display both side-by-side
    display_images(
        [original, copy],
        suptitle="RawImage Copy Test"
    )


if __name__ == "__main__":
    main()
