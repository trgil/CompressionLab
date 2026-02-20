from pathlib import Path

from image_io import read_image
from display import display_images
from compression_codecs.base import EncodeRequest
from compression_codecs.registry import CodecRegistry
from compression_codecs.identity import IdentityCodec


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load raw image
    # ---------------------------------------------------------
    image_path = Path("images/exmpl1.bmp")
    raw = read_image(image_path, image_name="exp_001")

    print(f"Loaded image: {raw.image_name}")
    print(f"Dimensions: {raw.width} x {raw.height}")
    print(f"Raw pixel memory: {raw.nbytes} bytes")
    print()

    # ---------------------------------------------------------
    # 2. Setup codec registry
    # ---------------------------------------------------------
    registry = CodecRegistry()
    registry.register(IdentityCodec())

    codec = registry.get("identity")

    # ---------------------------------------------------------
    # 3. Encode
    # ---------------------------------------------------------
    blob = codec.encode(EncodeRequest(image=raw, params={}))

    print(f"Codec: {codec.name}")
    print(f"Payload size: {blob.payload_size_bytes} bytes")
    print(f"Metadata size: {blob.compression_data_size_bytes} bytes")
    print(f"Total compressed size: {blob.total_size_bytes} bytes")

    # Calculate bits-per-pixel
    h, w, _ = raw.shape
    bpp = (blob.total_size_bytes * 8) / (h * w)

    print(f"Bits per pixel (bpp): {bpp:.2f}")
    print()

    # ---------------------------------------------------------
    # 4. Decode
    # ---------------------------------------------------------
    decoded = codec.decode(blob)

    # ---------------------------------------------------------
    # 5. Display comparison
    # ---------------------------------------------------------
    display_images(
        [raw, decoded],
        suptitle="Identity Codec Test (Baseline)"
    )


if __name__ == "__main__":
    main()
