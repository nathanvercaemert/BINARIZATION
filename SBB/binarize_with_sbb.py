"""
Binarize CROP-prefixed images using the SBB hybrid CNN-Transformer model.

Processes each CROP-prefixed image in image_dir through
the SBB binarization model, and writes a single-channel binary mask to
output_dir with the "BINARY" prefix.  Every image filename must start
with the prefix "CROP"; the script terminates immediately if any does
not.

Unlike other pipeline steps, this script does NOT modify or remove input
files.  The output directory receives the binarized results.

The SBB model is loaded once from a TensorFlow SavedModel directory
(containing saved_model.pb with variables/ and assets/ subdirectories,
or .h5 files).  The SbbBinarizer class handles patch extraction,
normalization, inference, and stitching internally.

Usage:
    py -3.11 binarize_with_sbb.py <image_dir> <output_dir> <model_dir>

Requires (Python 3.11, TensorFlow < 2.13):
    py -3.11 -m pip install "tensorflow<2.13" sbb-binarization pyvips-binary
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import pyvips
from sbb_binarize.sbb_binarize import SbbBinarizer

logger = logging.getLogger("binarize_with_sbb")


def _configure_logging() -> None:
    """Set up one-line structured logging to stderr."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}


def collect_images(directory: str) -> list[str]:
    """Return sorted list of image file paths in *directory*."""
    files = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(directory, name))
    files.sort()
    return files


def verify_crop_prefix(images: list[str]) -> None:
    """Terminate if any image filename does not start with 'CROP'."""
    for image_path in images:
        filename = os.path.basename(image_path)
        if not filename.startswith("CROP"):
            logger.error("filename does not start with 'CROP': %s", image_path)
            sys.exit(1)


def build_binary_path(image_path: str, output_root: str) -> str:
    """Map a CROP image path to its BINARY output path under output_root."""
    filename = os.path.basename(image_path)
    binary_name = "BINARY" + filename[4:]
    return os.path.join(output_root, binary_name)


def read_image_bgr8(path: str) -> np.ndarray:
    """Read an image as 8-bit BGR using OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def read_dpi(path: str) -> float:
    """Read the horizontal resolution from an image file via pyvips."""
    img = pyvips.Image.new_from_file(path, access="sequential")
    return img.get("Xres")


def save_bilevel_tiff(
    mask: np.ndarray, output_path: str, xres: float,
) -> None:
    """Save a 0/255 mask as an uncompressed 1-bit TIFF with DPI preserved."""
    h, w = mask.shape
    vimg = pyvips.Image.new_from_memory(mask.data, w, h, 1, "uchar")
    vimg = vimg.copy(xres=xres, yres=xres)
    vimg.write_to_file(
        output_path,
        compression="none",
        bitdepth=1,
    )


def binarize_image(
    image_path: str,
    output_path: str,
    binarizer: SbbBinarizer,
) -> None:
    """Read an image, run SBB binarization, and save the binary mask."""
    logger.debug("Binarizing '%s'", image_path)

    xres = read_dpi(image_path)
    img = read_image_bgr8(image_path)

    mask = binarizer.run(image=img)

    save_bilevel_tiff(mask.astype(np.uint8), output_path, xres)
    logger.debug("Saved '%s'", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize CROP-prefixed images using the SBB hybrid "
                    "CNN-Transformer model.  Writes binary masks with "
                    "'BINARY' prefix to output_dir.  Input files are not "
                    "modified."
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing CROP-prefixed images",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write BINARY-prefixed output masks",
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing the SBB TensorFlow SavedModel "
             "(saved_model.pb with variables/ and assets/) or .h5 files",
    )
    args = parser.parse_args()

    _configure_logging()

    if not os.path.isdir(args.image_dir):
        logger.error("image directory not found: %s", args.image_dir)
        sys.exit(1)

    if not os.path.isdir(args.model_dir):
        logger.error("model directory not found: %s", args.model_dir)
        sys.exit(1)

    images = collect_images(args.image_dir)
    if not images:
        logger.error("no image files found in %s", args.image_dir)
        sys.exit(1)

    verify_crop_prefix(images)

    logger.info("Loading SBB binarization model from '%s'", args.model_dir)
    binarizer = SbbBinarizer(args.model_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []

    logger.info("Found %d image(s) under '%s'", total, args.image_dir)
    logger.info("Output directory: '%s'", args.output_dir)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        output_path = build_binary_path(image_path, args.output_dir)

        try:
            binarize_image(image_path, output_path, binarizer)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK", idx, total, rel)

    binarizer.end_session()

    logger.info(
        "Processed %d/%d image(s) successfully.", total - len(failed), total,
    )
    if failed:
        logger.info("%d failure(s):", len(failed))
        for name, err in failed:
            logger.info("  - %s: %s", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
