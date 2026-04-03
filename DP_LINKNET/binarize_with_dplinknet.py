"""
Binarize CROP-prefixed images using a trained DP-LinkNet model.

Processes each CROP-prefixed image in image_dir through
DP-LinkNet for document binarization, and writes a single-channel binary
mask to output_dir with the "BINARY" prefix.  Every image filename must
start with the prefix "CROP"; the script terminates immediately if any
does not.

Unlike other pipeline steps, this script does NOT modify or remove input
files.  The output directory receives the binarized results.

The inference pipeline matches the DP-LinkNet repository's test-time
behavior: BGR input via OpenCV, repo-specific normalization, 256x256
overlapping tiles, optional 8-view test-time augmentation, and
thresholded sigmoid output.

The weights directory should contain .th files named as
{dataset}_{model}.th (e.g. dibco_dplinknet34.th).  The script selects
the correct file from --dataset and --model.

Usage:
    python binarize_with_dplinknet.py <image_dir> <output_dir> <weights_dir> --dataset dibco
    python binarize_with_dplinknet.py <image_dir> <output_dir> <weights_dir> --dataset dibco --model dplinknet34
    python binarize_with_dplinknet.py <image_dir> <output_dir> <weights_dir> --dataset dibco --no-tta

Requires (Python >= 3.9):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install "opencv-python>=4.5" "numpy>=1.22" pyvips
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import pyvips
import torch

from networks import MODEL_REGISTRY

logger = logging.getLogger("binarize_with_dplinknet")


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

TILE_SIZE = 256
PADDING_SIZE = 21

TTA_THRESHOLD = 5.0
SINGLE_THRESHOLD = 0.5


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


def load_model(weights_path: str, model_name: str) -> torch.nn.Module:
    """Instantiate the model, load weights on CPU, and set to eval mode."""
    if model_name not in MODEL_REGISTRY:
        logger.error(
            "unknown model '%s'. Choose from: %s",
            model_name, ", ".join(sorted(MODEL_REGISTRY.keys())),
        )
        sys.exit(1)

    model = MODEL_REGISTRY[model_name]()

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.removeprefix("module."): v for k, v in state_dict.items()
        }

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logger.error(
            "weight loading failed. This usually means the checkpoint was "
            "saved from a different architecture than '%s'. Check --model "
            "and verify that the layer names in BINARIZATION/networks.py "
            "match the checkpoint.", model_name,
        )
        raise

    model.eval()
    return model


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


def preprocess_tile(tile: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> float32 CHW tensor, DP-LinkNet normalization."""
    x = tile.astype(np.float32) / 255.0 * 3.2 - 1.6
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)


def predict_tile(model: torch.nn.Module, tile: np.ndarray) -> np.ndarray:
    """Run a single forward pass on one tile.  Returns HW float32 map."""
    tensor = preprocess_tile(tile).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze().cpu().numpy()


def predict_tile_tta(model: torch.nn.Module, tile: np.ndarray) -> np.ndarray:
    """Run batched 8-view TTA on one tile.  Returns summed HW float32 map."""
    t = tile.transpose(1, 0, 2)

    views = [
        tile,                                         # 0: original
        np.flip(tile, 1).copy(),                      # 1: hflip
        np.flip(tile, 0).copy(),                      # 2: vflip
        np.flip(np.flip(tile, 0), 1).copy(),          # 3: hvflip
        t.copy(),                                     # 4: transpose
        np.flip(t, 1).copy(),                         # 5: transpose + hflip
        np.flip(t, 0).copy(),                         # 6: transpose + vflip
        np.flip(np.flip(t, 0), 1).copy(),             # 7: transpose + hvflip
    ]

    batch = torch.stack([preprocess_tile(v) for v in views])
    with torch.no_grad():
        preds = model(batch).squeeze(1).cpu().numpy()

    # Invert each augmentation to align predictions back to original orientation
    preds[1] = np.flip(preds[1], 1)
    preds[2] = np.flip(preds[2], 0)
    preds[3] = np.flip(np.flip(preds[3], 0), 1)
    preds[4] = preds[4].T
    preds[5] = np.flip(preds[5], 1).T
    preds[6] = np.flip(preds[6], 0).T
    preds[7] = np.flip(np.flip(preds[7], 0), 1).T

    return preds.sum(axis=0).copy()


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
    model: torch.nn.Module,
    tta: bool,
    threshold: float,
) -> None:
    """Read an image, tile it, run inference, stitch, threshold, and save."""
    logger.debug("Binarizing '%s'", image_path)

    xres = read_dpi(image_path)
    img = read_image_bgr8(image_path)
    h, w = img.shape[:2]
    stride = TILE_SIZE - 2 * PADDING_SIZE

    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride

    img_padded = np.pad(
        img,
        ((PADDING_SIZE, PADDING_SIZE + pad_h),
         (PADDING_SIZE, PADDING_SIZE + pad_w),
         (0, 0)),
        mode="reflect",
    )

    n_y = (h + pad_h) // stride
    n_x = (w + pad_w) // stride
    total_tiles = n_y * n_x
    logger.debug(
        "Image %dx%d, padded to %dx%d, %dx%d = %d tile(s)",
        w, h, w + pad_w, h + pad_h, n_x, n_y, total_tiles,
    )

    output = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)

    tile_idx = 0
    for iy in range(n_y):
        for ix in range(n_x):
            tile_idx += 1
            y = iy * stride
            x = ix * stride
            tile = img_padded[y:y + TILE_SIZE, x:x + TILE_SIZE]

            if tta:
                pred = predict_tile_tta(model, tile)
            else:
                pred = predict_tile(model, tile)

            inner = pred[PADDING_SIZE:PADDING_SIZE + stride,
                         PADDING_SIZE:PADDING_SIZE + stride]
            output[iy * stride:(iy + 1) * stride,
                   ix * stride:(ix + 1) * stride] = inner

    output = output[:h, :w]

    mask = (output > threshold).astype(np.uint8) * 255

    save_bilevel_tiff(mask, output_path, xres)
    logger.debug("Saved '%s'", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binarize CROP-prefixed images using a trained DP-LinkNet "
                    "model.  Writes binary masks with 'BINARY' prefix to "
                    "output_dir.  Input files are not modified."
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
        "weights_dir",
        help="Directory containing .th weight files named "
             "{dataset}_{model}.th",
    )
    parser.add_argument(
        "--dataset", default="dibco",
        help="Dataset name used to select the weight file "
             "(default: dibco)",
    )
    parser.add_argument(
        "--model", default="dplinknet34",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture matching the weights (default: dplinknet34)",
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable 8-view test-time augmentation (faster, slightly lower "
             "quality).  Threshold changes from 5.0 to 0.5.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override binarization threshold (default: 5.0 with TTA, "
             "0.5 without TTA)",
    )
    args = parser.parse_args()

    _configure_logging()

    if not os.path.isdir(args.image_dir):
        logger.error("image directory not found: %s", args.image_dir)
        sys.exit(1)

    if not os.path.isdir(args.weights_dir):
        logger.error("weights directory not found: %s", args.weights_dir)
        sys.exit(1)

    weights_file = f"{args.dataset}_{args.model}.th"
    weights_path = os.path.join(args.weights_dir, weights_file)
    if not os.path.isfile(weights_path):
        available = [f for f in os.listdir(args.weights_dir) if f.endswith(".th")]
        logger.error(
            "weight file not found: %s; available in '%s': %s",
            weights_path, args.weights_dir,
            ", ".join(sorted(available)) if available else "(none)",
        )
        sys.exit(1)

    images = collect_images(args.image_dir)
    if not images:
        logger.error("no image files found in %s", args.image_dir)
        sys.exit(1)

    verify_crop_prefix(images)

    tta = not args.no_tta
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = TTA_THRESHOLD if tta else SINGLE_THRESHOLD

    logger.info("Loading model '%s' from '%s'", args.model, weights_path)
    model = load_model(weights_path, args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(images)
    failed: list[tuple[str, str]] = []
    mode_label = "8-view TTA" if tta else "single-view"

    logger.info("Found %d image(s) under '%s'", total, args.image_dir)
    logger.info("Output directory: '%s'", args.output_dir)
    logger.info("Mode: %s, threshold: %s", mode_label, threshold)

    for idx, image_path in enumerate(images, 1):
        rel = os.path.relpath(image_path, args.image_dir)
        output_path = build_binary_path(image_path, args.output_dir)

        try:
            binarize_image(image_path, output_path, model, tta, threshold)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("[%d/%d] %s -- %s", idx, total, rel, e)
            failed.append((rel, str(e)))
            continue

        logger.debug("[%d/%d] %s -- OK", idx, total, rel)

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
