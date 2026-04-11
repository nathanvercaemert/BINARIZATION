"""
Run DP-LinkNet and SBB binarization in sequence, then merge their outputs.

This orchestrator treats the existing pipeline scripts as black boxes:
  1. Run DP-LinkNet unchanged on the source images.
  2. Run SBB unchanged on the same source images.
  3. Combine the resulting binary masks with a pixelwise union.
  4. Save archival-quality bilevel TIFF outputs with preserved resolution.

The orchestrator does not modify either pipeline's inference behavior.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger("orchestrate_binarization")

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
}


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def collect_images(directory: Path) -> list[Path]:
    files = []
    for name in os.listdir(directory):
        path = directory / name
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    files.sort()
    return files


def verify_crop_prefix(images: list[Path]) -> None:
    for image_path in images:
        if not image_path.name.startswith("CROP"):
            raise ValueError(
                f"filename does not start with 'CROP': {image_path}"
            )


def build_binary_name(image_path: Path) -> str:
    return "BINARY" + image_path.name[4:]


def read_image_mask(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read mask image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img > 0


def read_resolution(path: Path) -> tuple[float | None, float | None]:
    import pyvips

    img = pyvips.Image.new_from_file(str(path), access="sequential")

    xres = img.get("Xres") if img.get_typeof("Xres") != 0 else None
    yres = img.get("Yres") if img.get_typeof("Yres") != 0 else xres
    return xres, yres


def save_bilevel_tiff(
    mask: np.ndarray,
    output_path: Path,
    xres: float | None,
    yres: float | None,
) -> None:
    import pyvips

    bilevel = (mask.astype(np.uint8) * 255)
    h, w = bilevel.shape
    vimg = pyvips.Image.new_from_memory(bilevel.data, w, h, 1, "uchar")
    if xres is not None or yres is not None:
        vimg = vimg.copy(xres=xres, yres=yres)
    vimg.write_to_file(
        str(output_path),
        compression="none",
        bitdepth=1,
    )


def run_pipeline(
    command: list[str],
    label: str,
    expected_output_dir: Path | None = None,
) -> None:
    logger.info("Running %s", label)
    logger.info("Command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    logger.info("%s return code: %d", label, completed.returncode)
    if completed.stdout:
        logger.info("%s stdout:\n%s", label, completed.stdout.rstrip())
    if completed.stderr:
        logger.warning("%s stderr:\n%s", label, completed.stderr.rstrip())
    if completed.returncode != 0:
        produced = []
        if expected_output_dir is not None and expected_output_dir.is_dir():
            produced = sorted(p.name for p in expected_output_dir.iterdir())
        detail = [
            f"{label} failed with exit code {completed.returncode}",
        ]
        if expected_output_dir is not None:
            detail.append(
                f"expected output dir: {expected_output_dir}"
            )
            detail.append(
                f"produced files: {produced if produced else 'none'}"
            )
        if not completed.stdout and not completed.stderr:
            detail.append("subprocess produced no stdout/stderr")
        raise RuntimeError(
            "; ".join(detail)
        )


def merge_outputs(
    images: list[Path],
    dp_output_dir: Path,
    sbb_output_dir: Path,
    final_output_dir: Path,
) -> None:
    failed: list[tuple[str, str]] = []
    total = len(images)

    for idx, image_path in enumerate(images, 1):
        binary_name = build_binary_name(image_path)
        dp_path = dp_output_dir / binary_name
        sbb_path = sbb_output_dir / binary_name
        final_path = final_output_dir / binary_name

        try:
            if not dp_path.is_file():
                raise FileNotFoundError(f"DP-LinkNet output missing: {dp_path}")
            if not sbb_path.is_file():
                raise FileNotFoundError(f"SBB output missing: {sbb_path}")

            dp_mask = read_image_mask(dp_path)
            sbb_mask = read_image_mask(sbb_path)

            if dp_mask.shape != sbb_mask.shape:
                raise ValueError(
                    f"shape mismatch: DP-LinkNet {dp_mask.shape} vs SBB "
                    f"{sbb_mask.shape}"
                )

            union_mask = np.logical_or(dp_mask, sbb_mask)
            xres, yres = read_resolution(image_path)
            save_bilevel_tiff(union_mask, final_path, xres, yres)
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
            rel = os.path.relpath(image_path, image_path.parent)
            logger.error("[%d/%d] %s -- %s", idx, total, rel, exc)
            failed.append((image_path.name, str(exc)))
            continue

        logger.info("[%d/%d] %s -- merged", idx, total, image_path.name)

    if failed:
        logger.info("%d merge failure(s):", len(failed))
        for name, err in failed:
            logger.info("  - %s: %s", name, err)
        raise RuntimeError("One or more merges failed")


def main() -> None:
    global np

    parser = argparse.ArgumentParser(
        description="Run DP-LinkNet and SBB binarization unchanged, then "
                    "merge the outputs as a pixelwise union."
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing CROP-prefixed source images",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write final merged BINARY-prefixed TIFF outputs",
    )
    parser.add_argument(
        "--dplinknet-weights-dir",
        default="DP_LINKNET/weights",
        help="Directory containing DP-LinkNet .th weight files",
    )
    parser.add_argument(
        "--sbb-model-dir",
        default="SBB/model",
        help="Directory containing the SBB SavedModel",
    )
    parser.add_argument(
        "--dataset",
        default="dibco",
        help="Dataset name for selecting DP-LinkNet weights",
    )
    parser.add_argument(
        "--model",
        default="dplinknet34",
        help="DP-LinkNet model architecture name",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Pass through to DP-LinkNet to disable 8-view TTA",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional DP-LinkNet threshold override",
    )
    parser.add_argument(
        "--dplinknet-python",
        default=sys.executable,
        help="Python executable used to invoke the DP-LinkNet script",
    )
    parser.add_argument(
        "--sbb-python",
        default=sys.executable,
        help="Python executable used to invoke the SBB script",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for intermediate pipeline outputs",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep DP-LinkNet and SBB intermediate output directories",
    )
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        import pyvips  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency for the orchestrator: "
            f"{exc.name}. Install the packages needed to read, merge, and "
            "write TIFF masks before running this script."
        ) from exc

    _configure_logging()

    root_dir = Path(__file__).resolve().parent
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dplinknet_weights_dir = Path(args.dplinknet_weights_dir).resolve()
    sbb_model_dir = Path(args.sbb_model_dir).resolve()
    dplinknet_script = root_dir / "DP_LINKNET" / "binarize_with_dplinknet.py"
    sbb_script = root_dir / "SBB" / "binarize_with_sbb.py"

    if not image_dir.is_dir():
        raise SystemExit(f"image directory not found: {image_dir}")
    if not dplinknet_weights_dir.is_dir():
        raise SystemExit(
            f"DP-LinkNet weights directory not found: {dplinknet_weights_dir}"
        )
    if not sbb_model_dir.is_dir():
        raise SystemExit(f"SBB model directory not found: {sbb_model_dir}")
    if not dplinknet_script.is_file():
        raise SystemExit(f"DP-LinkNet script not found: {dplinknet_script}")
    if not sbb_script.is_file():
        raise SystemExit(f"SBB script not found: {sbb_script}")

    images = collect_images(image_dir)
    if not images:
        raise SystemExit(f"no image files found in {image_dir}")
    verify_crop_prefix(images)

    output_dir.mkdir(parents=True, exist_ok=True)

    temp_ctx = None
    if args.work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="binarization_orchestrator_")
        work_dir = Path(temp_ctx.name)
    else:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    dp_output_dir = work_dir / "dplinknet_output"
    sbb_output_dir = work_dir / "sbb_output"
    dp_output_dir.mkdir(parents=True, exist_ok=True)
    sbb_output_dir.mkdir(parents=True, exist_ok=True)

    dplinknet_command = [
        args.dplinknet_python,
        "-u",
        str(dplinknet_script),
        str(image_dir),
        str(dp_output_dir),
        str(dplinknet_weights_dir),
        "--dataset",
        args.dataset,
        "--model",
        args.model,
    ]
    if args.no_tta:
        dplinknet_command.append("--no-tta")
    if args.threshold is not None:
        dplinknet_command.extend(["--threshold", str(args.threshold)])

    sbb_command = [
        args.sbb_python,
        "-u",
        str(sbb_script),
        str(image_dir),
        str(sbb_output_dir),
        str(sbb_model_dir),
    ]

    logger.info("Found %d input image(s) in '%s'", len(images), image_dir)
    logger.info("Final output directory: '%s'", output_dir)
    logger.info("Intermediate work directory: '%s'", work_dir)

    try:
        run_pipeline(
            dplinknet_command,
            "DP-LinkNet pipeline",
            expected_output_dir=dp_output_dir,
        )
        run_pipeline(
            sbb_command,
            "SBB pipeline",
            expected_output_dir=sbb_output_dir,
        )
        merge_outputs(images, dp_output_dir, sbb_output_dir, output_dir)
    finally:
        if temp_ctx is not None and not args.keep_intermediates:
            temp_ctx.cleanup()
        elif temp_ctx is None and not args.keep_intermediates:
            shutil.rmtree(dp_output_dir, ignore_errors=True)
            shutil.rmtree(sbb_output_dir, ignore_errors=True)

    logger.info("Merged %d image(s) successfully.", len(images))


if __name__ == "__main__":
    main()
