"""
Run DP-LinkNet and SBB binarization in sequence, then merge their outputs.

This orchestrator treats the existing pipeline scripts as black boxes:
  1. Run DP-LinkNet unchanged on the source images.
  2. Run SBB unchanged on the same source images.
  3. Combine the resulting binary masks so black wins at each pixel.
  4. Save archival-quality bilevel TIFF outputs with preserved resolution.

The orchestrator does not modify either pipeline's inference behavior.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PipelineResult:
    returncode: int
    stdout: str
    stderr: str


CommandBuilder = Callable[[Path, Path], list[str]]


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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def build_binary_name(image_path: Path) -> str:
    return "BINARY" + image_path.name[4:]


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def validate_image_file(path: Path, full_decode: bool = True) -> str | None:
    if not path.is_file():
        return "missing"

    try:
        import pyvips

        img = pyvips.Image.new_from_file(str(path), access="sequential")
        if img.width < 1 or img.height < 1:
            return f"invalid dimensions: {img.width}x{img.height}"
        if full_decode:
            # Force libvips to decode the image so truncated outputs are caught
            # before they are trusted for resume or SIGKILL recovery.
            img.avg()
    except Exception as exc:
        return f"unreadable: {exc}"

    return None


def output_validation_failures(
    images: list[Path],
    output_dir: Path,
) -> dict[Path, str]:
    failures = {}
    for image_path in images:
        binary_path = output_dir / build_binary_name(image_path)
        failure = validate_image_file(binary_path)
        if failure is not None:
            failures[image_path] = f"{binary_path}: {failure}"
    return failures


def final_output_is_valid(image_path: Path, output_dir: Path) -> bool:
    return validate_image_file(output_dir / build_binary_name(image_path)) is None


def format_image_names(images: list[Path], limit: int = 8) -> str:
    names = [p.name for p in images[:limit]]
    if len(images) > limit:
        names.append(f"... +{len(images) - limit} more")
    return ", ".join(names)


def read_image_mask(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read mask image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Treat black (0) as foreground so merging preserves any black pixel.
    return img == 0


def read_resolution(path: Path) -> tuple[float | None, float | None]:
    import pyvips

    img = pyvips.Image.new_from_file(str(path), access="sequential")

    xres = img.get("Xres") if img.get_typeof("Xres") != 0 else None
    yres = img.get("Yres") if img.get_typeof("Yres") != 0 else xres
    return xres, yres


def link_source_image(source: Path, destination: Path) -> None:
    try:
        os.symlink(source, destination)
        return
    except OSError:
        try:
            os.link(source, destination)
            return
        except OSError as hardlink_exc:
            raise RuntimeError(
                "Could not create symlink or hardlink for chunk input "
                f"'{source}' -> '{destination}'. Use a work directory on a "
                "filesystem that supports links; copying huge source images is "
                "intentionally avoided."
            ) from hardlink_exc


def build_chunk_input_dir(chunk_input_dir: Path, images: list[Path]) -> None:
    chunk_input_dir.mkdir(parents=True, exist_ok=False)
    for image_path in images:
        link_source_image(image_path, chunk_input_dir / image_path.name)


def prepare_sbb_compat_model_once(
    sbb_model_dir: Path,
    work_dir: Path,
) -> Path:
    saved_model_pb = sbb_model_dir / "saved_model.pb"
    saved_model_pbtxt = sbb_model_dir / "saved_model.pbtxt"
    if not saved_model_pb.is_file() and not saved_model_pbtxt.is_file():
        return sbb_model_dir

    compat_parent = work_dir / "sbb_model_compat"
    compat_model = compat_parent / sbb_model_dir.name
    compat_parent.mkdir(parents=True, exist_ok=True)

    if compat_model.exists():
        if (
            (compat_model / "saved_model.pb").is_file()
            or (compat_model / "saved_model.pbtxt").is_file()
        ):
            return compat_parent
        raise RuntimeError(
            f"Existing SBB compatibility path is not a SavedModel: "
            f"{compat_model}"
        )

    try:
        os.symlink(sbb_model_dir, compat_model, target_is_directory=True)
    except OSError:
        logger.warning(
            "Could not symlink SBB model into '%s'; copying model once instead",
            compat_model,
        )
        shutil.copytree(sbb_model_dir, compat_model)

    return compat_parent


def save_bilevel_tiff(
    mask: np.ndarray,
    output_path: Path,
    xres: float | None,
    yres: float | None,
) -> None:
    import pyvips

    # `mask` uses True for black pixels, but the TIFF data is encoded as 0=black,
    # 255=white.
    bilevel = ((~mask).astype(np.uint8) * 255)
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
) -> PipelineResult:
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
    return PipelineResult(
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


def unlink_failed_outputs(images: list[Path], output_dir: Path) -> None:
    for image_path in images:
        output_path = output_dir / build_binary_name(image_path)
        if output_path.exists():
            output_path.unlink()


def produced_file_names(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return sorted(p.name for p in output_dir.iterdir())


def pipeline_failure_message(
    label: str,
    result: PipelineResult,
    output_dir: Path,
    failures: dict[Path, str],
) -> str:
    details = [
        f"{label} failed with exit code {result.returncode}",
        f"expected output dir: {output_dir}",
        f"produced files: {produced_file_names(output_dir) or 'none'}",
    ]
    if result.returncode < 0:
        details.append(f"subprocess was terminated by signal {-result.returncode}")
    if failures:
        details.append(
            "invalid/missing outputs: "
            + "; ".join(f"{p.name}: {err}" for p, err in failures.items())
        )
    if not result.stdout and not result.stderr:
        details.append("subprocess produced no stdout/stderr")
    return "; ".join(details)


def process_stage_chunk(
    images: list[Path],
    stage_label: str,
    command_builder: CommandBuilder,
    output_dir: Path,
    attempt_root: Path,
    keep_intermediates: bool,
    remaining_retries: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_root.mkdir(parents=True, exist_ok=True)

    pending_failures = output_validation_failures(images, output_dir)
    pending_images = [image_path for image_path in images if image_path in pending_failures]
    if not pending_images:
        logger.info(
            "%s outputs already valid for %d image(s): %s",
            stage_label,
            len(images),
            format_image_names(images),
        )
        return

    unlink_failed_outputs(pending_images, output_dir)

    attempt_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{stage_label.lower().replace(' ', '_')}_attempt_",
            dir=attempt_root,
        )
    )
    input_dir = attempt_dir / "input"

    try:
        build_chunk_input_dir(input_dir, pending_images)
        result = run_pipeline(
            command_builder(input_dir, output_dir),
            f"{stage_label} pipeline",
        )
        output_failures = output_validation_failures(pending_images, output_dir)

        if result.returncode == 0 and not output_failures:
            return

        if result.returncode == -9 and not output_failures:
            logger.warning(
                "%s was killed with SIGKILL after producing all expected "
                "outputs for %d image(s); continuing",
                stage_label,
                len(pending_images),
            )
            return

        if not output_failures:
            raise RuntimeError(
                pipeline_failure_message(
                    stage_label,
                    result,
                    output_dir,
                    output_failures,
                )
            )

        retry_images = (
            [image_path for image_path in pending_images if image_path in output_failures]
            if output_failures else pending_images
        )

        if len(pending_images) == 1 or remaining_retries < 1:
            raise RuntimeError(
                pipeline_failure_message(
                    stage_label,
                    result,
                    output_dir,
                    output_failures,
                )
            )

        if len(retry_images) == 1:
            logger.warning(
                "%s failed for one image inside a larger chunk; retrying %s "
                "alone",
                stage_label,
                retry_images[0].name,
            )
            process_stage_chunk(
                retry_images,
                stage_label,
                command_builder,
                output_dir,
                attempt_root,
                keep_intermediates,
                remaining_retries - 1,
            )
            return

        midpoint = max(1, len(retry_images) // 2)
        logger.warning(
            "%s failed for %d image(s); retrying as %d and %d image chunk(s)",
            stage_label,
            len(retry_images),
            midpoint,
            len(retry_images) - midpoint,
        )
        process_stage_chunk(
            retry_images[:midpoint],
            stage_label,
            command_builder,
            output_dir,
            attempt_root,
            keep_intermediates,
            remaining_retries - 1,
        )
        process_stage_chunk(
            retry_images[midpoint:],
            stage_label,
            command_builder,
            output_dir,
            attempt_root,
            keep_intermediates,
            remaining_retries - 1,
        )
    finally:
        if not keep_intermediates:
            shutil.rmtree(attempt_dir, ignore_errors=True)


def process_stage_images(
    images: list[Path],
    stage_label: str,
    command_builder: CommandBuilder,
    output_dir: Path,
    attempt_root: Path,
    chunk_size: int,
    keep_intermediates: bool,
    max_retries: int,
) -> None:
    stage_chunks = list(chunked(images, chunk_size))
    total_chunks = len(stage_chunks)
    for idx, image_chunk in enumerate(stage_chunks, 1):
        logger.info(
            "%s chunk %d/%d: %d image(s): %s",
            stage_label,
            idx,
            total_chunks,
            len(image_chunk),
            format_image_names(image_chunk),
        )
        process_stage_chunk(
            image_chunk,
            stage_label,
            command_builder,
            output_dir,
            attempt_root,
            keep_intermediates,
            max_retries,
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
            temp_final_path = final_path.with_name(
                f".{final_path.name}.tmp{final_path.suffix}"
            )
            if temp_final_path.exists():
                temp_final_path.unlink()
            try:
                save_bilevel_tiff(union_mask, temp_final_path, xres, yres)
                os.replace(temp_final_path, final_path)
            finally:
                if temp_final_path.exists():
                    temp_final_path.unlink()
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
                    "merge the outputs so black wins per pixel."
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
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=1,
        help="Default number of source images per orchestrator chunk",
    )
    parser.add_argument(
        "--dplinknet-chunk-size",
        type=positive_int,
        default=None,
        help="Override source images per DP-LinkNet subprocess",
    )
    parser.add_argument(
        "--sbb-chunk-size",
        type=positive_int,
        default=None,
        help="Override source images per SBB subprocess",
    )
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
        default=10,
        help="Maximum split-retry depth for failed multi-image chunks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip source images whose final output already exists and is readable",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess images even when --resume would skip them",
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

    original_image_count = len(images)
    if args.resume and not args.force:
        skipped = [
            image_path
            for image_path in images
            if final_output_is_valid(image_path, output_dir)
        ]
        skipped_set = set(skipped)
        images = [image_path for image_path in images if image_path not in skipped_set]
        logger.info(
            "Resume enabled: skipping %d/%d image(s) with valid final outputs",
            len(skipped),
            original_image_count,
        )
        if not images:
            logger.info("No images require processing.")
            return

    temp_ctx = None
    if args.work_dir is None:
        if args.keep_intermediates:
            work_dir = Path(tempfile.mkdtemp(prefix="binarization_orchestrator_"))
        else:
            temp_ctx = tempfile.TemporaryDirectory(prefix="binarization_orchestrator_")
            work_dir = Path(temp_ctx.name)
    else:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    sbb_compat_existed = (work_dir / "sbb_model_compat").exists()
    sbb_model_for_runs = prepare_sbb_compat_model_once(sbb_model_dir, work_dir)
    cleanup_paths: list[Path] = []
    if (
        temp_ctx is None
        and sbb_model_for_runs != sbb_model_dir
        and not sbb_compat_existed
    ):
        cleanup_paths.append(sbb_model_for_runs)

    dp_chunk_size = args.dplinknet_chunk_size or args.chunk_size
    sbb_chunk_size = args.sbb_chunk_size or args.chunk_size
    group_chunk_size = max(dp_chunk_size, sbb_chunk_size)

    def build_dplinknet_command(input_dir: Path, output_dir_for_stage: Path) -> list[str]:
        command = [
            args.dplinknet_python,
            "-u",
            str(dplinknet_script),
            str(input_dir),
            str(output_dir_for_stage),
            str(dplinknet_weights_dir),
            "--dataset",
            args.dataset,
            "--model",
            args.model,
        ]
        if args.no_tta:
            command.append("--no-tta")
        if args.threshold is not None:
            command.extend(["--threshold", str(args.threshold)])
        return command

    def build_sbb_command(input_dir: Path, output_dir_for_stage: Path) -> list[str]:
        return [
            args.sbb_python,
            "-u",
            str(sbb_script),
            str(input_dir),
            str(output_dir_for_stage),
            str(sbb_model_for_runs),
        ]

    logger.info("Found %d input image(s) in '%s'", original_image_count, image_dir)
    logger.info("Processing %d image(s) in this run", len(images))
    logger.info("Final output directory: '%s'", output_dir)
    logger.info("Intermediate work directory: '%s'", work_dir)
    logger.info(
        "Chunk sizes: orchestrator=%d, DP-LinkNet=%d, SBB=%d",
        group_chunk_size,
        dp_chunk_size,
        sbb_chunk_size,
    )

    processed_count = 0
    try:
        image_chunks = list(chunked(images, group_chunk_size))
        for chunk_idx, image_chunk in enumerate(image_chunks, 1):
            chunk_root = Path(
                tempfile.mkdtemp(prefix=f"chunk_{chunk_idx:05d}_", dir=work_dir)
            )
            try:
                logger.info(
                    "Orchestrator chunk %d/%d: %d image(s): %s",
                    chunk_idx,
                    len(image_chunks),
                    len(image_chunk),
                    format_image_names(image_chunk),
                )
                dp_output_dir = chunk_root / "dplinknet_output"
                sbb_output_dir = chunk_root / "sbb_output"
                attempt_root = chunk_root / "attempts"

                process_stage_images(
                    image_chunk,
                    "DP-LinkNet",
                    build_dplinknet_command,
                    dp_output_dir,
                    attempt_root,
                    dp_chunk_size,
                    args.keep_intermediates,
                    args.max_retries,
                )
                process_stage_images(
                    image_chunk,
                    "SBB",
                    build_sbb_command,
                    sbb_output_dir,
                    attempt_root,
                    sbb_chunk_size,
                    args.keep_intermediates,
                    args.max_retries,
                )
                merge_outputs(image_chunk, dp_output_dir, sbb_output_dir, output_dir)
                processed_count += len(image_chunk)
            finally:
                if not args.keep_intermediates:
                    shutil.rmtree(chunk_root, ignore_errors=True)
    finally:
        if temp_ctx is not None and not args.keep_intermediates:
            temp_ctx.cleanup()
        elif temp_ctx is None and not args.keep_intermediates:
            for cleanup_path in cleanup_paths:
                shutil.rmtree(cleanup_path, ignore_errors=True)

    logger.info("Merged %d image(s) successfully.", processed_count)


if __name__ == "__main__":
    main()
