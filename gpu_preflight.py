"""
Validate that container GPU resources are visible and usable.

This intentionally avoids loading any model files. It is meant to fail fast
before the full binarization pipeline spends time on image or model setup.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap


def print_status(message: str) -> None:
    print(f"GPU PREFLIGHT: {message}", flush=True)


def fail(message: str) -> None:
    print_status(f"FAIL - {message}")
    raise SystemExit(1)


def run_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        fail("nvidia-smi was not found in the container")

    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"nvidia-smi failed with exit code {completed.returncode}: {detail}")

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    if not first_line:
        fail("nvidia-smi returned no GPU records")

    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) >= 3:
        name, driver, memory_mb = parts[:3]
        print_status(
            f"nvidia-smi OK - {name}, driver {driver}, {memory_mb} MiB total"
        )
    else:
        print_status(f"nvidia-smi OK - {first_line}")


def run_python_probe(label: str, code: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"{label} check failed: {detail}")

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        fail(f"{label} check produced no output")

    for line in lines:
        print_status(line)


def run_pytorch_check(device_name: str) -> None:
    code = f"""
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("PyTorch reports CUDA unavailable")

        device = torch.device({device_name!r})
        if device.type != "cuda":
            raise SystemExit(f"PyTorch device must be CUDA, got {{device}}")

        device_count = torch.cuda.device_count()
        device_index = device.index if device.index is not None else 0
        if device_index < 0 or device_index >= device_count:
            raise SystemExit(
                f"PyTorch device {{device}} is out of range for "
                f"{{device_count}} CUDA device(s)"
            )

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        x = torch.ones((32, 32), device=device, dtype=torch.float32)
        y = torch.mm(x, x)
        torch.cuda.synchronize(device)
        result = float(y[0, 0].detach().cpu())
        if result != 32.0:
            raise SystemExit(
                f"PyTorch CUDA tensor test returned unexpected value {{result}}"
            )

        print(
            "PyTorch OK - CUDA available, "
            f"device {{device}}, {{torch.cuda.get_device_name(device_index)}}, "
            "tensor test passed",
            flush=True,
        )
    """
    run_python_probe("PyTorch", textwrap.dedent(code))


def run_tensorflow_check() -> None:
    code = """
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise SystemExit("TensorFlow reports no visible GPU devices")

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        with tf.device("/GPU:0"):
            x = tf.ones((32, 32), dtype=tf.float32)
            y = tf.linalg.matmul(x, x)
            result = float(y.numpy()[0, 0])

        if result != 32.0:
            raise SystemExit(
                f"TensorFlow GPU matmul test returned unexpected value {result}"
            )

        names = ", ".join(gpu.name for gpu in gpus)
        print(
            f"TensorFlow OK - {len(gpus)} GPU device(s) visible ({names}), "
            "matmul test passed",
            flush=True,
        )
    """
    run_python_probe("TensorFlow", textwrap.dedent(code))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate nvidia-smi, PyTorch CUDA, and TensorFlow GPU access."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for the PyTorch tensor check (default: cuda:0)",
    )
    args = parser.parse_args()

    run_nvidia_smi()
    run_pytorch_check(args.device)
    run_tensorflow_check()
    print_status("PASS - GPU resources are operational")


if __name__ == "__main__":
    main()
