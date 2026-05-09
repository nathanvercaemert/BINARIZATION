# BINARIZATION

This repository can be packaged into a portable Docker image that:

- installs the runtime dependencies for the orchestrator and both model pipelines
- copies the current checkout during the image build so the code, weights, and SBB model are baked into the image
- can then be run on a GPU-enabled Docker host with bind-mounted input and output directories

## Dockerfile

Use the root-level [Dockerfile](/mnt/c/Users/natha/OneDrive/Desktop/BINARIZATION/Dockerfile). It copies the current checkout during image build so the code, weights, and model assets are baked into the image. Make sure Git LFS files are present locally before building.

## Build

Build the image:

```bash
docker build -t binarization-orchestrator:gpu .
```

Build it from scratch with no cache:

```bash
docker build --no-cache -t binarization-orchestrator:gpu .
```

If you want to build a different ref, check out that ref and run `git lfs pull`
before building.

## Run

Run the GPU preflight check first. This loads no model files and does not
process images:

```bash
docker run --rm --gpus all binarization-orchestrator:gpu --preflight-only
```

Bind-mount a host input directory and output directory, then pass those mount points to the orchestrator:

```bash
docker run --rm --gpus all -v /mnt/PRIMARY/PRIMARY/BINARIZATION/working_slim:/input -v /mnt/PRIMARY/PRIMARY/BINARIZATION/binary_slim:/output binarization-orchestrator:gpu /input /output
```

```bash
docker run --rm --gpus all -v "C:\Users\natha\OneDrive\Desktop\TEST_IMAGE_PROCESSING\working_slim":/input -v "C:\Users\natha\OneDrive\Desktop\TEST_IMAGE_PROCESSING\binary_slim":/output binarization-orchestrator:gpu /input /output --resume
```

If you want to preserve intermediate pipeline outputs on the host as well:

```bash
docker run --rm --gpus all \
  -v /absolute/path/to/input:/input \
  -v /absolute/path/to/output:/output \
  -v /absolute/path/to/work:/work \
  binarization-orchestrator:gpu \
  /input /output --work-dir /work --keep-intermediates
```

For very large batches or very large source images, keep the SBB subprocess
small so TensorFlow memory is released between images:

```bash
docker run --rm --gpus all \
  -v /absolute/path/to/input:/input \
  -v /absolute/path/to/output:/output \
  -v /absolute/path/to/work:/work \
  binarization-orchestrator:gpu \
  /input /output --work-dir /work --sbb-chunk-size 1 --resume
```

The orchestrator processes full-resolution source images through the same
DP-LinkNet and SBB scripts; chunking only changes subprocess boundaries and
does not resize, downsample, or otherwise alter model inference.

To move the image to another machine:

```bash
docker save binarization-orchestrator:gpu -o binarization-orchestrator-gpu.tar
docker load -i binarization-orchestrator-gpu.tar
```

## Input Directory Requirements

The orchestrator expects:

- a flat input directory, not recursive
- at least one image file
- every image filename to start with `CROP`
- supported image extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.webp`

The output directory can be empty or missing; the orchestrator will create it if needed and will write merged `BINARY...` TIFF files there. The merge is pixelwise with black winning, so a pixel is black in the final output if either model marks it black.
