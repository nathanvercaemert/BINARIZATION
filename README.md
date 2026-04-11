# BINARIZATION

This repository can be packaged into a portable Docker image that:

- installs the runtime dependencies for the orchestrator and both model pipelines
- downloads this repository during the image build so the code, weights, and SBB model are baked into the image
- can then be run with bind-mounted input and output directories

## Dockerfile

Use the root-level [Dockerfile](/mnt/c/Users/natha/OneDrive/Desktop/BINARIZATION/Dockerfile). It clones this repository during image build so the code, weights, and model assets are baked into the image.

## Build

Build the image:

```bash
docker build -t binarization-orchestrator .
```

Build it from scratch with no cache:

```bash
docker build --no-cache -t binarization-orchestrator .
```

If you want to pin a different repository or ref at build time:

```bash
docker build \
  --build-arg REPO_URL=https://github.com/nathanvercaemert/BINARIZATION.git \
  --build-arg REPO_REF=main \
  -t binarization-orchestrator .
```

## Run

Bind-mount a host input directory and output directory, then pass those mount points to the orchestrator:

```bash
docker run --rm \
  -v /absolute/path/to/input:/input \
  -v /absolute/path/to/output:/output \
  binarization-orchestrator \
  /input /output
```

If you want to preserve intermediate pipeline outputs on the host as well:

```bash
docker run --rm \
  -v /absolute/path/to/input:/input \
  -v /absolute/path/to/output:/output \
  -v /absolute/path/to/work:/work \
  binarization-orchestrator \
  /input /output --work-dir /work --keep-intermediates
```

## Input Directory Requirements

The orchestrator expects:

- a flat input directory, not recursive
- at least one image file
- every image filename to start with `CROP`
- supported image extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.webp`

The output directory can be empty or missing; the orchestrator will create it if needed and will write merged `BINARY...` TIFF files there.
