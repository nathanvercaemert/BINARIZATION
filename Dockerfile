FROM python:3.11-slim

ARG REPO_URL=https://github.com/nathanvercaemert/BINARIZATION.git
ARG REPO_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libvips \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" BINARIZATION

WORKDIR /opt/BINARIZATION

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    "numpy>=1.22" \
    "opencv-python>=4.5" \
    "pyvips>=2.2" \
    "tensorflow<2.13" \
    "sbb-binarization" && \
    python -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "/opt/BINARIZATION/orchestrate_binarization.py"]
