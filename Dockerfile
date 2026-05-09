FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TF_USE_LEGACY_KERAS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/BINARIZATION

RUN python3 -m venv /opt/venv && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    "numpy>=1.26" \
    "opencv-python>=4.5" \
    "pyvips>=2.2" \
    "pyvips-binary" \
    "tensorflow[and-cuda]==2.21.0" \
    "tf-keras==2.21.0" \
    "sbb-binarization" && \
    python -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY . /opt/BINARIZATION

ENTRYPOINT ["python", "/opt/BINARIZATION/orchestrate_binarization.py"]
