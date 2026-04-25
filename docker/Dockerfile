# ─────────────────────────────────────────────────────────────────────────────
# LightOnOCR-2-1B — Dockerfile
#
# Build:
#   docker build -f docker/Dockerfile -t lightonocr:latest .
#
# Model weights KHÔNG bake vào image — mount runtime:
#   docker run --gpus all -p 7860:7860 \
#     -v /path/to/model/weights:/weights:ro \
#     -e MODEL_PATH=/weights \
#     lightonocr:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Metadata
LABEL maintainer="local-pipeline" \
      description="LightOnOCR-2-1B — local OCR pipeline"

# Tránh prompt interactive của apt
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 + libglib2.0 cần cho OpenCV/Pillow (PDF rendering)
# git cần để cài transformers từ source
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY docker/requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── App code (không bao gồm model weights) ───────────────────────────────────
COPY pipeline/   ./pipeline/
COPY demo.py     ./demo.py
COPY run.py      ./run.py
COPY test_export.py ./test_export.py

# ── Runtime config ────────────────────────────────────────────────────────────
# MODEL_PATH  : nơi mount model weights (override mặc định = thư mục hiện tại)
# GRADIO_HOST : 0.0.0.0 để expose ra ngoài container
# GRADIO_PORT : port Gradio (mặc định 7860)
ENV MODEL_PATH=/weights \
    GRADIO_HOST=0.0.0.0 \
    GRADIO_PORT=7860

EXPOSE 7860

# Volumes:
#   /weights  — model weights (mount read-only)
#   /app/outputs — kết quả batch (mount read-write)
VOLUME ["/weights", "/app/outputs"]

# Mặc định chạy web demo; override bằng --entrypoint hoặc command
CMD ["python", "demo.py"]
