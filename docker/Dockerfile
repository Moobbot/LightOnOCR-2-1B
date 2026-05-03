# ─────────────────────────────────────────────────────────────────────────────
# LightOnOCR-2-1B — Dockerfile
#
# Build:
#   docker build -f docker/Dockerfile -t lightonocr:latest .
#
# Chạy (CPU):
#   docker run --rm -p 7861:7861 \
#     -v /path/to/weights:/app/model \
#     -e MODEL_PATH=/app/model \
#     -e LIGHTONOCR_DEVICE=cpu \
#     lightonocr:latest
#
# Chạy (GPU):
#   docker run --rm --gpus all -p 7861:7861 \
#     -v /path/to/weights:/app/model \
#     -e MODEL_PATH=/app/model \
#     -e LIGHTONOCR_DEVICE=gpu \
#     lightonocr:latest
#
# Model weights KHÔNG bake vào image — mount runtime qua volume.
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL maintainer="local-pipeline" \
      description="LightOnOCR-2-1B — local OCR pipeline"

# Tránh prompt interactive của apt
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 + libglib2.0 : cần cho Pillow/OpenCV (xử lý ảnh, PDF rendering)
# git                 : cần để cài transformers từ source
# wget + unzip        : dùng bởi entrypoint để tải model weights nếu chưa có
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY docker/requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── App code (không bao gồm model weights) ───────────────────────────────────
COPY pipeline/      ./pipeline/
COPY api.py         ./api.py
COPY demo.py        ./demo.py
COPY run.py         ./run.py
COPY test_export.py ./test_export.py

# Bundle model config/tokenizer assets into image. These are copied into
# MODEL_PATH by entrypoint when the runtime model volume only has weights.
COPY config.json             /app/model-bootstrap/config.json
COPY generation_config.json  /app/model-bootstrap/generation_config.json
COPY tokenizer.json          /app/model-bootstrap/tokenizer.json
COPY tokenizer_config.json   /app/model-bootstrap/tokenizer_config.json
COPY processor_config.json   /app/model-bootstrap/processor_config.json
COPY special_tokens_map.json /app/model-bootstrap/special_tokens_map.json
COPY added_tokens.json       /app/model-bootstrap/added_tokens.json

# ── Runtime scripts ──────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ── Default environment ───────────────────────────────────────────────────────
# Tất cả biến môi trường đều có thể override khi chạy container.
ENV MODEL_PATH=/app/model \
    LIGHTONOCR_DEVICE=auto \
    LIGHTONOCR_DTYPE=auto \
    API_HOST=0.0.0.0 \
    API_PORT=7861 \
    CORS_ALLOW_ORIGINS=* \
    LOG_LEVEL=INFO \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# ── Ports & Volumes ───────────────────────────────────────────────────────────
EXPOSE 7861

# /app/model   — model weights (mount read-only từ host)
# /app/outputs — kết quả xuất ra (mount read-write)
VOLUME ["/app/model", "/app/outputs"]

ENTRYPOINT ["/app/entrypoint.sh"]

# Mặc định chạy REST API server; override bằng command
CMD ["python", "api.py"]
