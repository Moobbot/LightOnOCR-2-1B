#!/bin/bash
set -e

# ============================================================
# LightOnOCR-2-1B Entrypoint Script
# Checks for required model files and downloads if missing.
# ============================================================

MODEL_URL="https://github.com/Moobbot/LightOnOCR-2-1B/releases/download/v-1.0.0/model.zip"

# Sử dụng MODEL_PATH từ environment, mặc định là /app/model
TARGET_DIR="${MODEL_PATH:-/app/model}"
BOOTSTRAP_DIR="/app/model-bootstrap"

# Các file bắt buộc phải có (và không được rỗng)
REQUIRED_FILES=(
    "model.safetensors"
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "processor_config.json"
    "special_tokens_map.json"
)

# Các file JSON cần kiểm tra tính hợp lệ (parse được)
JSON_FILES=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "processor_config.json"
    "special_tokens_map.json"
)

echo "------------------------------------------------------------"
echo "LightOnOCR-2-1B Startup"
echo "Target directory: $TARGET_DIR"
echo "------------------------------------------------------------"

# ── Bổ sung file cấu hình/tokenizer từ image (nếu thiếu) ───────────────────
# Model zip có thể chỉ chứa model.safetensors; các file JSON được bundle sẵn
# trong image để tránh crash khi transformers load generation config/tokenizer.
bootstrap_model_files() {
    if [ ! -d "$BOOTSTRAP_DIR" ]; then
        return 0
    fi

    mkdir -p "$TARGET_DIR"
    for f in "${REQUIRED_FILES[@]}"; do
        src="$BOOTSTRAP_DIR/$f"
        dst="$TARGET_DIR/$f"
        if [ ! -f "$dst" ] && [ -f "$src" ]; then
            echo "[INFO] Bổ sung file thiếu từ bootstrap: $f"
            cp -f "$src" "$dst"
        fi
    done
}

bootstrap_model_files

# ── Kiểm tra đủ file và không rỗng ──────────────────────────
_needs_download=0

if [ "${FORCE_DOWNLOAD:-0}" = "1" ]; then
    echo "[INFO] FORCE_DOWNLOAD=1 — Bỏ qua kiểm tra, tải lại model."
    _needs_download=1
fi

if [ "$_needs_download" = "0" ]; then
    for f in "${REQUIRED_FILES[@]}"; do
        fpath="$TARGET_DIR/$f"
        if [ ! -f "$fpath" ]; then
            echo "[WARN] File bắt buộc không tìm thấy: $f"
            _needs_download=1
            break
        fi
        # Kiểm tra file không rỗng
        if [ ! -s "$fpath" ]; then
            echo "[WARN] File bị rỗng (0 bytes): $f — cần tải lại."
            _needs_download=1
            break
        fi
    done
fi

# ── Kiểm tra JSON hợp lệ (phát hiện file corrupt) ───────────
if [ "$_needs_download" = "0" ]; then
    for f in "${JSON_FILES[@]}"; do
        fpath="$TARGET_DIR/$f"
        if [ -f "$fpath" ]; then
            if ! python3 -c "import json, sys; json.load(open('$fpath'))" 2>/dev/null; then
                echo "[WARN] File JSON bị corrupt: $f — cần tải lại."
                _needs_download=1
                break
            fi
        fi
    done
fi

if [ "$_needs_download" = "0" ]; then
    echo "[INFO] Tất cả file model đã có. Bỏ qua download."
else
    echo "[INFO] Khởi tạo tải model (~2GB từ GitHub)..."
    mkdir -p "$TARGET_DIR"

    # Dùng file zip tạm nếu chưa có (tránh tải lại khi restart)
    if [ ! -f "/tmp/model.zip" ] || [ ! -s "/tmp/model.zip" ]; then
        echo "[INFO] Đang tải model.zip..."
        wget --show-progress -O /tmp/model.zip "$MODEL_URL"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Tải model thất bại."
            exit 1
        fi
    else
        echo "[INFO] Dùng lại /tmp/model.zip đã có."
    fi

    echo "[INFO] Giải nén vào $TARGET_DIR..."
    unzip -o /tmp/model.zip -d "$TARGET_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Giải nén thất bại."
        exit 1
    fi

    rm -f /tmp/model.zip
    echo "[OK] Model đã sẵn sàng."

    # Xác nhận lại sau khi giải nén
    bootstrap_model_files
    for f in "${REQUIRED_FILES[@]}"; do
        fpath="$TARGET_DIR/$f"
        if [ ! -f "$fpath" ] || [ ! -s "$fpath" ]; then
            echo "[ERROR] File vẫn thiếu hoặc rỗng sau khi giải nén: $f"
            echo "[ERROR] Kiểm tra nội dung model.zip trên GitHub."
            exit 1
        fi
    done
fi

echo "------------------------------------------------------------"
echo "Executing: $@"
echo "------------------------------------------------------------"

exec "$@"
