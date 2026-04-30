#!/bin/bash
set -e

# ============================================================
# LightOnOCR-2-1B Entrypoint Script
# Checks for model weights and downloads them if missing.
# ============================================================

MODEL_FILE="model.safetensors"
MODEL_URL="https://github.com/Moobbot/LightOnOCR-2-1B/releases/download/v-1.0.0/model.zip"

# Sử dụng MODEL_PATH từ environment, mặc định là /app/model
TARGET_DIR="${MODEL_PATH:-/app/model}"

echo "------------------------------------------------------------"
echo "LightOnOCR-2-1B Startup"
echo "Target directory: $TARGET_DIR"
echo "------------------------------------------------------------"

if [ ! -f "$TARGET_DIR/$MODEL_FILE" ]; then
    echo "[INFO] $MODEL_FILE not found in $TARGET_DIR."
    echo "[INFO] Initializing model download (approx. 2GB)..."
    
    mkdir -p "$TARGET_DIR"
    
    # Kiểm tra xem file zip đã có chưa (phòng trường hợp restart)
    if [ ! -f "/tmp/model.zip" ]; then
        echo "[INFO] Downloading model.zip from GitHub..."
        wget -O /tmp/model.zip "$MODEL_URL"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to download model weights."
            exit 1
        fi
    fi
    
    echo "[INFO] Extracting model.zip to $TARGET_DIR..."
    unzip -o /tmp/model.zip -d "$TARGET_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to extract model weights."
        exit 1
    fi
    
    echo "[INFO] Cleaning up..."
    rm -f /tmp/model.zip
    echo "[OK] Model weights initialized successfully."
else
    echo "[INFO] Model weights already present. Skipping download."
fi

echo "------------------------------------------------------------"
echo "Executing: $@"
echo "------------------------------------------------------------"

exec "$@"
