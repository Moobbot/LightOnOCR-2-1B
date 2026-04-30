#!/bin/bash

# ============================================================
# setup_env.sh
# Script cài đặt packages cho môi trường extract-pdf (Linux)
# Chạy: bash setup_env.sh
# ============================================================

# Dừng script nếu có lỗi
set -e

echo ""
echo "=== Activating extract-pdf environment ==="
# Tìm đường dẫn conda để load shell function
CONDA_PATH=$(conda info --base)
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate extract-pdf
else
    echo "[WARNING] Khong tim thay conda.sh. Dang thu activate truc tiep..."
    conda activate extract-pdf || echo "[ERROR] Khong the activate extract-pdf. Hay chay 'conda activate extract-pdf' truoc."
fi

echo ""
echo "=== Python info ==="
python3 --version || python --version
python3 -c "import sys; print('Executable:', sys.executable)"

echo ""
echo "=== Step 1: Install PyTorch (CUDA 12.1) ==="
echo "Neu khong co GPU, doi thanh: pip install torch"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Step 2: Install transformers from source (requires v5.0+) ==="
pip install git+https://github.com/huggingface/transformers.git

echo ""
echo "=== Step 3: Install other requirements === "
pip install Pillow pandas openpyxl pypdfium2

echo ""
echo "=== Step 4: Download and Extract Model weights ==="
MODEL_URL="https://github.com/Moobbot/LightOnOCR-2-1B/releases/download/v-1.0.0/model.zip"
MODEL_FILE="model.safetensors"

if [ ! -f "$MODEL_FILE" ]; then
    echo "[INFO] $MODEL_FILE khong tim thay. Dang chuan bi tai model..."
    if [ ! -f "model.zip" ]; then
        echo "[INFO] Dang tai model.zip tu GitHub..."
        if command -v wget >/dev/null 2>&1; then
            wget -O model.zip "$MODEL_URL"
        elif command -v curl >/dev/null 2>&1; then
            curl -L -o model.zip "$MODEL_URL"
        else
            echo "[ERROR] Khong tim thay wget hoac curl. Hay cai dat mot trong hai."
            exit 1
        fi
    else
        echo "[INFO] Da co file model.zip."
    fi
    
    echo "[INFO] Dang giai nen model.zip..."
    if command -v unzip >/dev/null 2>&1; then
        unzip -o model.zip
    else
        echo "[ERROR] Khong tim thay lenh 'unzip'. Hay cai dat unzip (sudo apt install unzip)."
        exit 1
    fi
    
    echo "[INFO] Dang xoa file model.zip sau khi giai nen..."
    rm model.zip
else
    echo "[INFO] Da co file $MODEL_FILE. Bo qua buoc tai model."
fi

echo ""
echo "=== Verification ==="
python3 -c "import torch, transformers, PIL, pandas, openpyxl; print('[OK] torch:', torch.__version__); print('[OK] transformers:', transformers.__version__); print('[OK] CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== Setup complete! ==="
