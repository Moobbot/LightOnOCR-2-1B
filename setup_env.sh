#!/bin/bash

# ============================================================
# setup_env.sh
# Script cai dat moi truong Conda cho LightOnOCR-2-1B (Linux/macOS)
# Chay: bash setup_env.sh [--name <env>] [--python <version>] [--cpu|--gpu]
# ============================================================

# Dừng script nếu có lỗi
set -e

ENV_NAME="extract-pdf"
PYTHON_VERSION="3.10"
DEVICE_MODE="gpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            ENV_NAME="${2:-}"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="${2:-}"
            shift 2
            ;;
        --cpu)
            DEVICE_MODE="cpu"
            shift
            ;;
        --gpu)
            DEVICE_MODE="gpu"
            shift
            ;;
        *)
            echo "[ERROR] Tham so khong hop le: $1"
            echo "Dung: bash setup_env.sh [--name <env>] [--python <version>] [--cpu|--gpu]"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Config ==="
echo "ENV_NAME       = $ENV_NAME"
echo "PYTHON_VERSION = $PYTHON_VERSION"
echo "DEVICE_MODE    = $DEVICE_MODE"

echo ""
echo "=== Activating $ENV_NAME environment ==="
# Tìm đường dẫn conda để load shell function
CONDA_PATH=$(conda info --base)
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "[INFO] Env $ENV_NAME chua ton tai. Dang tao moi voi python=$PYTHON_VERSION..."
        conda create -n "$ENV_NAME" -y "python=$PYTHON_VERSION" pip
    fi
    conda activate "$ENV_NAME"
else
    echo "[ERROR] Khong tim thay conda.sh. Hay chay script trong shell da duoc conda init."
    exit 1
fi

echo ""
echo "=== Python info ==="
python3 --version || python --version
python3 -c "import sys; print('Executable:', sys.executable)"

echo ""
if [[ "$DEVICE_MODE" == "cpu" ]]; then
    echo "=== Step 1: Install PyTorch (CPU) ==="
    pip install torch torchvision
else
    echo "=== Step 1: Install PyTorch (CUDA 12.1) ==="
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo ""
echo "=== Step 2: Install project requirements ==="
pip install -r requirements.txt

echo ""
echo "=== Step 3: Download and Extract Model weights ==="
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
python3 -c "import torch, transformers, PIL, pandas, openpyxl, pypdfium2; print('[OK] torch:', torch.__version__); print('[OK] transformers:', transformers.__version__); print('[OK] CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== Setup complete! ==="
echo "De su dung: conda activate $ENV_NAME"
read -p "Press Enter to exit"
