@echo off
REM ============================================================
REM setup_env.bat
REM Script cài đặt packages cho môi trường extract-pdf
REM Chạy: .\setup_env.bat
REM ============================================================

echo.
echo === Activating extract-pdf environment ===
call conda activate extract-pdf
if errorlevel 1 (
    echo [ERROR] Khong the activate extract-pdf. Chay 'conda init' truoc.
    pause
    exit /b 1
)

echo.
echo === Python info ===
python --version
python -c "import sys; print('Executable:', sys.executable)"

echo.
echo === Step 1: Install PyTorch (CUDA 12.1) ===
echo Neu khong co GPU, doi thanh: pip install torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo === Step 2: Install transformers from source (requires v5.0+) ===
pip install git+https://github.com/huggingface/transformers.git

echo.
echo === Step 3: Install other requirements ===
pip install Pillow pandas openpyxl

echo.
echo === Step 4: Download and Extract Model weights ===
set MODEL_URL=https://github.com/Moobbot/LightOnOCR-2-1B/releases/download/v-1.0.0/model.zip
set MODEL_FILE=model.safetensors

if not exist "%MODEL_FILE%" (
    echo [INFO] %MODEL_FILE% khong tim thay. Dang chuan bi tai model...
    if not exist "model.zip" (
        echo [INFO] Dang tai model.zip tu GitHub...
        powershell -Command "Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile 'model.zip'"
        if errorlevel 1 (
            echo [ERROR] Khong the tai model.zip.
            pause
            exit /b 1
        )
    ) else (
        echo [INFO] Da co file model.zip.
    )
    
    echo [INFO] Dang giai nen model.zip...
    powershell -Command "Expand-Archive -Path 'model.zip' -DestinationPath '.' -Force"
    if errorlevel 1 (
        echo [ERROR] Khong the giai nen model.zip.
        pause
        exit /b 1
    )
    
    echo [INFO] Dang xoa file model.zip sau khi giai nen...
    del model.zip
) else (
    echo [INFO] Da co file %MODEL_FILE%. Bo qua buoc tai model.
)

echo.
echo === Verification ===
python -c "import torch, transformers, PIL, pandas, openpyxl; print('[OK] torch:', torch.__version__); print('[OK] transformers:', transformers.__version__); print('[OK] CUDA available:', torch.cuda.is_available())"

echo.
echo === Setup complete! ===
pause
