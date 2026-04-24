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
echo === Verification ===
python -c "import torch, transformers, PIL, pandas, openpyxl; print('[OK] torch:', torch.__version__); print('[OK] transformers:', transformers.__version__); print('[OK] CUDA available:', torch.cuda.is_available())"

echo.
echo === Setup complete! ===
pause
