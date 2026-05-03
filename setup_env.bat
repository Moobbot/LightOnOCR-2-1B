@echo off
setlocal EnableExtensions
REM ============================================================
REM setup_env.bat
REM Script cai dat moi truong Conda cho LightOnOCR-2-1B (Windows)
REM Chay: .\setup_env.bat [--name <env>] [--python <version>] [--cpu|--gpu]
REM ============================================================

set "ENV_NAME=extract-pdf"
set "PYTHON_VERSION=3.10"
set "DEVICE_MODE=gpu"

:parse_args
if "%~1"=="" goto after_parse
if /I "%~1"=="--name" (
    if "%~2"=="" (
        echo [ERROR] Thieu gia tri sau --name
        exit /b 1
    )
    set "ENV_NAME=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--python" (
    if "%~2"=="" (
        echo [ERROR] Thieu gia tri sau --python
        exit /b 1
    )
    set "PYTHON_VERSION=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--cpu" (
    set "DEVICE_MODE=cpu"
    shift
    goto parse_args
)
if /I "%~1"=="--gpu" (
    set "DEVICE_MODE=gpu"
    shift
    goto parse_args
)

echo [ERROR] Tham so khong hop le: %~1
echo Dung: .\setup_env.bat [--name ^<env^>] [--python ^<version^>] [--cpu^|--gpu]
exit /b 1

:after_parse

echo.
echo === Config ===
echo ENV_NAME       = %ENV_NAME%
echo PYTHON_VERSION = %PYTHON_VERSION%
echo DEVICE_MODE    = %DEVICE_MODE%

echo.
echo === Ensure Conda environment exists ===
conda env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul
if errorlevel 1 (
    echo [INFO] Env %ENV_NAME% chua ton tai. Dang tao moi voi python=%PYTHON_VERSION%...
    conda create -n %ENV_NAME% -y python=%PYTHON_VERSION% pip
    if errorlevel 1 (
        echo [ERROR] Tao env that bai.
        exit /b 1
    )
)

echo.
echo === Activating %ENV_NAME% environment ===
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Khong the activate %ENV_NAME%. Chay 'conda init powershell' hoac mo Anaconda Prompt.
    exit /b 1
)

echo.
echo === Python info ===
python --version
python -c "import sys; print('Executable:', sys.executable)"

echo.
if /I "%DEVICE_MODE%"=="cpu" (
    echo === Step 1: Install PyTorch (CPU) ===
    pip install torch torchvision
) else (
    echo === Step 1: Install PyTorch (CUDA 12.1) ===
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 (
    echo [ERROR] Cai PyTorch that bai.
    exit /b 1
)

echo.
echo === Step 2: Install project requirements ===
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Cai requirements that bai.
    exit /b 1
)

echo.
echo === Step 3: Download and Extract Model weights ===
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
python -c "import torch, transformers, PIL, pandas, openpyxl, pypdfium2; print('[OK] torch:', torch.__version__); print('[OK] transformers:', transformers.__version__); print('[OK] CUDA available:', torch.cuda.is_available())"

echo.
echo === Setup complete! ===
echo De su dung: conda activate %ENV_NAME%
endlocal
