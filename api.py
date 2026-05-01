"""
LightOnOCR-2-1B — FastAPI REST Server

Endpoints:
  GET  /          Health check + thông tin device
  POST /extract   OCR một file ảnh hoặc PDF
  POST /download  Tải file output (JSON / Excel)

Environment variables:
  API_HOST            : 0.0.0.0
  API_PORT            : 7861
  CORS_ALLOW_ORIGINS  : * hoặc danh sách origin cách nhau bởi dấu phẩy
  LOG_LEVEL           : DEBUG | INFO | WARNING | ERROR (mặc định: INFO)

Device / Model:
  MODEL_PATH          : Đường dẫn tới model weights
  LIGHTONOCR_DEVICE   : cpu | gpu | auto (mặc định: auto)
  LIGHTONOCR_DTYPE    : float32 | bfloat16 | auto (mặc định: auto)
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

import psutil
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.lightonocr_common import process_uploaded_document
from pipeline.model import DEVICE, DTYPE, LOCAL_MODEL_PATH, get_model

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "root": {"level": _LOG_LEVEL, "handlers": ["console"]},
        # Tắt log trùng lặp từ uvicorn access
        "loggers": {
            "uvicorn.access": {"level": "INFO", "propagate": True},
        },
    }
)

logger = logging.getLogger("lightonocr.api")


# ---------------------------------------------------------------------------
# Helpers giám sát
# ---------------------------------------------------------------------------


def _log_memory(label: str = "") -> None:
    """Ghi log dung lượng RAM tiến trình hiện tại."""
    try:
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info("Memory usage%s: %.1f MB", f" [{label}]" if label else "", rss_mb)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_cors_raw = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = (
    ["*"]
    if _cors_raw == "*"
    else [o.strip() for o in _cors_raw.split(",") if o.strip()]
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Warm-up model khi khởi động; log khi tắt."""
    logger.info(
        "Khởi động server | device=%s | dtype=%s | model=%s",
        DEVICE.upper(),
        DTYPE,
        LOCAL_MODEL_PATH,
    )
    _log_memory("before load")
    get_model()
    _log_memory("after load")
    logger.info("Model sẵn sàng — đang lắng nghe yêu cầu.")
    yield
    logger.info("Server đang tắt.")
    _log_memory("shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LightOnOCR-2-1B API",
    description=(
        "REST API cho LightOnOCR-2-1B — trích xuất text và bảng từ ảnh / PDF."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", summary="Health check")
def health_check():
    """Kiểm tra trạng thái server và thông tin device đang sử dụng."""
    return {
        "status": "ok",
        "message": "LightOnOCR-2-1B API đang chạy.",
        "device": DEVICE.upper(),
        "dtype": str(DTYPE),
        "model_path": LOCAL_MODEL_PATH,
    }


@app.post("/extract", summary="OCR một file ảnh hoặc PDF")
async def extract_document(
    file: UploadFile = File(..., description="File ảnh (jpg/png/...) hoặc PDF"),
    page_num: int = Form(1, description="Số trang cần xử lý (chỉ áp dụng với PDF, 1-indexed)"),
    prompt: str = Form(
        "Extract all text and tables from this image.",
        description="Câu lệnh hướng dẫn model",
    ),
    temperature: float = Form(0.2, ge=0.0, le=2.0, description="Độ ngẫu nhiên (0 = greedy)"),
    max_tokens: int = Form(4096, ge=1, le=8192, description="Giới hạn token đầu ra"),
):
    """Nhận file và trả về kết quả OCR có cấu trúc (text, bảng, JSON)."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Không có file được upload.")

    logger.info(
        "POST /extract | file=%s | page=%d | max_tokens=%d | temperature=%.2f",
        file.filename,
        page_num,
        max_tokens,
        temperature,
    )

    temp_dir = tempfile.mkdtemp()
    # Làm sạch tên file để tránh path traversal
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(temp_dir, safe_filename)

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        loaded, bundle = process_uploaded_document(
            file_input=file_path,
            page_num=page_num,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result_data: dict = {}
        if bundle.json_str:
            try:
                result_data = json.loads(bundle.json_str)
            except json.JSONDecodeError:
                logger.warning("Không parse được JSON từ bundle.json_str.")

        logger.info("POST /extract | status=%s", bundle.status)
        return {
            "status": bundle.status,
            "rendered_text": bundle.rendered_text,
            "raw_text": bundle.raw_text,
            "data": result_data,
            "json_path": bundle.json_path,
            "excel_path": bundle.excel_path,
            "file_info": {
                "source_name": loaded.source_name,
                "page_info": loaded.page_info,
                "total_pages": loaded.total_pages,
                "actual_page": loaded.actual_page,
                "is_pdf": loaded.is_pdf,
            },
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Lỗi không xác định trong POST /extract | file=%s | page=%d",
            file.filename,
            page_num,
        )
        raise HTTPException(status_code=500, detail="Lỗi xử lý nội bộ. Xem log server để biết chi tiết.")
    finally:
        # Dọn dẹp file tạm
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


class DownloadRequest(BaseModel):
    path: str


@app.post("/download", summary="Tải file output")
def download_file(req: DownloadRequest):
    """Tải file JSON hoặc Excel từ đường dẫn đã được trả về bởi /extract."""
    path = req.path
    if not path:
        raise HTTPException(status_code=400, detail="Đường dẫn file không được để trống.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File không tìm thấy: {path}")
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail="Đường dẫn không phải file.")

    ext = os.path.splitext(path)[1].lower()
    media_type_map = {
        ".json": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    logger.info("POST /download | path=%s", path)
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "7861"))

    banner = (
        f"\n{'=' * 52}\n"
        f"  LightOnOCR-2-1B — API Server\n"
        f"  Device     : {DEVICE.upper()}\n"
        f"  DType      : {DTYPE}\n"
        f"  Model Path : {LOCAL_MODEL_PATH}\n"
        f"  URL        : http://{host}:{port}\n"
        f"  Docs       : http://{host}:{port}/docs\n"
        f"{'=' * 52}\n"
    )
    print(banner)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=_LOG_LEVEL.lower(),
    )
