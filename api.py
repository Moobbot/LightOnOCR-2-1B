import os
import shutil
import tempfile
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pipeline.lightonocr_common import process_uploaded_document
from pipeline.model import DEVICE, get_model
from pydantic import BaseModel

app = FastAPI(
    title="LightOnOCR-2-1B API",
    description="API for extracting text and tables from images and PDFs using LightOnOCR-2-1B model.",
    version="1.0.0",
)

_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
if _cors_origins_env == "*":
    _cors_allow_origins = ["*"]
else:
    _cors_allow_origins = [
        origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger("lightonocr.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OCR model once at startup, keep it cached for the app lifetime."""
    logger.info("Warming up OCR model at startup")
    get_model()
    yield


app.router.lifespan_context = lifespan


@app.get("/")
def root():
    return {"message": "LightOnOCR-2-1B API is running", "device": DEVICE.upper()}


@app.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    page_num: int = Form(1),
    prompt: str = Form("Extract all text and tables from this image."),
    temperature: float = Form(0.2),
    max_tokens: int = Form(4096),
):
    print("\n" + "="*80)
    print(f"[TRACE Step 1] api.py: extract_document called.")
    print(f"  - filename: {file.filename if file else 'None'}")
    print(f"  - page_num: {page_num}, temp: {temperature}, max_tokens: {max_tokens}")
    print("="*80 + "\n")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loaded, bundle = process_uploaded_document(
            file_input=file_path,
            page_num=page_num,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result_data = {}
        if bundle.json_str:
            try:
                result_data = json.loads(bundle.json_str)
            except Exception:
                pass

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
    except Exception as e:
        logger.exception(
            "Unhandled error in /extract (file=%s, page_num=%s, max_tokens=%s)",
            getattr(file, "filename", None),
            page_num,
            max_tokens,
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


class DownloadRequest(BaseModel):
    path: str


@app.post("/download")
def download_file(req: DownloadRequest):
    path = req.path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    if path.endswith(".json"):
        media_type = "application/json"
    elif path.endswith(".xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        media_type = "application/octet-stream"

    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 7861))

    print(f"\n{'=' * 50}")
    print("  LightOnOCR-2-1B — API Server")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  URL    : http://{host}:{port}")
    print(f"  Docs   : http://{host}:{port}/docs")
    print(f"{'=' * 50}\n")

    uvicorn.run(app, host=host, port=port)
