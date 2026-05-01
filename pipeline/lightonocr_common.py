"""Shared helpers cho các entrypoint của LightOnOCR (API, demo, CLI).

Module này tập trung logic load tài liệu, chạy OCR, và export kết quả để
đảm bảo FastAPI server, Gradio demo, và các wrapper API đều hoạt động nhất quán.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from .exporter import save_json, json_to_excel
from .model import get_model
from .ocr_engine import extract_text, is_blank_page
from .pdf_renderer import HAS_PDFIUM, render_pdf_page
from .table_parser import extract_structured_data

logger = logging.getLogger("lightonocr.common")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoadedDocument:
    image: Image.Image
    source_name: str
    page_info: str
    total_pages: int = 1
    actual_page: int = 1
    is_pdf: bool = False


@dataclass(slots=True)
class OCRBundle:
    status: str
    rendered_text: str
    raw_text: str
    json_str: str
    json_path: str | None
    excel_path: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_file_path(file_input) -> str:
    """Chuẩn hoá file_input thành đường dẫn chuỗi."""
    if file_input is None:
        raise ValueError("Không có file được cung cấp.")
    if isinstance(file_input, str):
        return file_input
    if hasattr(file_input, "name"):
        return file_input.name
    return str(file_input)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_uploaded_document(file_input, page_num: int = 1) -> LoadedDocument:
    """Load ảnh/PDF và chuẩn hoá thành PIL Image.

    Args:
        file_input: Đường dẫn file (str) hoặc object có thuộc tính .name.
        page_num: Số trang cần render (1-indexed, chỉ áp dụng cho PDF).

    Returns:
        LoadedDocument chứa ảnh và thông tin file.

    Raises:
        RuntimeError: Nếu file là PDF nhưng pypdfium2 chưa được cài.
        FileNotFoundError: Nếu file không tồn tại.
    """
    file_path = _resolve_file_path(file_input)
    logger.debug("load_uploaded_document | path=%s | page=%d", file_path, page_num)

    if file_path.lower().endswith(".pdf"):
        if not HAS_PDFIUM:
            raise RuntimeError(
                "pypdfium2 chưa được cài. Chạy: pip install pypdfium2"
            )
        image, total_pages, actual_page = render_pdf_page(file_path, int(page_num))
        return LoadedDocument(
            image=image,
            source_name=Path(file_path).name,
            page_info=f"Trang {actual_page}/{total_pages}",
            total_pages=total_pages,
            actual_page=actual_page,
            is_pdf=True,
        )

    image = Image.open(file_path).convert("RGB")
    return LoadedDocument(
        image=image,
        source_name=Path(file_path).name,
        page_info=Path(file_path).name,
    )


def decode_base64_image(image_base64: str) -> Image.Image:
    """Giải mã chuỗi base64 thành RGB PIL.Image.

    Args:
        image_base64: Chuỗi base64 của ảnh.

    Returns:
        PIL.Image ở chế độ RGB.
    """
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data)).convert("RGB")


# ---------------------------------------------------------------------------
# OCR pipeline
# ---------------------------------------------------------------------------


def extract_ocr_from_image(
    image: Image.Image,
    source_name: str,
    prompt: str = "Extract all text and tables from this image.",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    do_sample: bool | None = None,
) -> OCRBundle:
    """Chạy OCR trên một ảnh và export kết quả có cấu trúc.

    Args:
        image: Ảnh PIL cần xử lý.
        source_name: Tên file nguồn (dùng để đặt tên output).
        prompt: Câu lệnh hướng dẫn model.
        temperature: Độ ngẫu nhiên khi sinh text (0 = greedy).
        max_tokens: Giới hạn số token sinh ra.
        do_sample: Bật sampling. Nếu None, tự suy ra từ temperature.

    Returns:
        OCRBundle chứa text, JSON và đường dẫn file export.
    """
    logger.info("extract_ocr_from_image | source=%s", source_name)

    if is_blank_page(image):
        logger.warning("Phát hiện trang trắng/rỗng — bỏ qua: %s", source_name)
        return OCRBundle(
            status="BLANK_PAGE",
            rendered_text="",
            raw_text="",
            json_str="{}",
            json_path=None,
            excel_path=None,
        )

    model, processor = get_model()
    sample_flag = (float(temperature) > 0) if do_sample is None else do_sample

    raw_text = extract_text(
        model,
        processor,
        image,
        prompt=prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        do_sample=sample_flag,
    )

    structured = extract_structured_data(source_name, raw_text)
    rendered_text = structured.get("markdown", raw_text) or raw_text

    preview_json = {
        "tables": structured["tables"],
        "text_lines": structured["text_lines"],
        "kv_pairs": structured["kv_pairs"],
    }

    temp_dir = tempfile.mkdtemp()
    stem = Path(source_name).stem
    json_path = Path(temp_dir) / f"{stem}.json"
    excel_path = Path(temp_dir) / f"{stem}.xlsx"

    json_output_path: str | None = None
    excel_output_path: str | None = None
    export_errors: list[str] = []

    try:
        json_output_path = save_json([structured], str(json_path))
    except Exception:
        logger.exception("JSON export thất bại cho '%s'.", source_name)
        export_errors.append("JSON export failed")

    try:
        excel_output_path = json_to_excel([structured], str(excel_path))
    except Exception:
        logger.exception("Excel export thất bại cho '%s'.", source_name)
        export_errors.append("Excel export failed")

    table_count = structured.get("table_count", 0)
    text_count = len(structured.get("text_lines", []))
    kv_count = len(structured.get("kv_pairs", {}))

    status_parts = [
        f"source={source_name}",
        f"tables={table_count}",
        f"text_lines={text_count}",
        f"kv_pairs={kv_count}",
    ]
    if export_errors:
        status_parts.append(f"errors={'; '.join(export_errors)}")
    status = "OK | " + " | ".join(status_parts)

    import json

    return OCRBundle(
        status=status,
        rendered_text=rendered_text,
        raw_text=raw_text,
        json_str=json.dumps(preview_json, ensure_ascii=False, indent=2),
        json_path=json_output_path,
        excel_path=excel_output_path,
    )


def process_uploaded_document(
    file_input,
    page_num: int,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[LoadedDocument, OCRBundle]:
    """Load file và chạy OCR trong một lần gọi.

    Args:
        file_input: Đường dẫn file hoặc object Gradio/FastAPI.
        page_num: Số trang cần xử lý (chỉ áp dụng cho PDF).
        prompt: Câu lệnh hướng dẫn model.
        temperature: Độ ngẫu nhiên khi sinh text.
        max_tokens: Giới hạn số token sinh ra.

    Returns:
        Tuple (LoadedDocument, OCRBundle).
    """
    logger.info(
        "process_uploaded_document | file=%s | page=%d | max_tokens=%d",
        file_input,
        page_num,
        max_tokens,
    )
    loaded = load_uploaded_document(file_input, page_num)
    bundle = extract_ocr_from_image(
        loaded.image,
        loaded.source_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if bundle.status == "BLANK_PAGE":
        bundle = OCRBundle(
            status=bundle.status,
            rendered_text="",
            raw_text="",
            json_str="{}",
            json_path=None,
            excel_path=None,
        )
    return loaded, bundle
