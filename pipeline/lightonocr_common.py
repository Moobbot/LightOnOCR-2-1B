"""Shared helpers for LightOnOCR UI/API entrypoints.

This module keeps the document loading, OCR execution, and export logic in one
place so the FastAPI server, Gradio demo, and any API wrapper stay aligned.
"""

from __future__ import annotations

import base64
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


def _resolve_file_path(file_input) -> str:
    if file_input is None:
        raise ValueError("No file provided")
    if isinstance(file_input, str):
        return file_input
    if hasattr(file_input, "name"):
        return file_input.name
    return str(file_input)


def load_uploaded_document(file_input, page_num: int = 1) -> LoadedDocument:
    """Load an uploaded image/PDF and normalize it to a PIL image."""
    file_path = _resolve_file_path(file_input)

    if file_path.lower().endswith(".pdf"):
        if not HAS_PDFIUM:
            raise RuntimeError("pypdfium2 is required to render PDF files")
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
    """Decode a base64 image payload into RGB PIL.Image."""
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data)).convert("RGB")


def extract_ocr_from_image(
    image: Image.Image,
    source_name: str,
    prompt: str = "Extract all text and tables from this image.",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    do_sample: bool | None = None,
) -> OCRBundle:
    """Run OCR on a single image and export structured results."""
    if is_blank_page(image):
        return OCRBundle(
            status="⚠️ Ảnh trắng/rỗng — bỏ qua.",
            rendered_text="",
            raw_text="",
            json_str="{}",
            json_path=None,
            excel_path=None,
        )

    model, processor = get_model()
    sample_flag = float(temperature) > 0 if do_sample is None else do_sample

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
    json_path = Path(temp_dir) / f"{Path(source_name).stem}.json"
    excel_path = Path(temp_dir) / f"{Path(source_name).stem}.xlsx"

    json_output_path = None
    excel_output_path = None
    export_errors = []

    try:
        json_output_path = save_json([structured], str(json_path))
    except Exception as exc:
        export_errors.append(f"JSON export failed: {exc}")

    try:
        excel_output_path = json_to_excel([structured], str(excel_path))
    except Exception as exc:
        export_errors.append(f"Excel export failed: {exc}")

    status = (
        f"✅ {source_name} | bảng={structured['table_count']} | "
        f"text={len(structured.get('text_lines', []))} | kv={len(structured.get('kv_pairs', {}))}"
    )
    if export_errors:
        status = f"{status} | {'; '.join(export_errors)}"

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
    """Load an uploaded file and run OCR in one pass."""
    loaded = load_uploaded_document(file_input, page_num)
    bundle = extract_ocr_from_image(
        loaded.image,
        loaded.source_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if bundle.status.startswith("⚠️ Ảnh trắng/rỗng"):
        bundle = OCRBundle(
            status=bundle.status,
            rendered_text="",
            raw_text="",
            json_str="{}",
            json_path=None,
            excel_path=None,
        )
    return loaded, bundle
