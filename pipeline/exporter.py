"""
pipeline/exporter.py

Export kết quả pipeline ra JSON và Excel.

Excel layout:
  - Mỗi tập hợp headers duy nhất → 1 sheet riêng
  - Trong mỗi sheet:
      • Các dòng text (ngoài bảng) → 1 dòng, cột đầu = filename, cột 2 = nội dung text
      • Các dòng bảng → đúng theo cột header
  - Sheet "OCR_Raw": ảnh không có bảng và không có KV nào
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


# ── JSON ──────────────────────────────────────────────────────────────────────

def save_json(results: List[Dict[str, Any]], output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


# ── Excel helpers ─────────────────────────────────────────────────────────────

_INVALID_SHEET_CHARS = re.compile(r"[\\/*?\[\]:]")


def _sanitize(name: str) -> str:
    name = _INVALID_SHEET_CHARS.sub("", name).strip()
    return name[:31] if name else "Sheet"


def _make_sheet_name(headers: List[str]) -> str:
    parts = [str(h)[:10] for h in headers[:3]]
    return _sanitize("_".join(parts)) or "Table"


def _unique_name(base: str, existing: Dict) -> str:
    if base not in existing:
        return base
    i = 1
    while f"{base[:28]}_{i}" in existing:
        i += 1
    return f"{base[:28]}_{i}"


# ── Excel writer ──────────────────────────────────────────────────────────────

_HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
_TEXT_FILL   = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_TEXT_FONT   = Font(italic=True, color="444444")


def _write_sheet(ws, full_headers: List[str], rows: List[Dict[str, Any]]):
    """
    Ghi một sheet vào workbook.
    - full_headers: tên cột (đã bao gồm 'filename' ở đầu)
    - rows: list dict với key '_row_type' = 'text' hoặc 'data'
    """
    # Header row
    ws.append(full_headers)
    for cell in ws[1]:
        cell.font = _HEADER_FILL and _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in rows:
        row_type = row.get("_row_type", "data")
        if row_type == "text":
            # Dòng text: filename + nội dung text, các cột khác để trống
            values = [row.get("filename", ""), row.get("_text", "")]
            values += [""] * (len(full_headers) - 2)
            ws.append(values)
            # Style dòng text
            for cell in ws[ws.max_row]:
                cell.fill = _TEXT_FILL
                cell.font = _TEXT_FONT
        else:
            # Dòng dữ liệu bảng
            ws.append([row.get(h, "") for h in full_headers])


# ── Main export function ──────────────────────────────────────────────────────

def json_to_excel(results: List[Dict[str, Any]], output_path: str) -> str:
    """
    Chuyển results sang Excel.

    Mỗi sheet chứa 1 loại cấu trúc bảng:
      - Dòng xanh nhạt (in nghiêng): text ngoài bảng (metadata)
      - Dòng trắng: dữ liệu bảng
    """
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    # Map: headers_tuple → sheet_name
    headers_to_sheet: Dict[Tuple, str] = {}
    # Map: sheet_name → (full_headers, [rows])
    sheets: Dict[str, Any] = {}

    raw_rows: List[Dict] = []   # fallback khi không có bảng

    for result in results:
        filename   = result.get("filename", "unknown")
        tables     = result.get("tables", [])
        text_lines = result.get("text_lines", [])
        ocr_text   = result.get("ocr_text", "")

        if not tables:
            # Không parse được bảng → đưa vào sheet OCR_Raw
            raw_rows.append({"filename": filename, "ocr_text": ocr_text})
            continue

        for table in tables:
            headers: List[str] = table.get("headers", [])
            rows: List[Dict]   = table.get("rows", [])
            if not headers:
                continue

            hkey = tuple(headers)
            if hkey not in headers_to_sheet:
                sname = _unique_name(_make_sheet_name(headers), sheets)
                headers_to_sheet[hkey] = sname
                full_headers = ["filename", "_text_content"] + headers
                sheets[sname] = {"full_headers": full_headers, "rows": []}

            sname = headers_to_sheet[hkey]
            full_headers = sheets[sname]["full_headers"]

            # Thêm text lines trước (mỗi dòng = 1 row loại "text")
            for line in text_lines:
                sheets[sname]["rows"].append({
                    "_row_type": "text",
                    "filename": filename,
                    "_text": line,
                })

            # Thêm dữ liệu bảng
            for row in rows:
                enriched = {"_row_type": "data", "filename": filename, "_text_content": "", **row}
                sheets[sname]["rows"].append(enriched)

    # Ghi các sheet bảng
    for sname, info in sheets.items():
        ws = wb.create_sheet(title=sname[:31])
        _write_sheet(ws, info["full_headers"], info["rows"])

    # Sheet OCR_Raw (fallback)
    if raw_rows:
        ws = wb.create_sheet(title="OCR_Raw")
        ws.append(["filename", "ocr_text"])
        for cell in ws[1]:
            cell.font = _HEADER_FONT
            cell.fill = _HEADER_FILL
        for row in raw_rows:
            ws.append([row["filename"], row["ocr_text"]])

    if not wb.sheetnames:
        ws = wb.create_sheet("Empty")
        ws.append(["message"])
        ws.append(["No data extracted"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path
