"""
pipeline/exporter.py

Export pipeline results to JSON and Excel.

Excel strategy:
  - Each unique set of column headers → its own sheet
  - A "KV_Pairs" sheet collects all key-value fields (filename | key | value)
  - An "OCR_Raw" sheet collects images where no table/kv was found

Sheet naming:
  - Sheet name = first 3 header names joined by "_", truncated to 31 chars
  - Duplicate base names get a numeric suffix (_1, _2, …)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_json(results: List[Dict[str, Any]], output_path: str) -> str:
    """Save results list to a JSON file. Returns the path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

_INVALID_SHEET_CHARS = re.compile(r"[\\/*?\[\]:]")


def _sanitize_sheet_name(name: str) -> str:
    """Strip chars invalid in Excel sheet names and enforce 31-char limit."""
    name = _INVALID_SHEET_CHARS.sub("", name).strip()
    return name[:31] if name else "Sheet"


def _make_sheet_name(headers: List[str]) -> str:
    """Generate a human-readable sheet name from column headers."""
    parts = [str(h)[:10] for h in headers[:3]]
    return _sanitize_sheet_name("_".join(parts)) or "Table"


def _unique_sheet_name(base: str, existing: Dict[str, Any]) -> str:
    """Return a unique sheet name by appending _N suffix if needed."""
    if base not in existing:
        return base
    i = 1
    while f"{base[:28]}_{i}" in existing:
        i += 1
    return f"{base[:28]}_{i}"


# ---------------------------------------------------------------------------
# JSON → Excel conversion
# ---------------------------------------------------------------------------

def json_to_excel(results: List[Dict[str, Any]], output_path: str) -> str:
    """
    Convert pipeline results to an Excel workbook.

    Sheet layout:
      - One sheet per unique table column structure
        (tables with identical headers land in the same sheet)
      - "KV_Pairs" sheet: all extracted key-value fields
      - "OCR_Raw" sheet: images with no parsed tables/KVs (fallback)

    Each data row includes a "filename" column as the first column.
    """
    # headers_tuple → sheet_name mapping
    headers_to_sheet: Dict[Tuple, str] = {}
    # sheet_name → list of row dicts
    sheets: Dict[str, List[Dict[str, Any]]] = {}

    kv_rows: List[Dict[str, str]] = []
    raw_rows: List[Dict[str, str]] = []

    for result in results:
        filename = result.get("filename", "unknown")
        tables: List[Dict[str, Any]] = result.get("tables", [])
        kv_pairs: Dict[str, str] = result.get("kv_pairs", {})
        ocr_text: str = result.get("ocr_text", "")

        # --- KV pairs ---
        for key, value in kv_pairs.items():
            kv_rows.append({"filename": filename, "field": key, "value": value})

        # --- Tables ---
        for table in tables:
            headers: List[str] = table.get("headers", [])
            rows: List[Dict[str, Any]] = table.get("rows", [])
            if not headers or not rows:
                continue

            headers_key = tuple(headers)

            if headers_key not in headers_to_sheet:
                base_name = _make_sheet_name(headers)
                sheet_name = _unique_sheet_name(base_name, sheets)
                headers_to_sheet[headers_key] = sheet_name
                sheets[sheet_name] = []

            sheet_name = headers_to_sheet[headers_key]
            for row in rows:
                enriched = {"filename": filename, **row}
                sheets[sheet_name].append(enriched)

        # --- Fallback: no tables and no KV pairs ---
        if not tables and not kv_pairs:
            raw_rows.append({"filename": filename, "ocr_text": ocr_text})

    # --- Build KV sheet ---
    if kv_rows:
        kv_sheet = _unique_sheet_name("KV_Pairs", sheets)
        sheets[kv_sheet] = kv_rows

    # --- Build Raw sheet ---
    if raw_rows:
        raw_sheet = _unique_sheet_name("OCR_Raw", sheets)
        sheets[raw_sheet] = raw_rows

    # --- Write workbook ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not sheets:
        # Nothing to write — create a placeholder
        sheets["Empty"] = [{"message": "No data extracted"}]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, rows in sheets.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # Ensure "filename" is always the first column
            if "filename" in df.columns:
                other_cols = [c for c in df.columns if c != "filename"]
                df = df[["filename"] + other_cols]
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return output_path
