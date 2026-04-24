"""
pipeline/table_parser.py

Parse markdown tables from OCR output text into structured JSON.

LightOnOCR outputs markdown-formatted text. This module:
1. Detects markdown table blocks (header | separator | rows)
2. Converts each table to a list of dicts keyed by column headers
3. Extracts all key-value pairs found outside tables (colon-separated lines)

Output per image:
{
    "filename": "...",
    "ocr_text": "<raw markdown>",
    "tables": [
        {
            "headers": ["col1", "col2", ...],
            "rows": [{"col1": "val", "col2": "val"}, ...]
        },
        ...
    ],
    "kv_pairs": {"field": "value", ...},   # non-table key:value lines
    "table_count": N,
}
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Markdown table detection
# A markdown table looks like:
#   | Header1 | Header2 |
#   |---------|---------|
#   | val1    | val2    |
# ---------------------------------------------------------------------------
_TABLE_BLOCK_RE = re.compile(
    r"(\|[^\n]+\|\n[ \t]*\|[-:| \t]+\|\n(?:[ \t]*\|[^\n]+\|\n?)*)",
    re.MULTILINE,
)


def _parse_table_block(block: str) -> Optional[Dict[str, Any]]:
    """Parse a single markdown table block into {headers, rows}."""
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return None

    # Row 0: headers
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    headers = [h for h in headers if h]  # remove empty edge cells

    # Row 1: separator — skip
    # Rows 2+: data rows
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        # Pad or trim to match header count
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[: len(headers)]
        row_dict = {headers[i]: cells[i] for i in range(len(headers))}
        rows.append(row_dict)

    if not headers or not rows:
        return None

    return {"headers": headers, "rows": rows}


# ---------------------------------------------------------------------------
# Key-Value pair extraction (lines like "Field Name: value")
# ---------------------------------------------------------------------------
_KV_RE = re.compile(r"^\*{0,2}([^|*\n]+?)\*{0,2}\s*[:\uff1a]\s*(.+)$", re.MULTILINE)


def _extract_kv_pairs(text: str) -> Dict[str, str]:
    """
    Extract simple key: value pairs from non-table lines.
    Handles both ASCII ':' and full-width '：'.
    Returns a dict. If the same key appears multiple times, last value wins.
    """
    kv: Dict[str, str] = {}
    # Remove markdown table blocks first to avoid false positives
    cleaned = _TABLE_BLOCK_RE.sub("", text)
    for match in _KV_RE.finditer(cleaned):
        key = match.group(1).strip(" *#>")
        value = match.group(2).strip()
        if key and len(key) < 80:  # sanity guard
            kv[key] = value
    return kv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_markdown_tables(text: str) -> List[Dict[str, Any]]:
    """Return a list of parsed table dicts from OCR markdown text."""
    tables = []
    for match in _TABLE_BLOCK_RE.finditer(text):
        parsed = _parse_table_block(match.group(0))
        if parsed:
            tables.append(parsed)
    return tables


def extract_structured_data(filename: str, ocr_text: str) -> Dict[str, Any]:
    """
    Main entry point: combine table parsing + KV extraction for one image.

    Returns:
    {
        "filename": str,
        "ocr_text": str,
        "tables": [ {headers, rows}, ... ],
        "kv_pairs": { key: value, ... },
        "table_count": int,
    }
    """
    tables = parse_markdown_tables(ocr_text)
    kv_pairs = _extract_kv_pairs(ocr_text)

    return {
        "filename": filename,
        "ocr_text": ocr_text,
        "tables": tables,
        "kv_pairs": kv_pairs,
        "table_count": len(tables),
    }
