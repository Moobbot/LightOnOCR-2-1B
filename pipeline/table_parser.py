"""
pipeline/table_parser.py

Parse bảng từ OCR output text (hỗ trợ cả HTML table lẫn Markdown table).

Model LightOnOCR thường output HTML table format:
  <table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody></table>

Output mỗi ảnh:
{
    "filename": "...",
    "ocr_text": "<raw text>",
    "tables": [{"headers": [...], "rows": [{col: val}, ...]}],
    "text_lines": ["dòng text 1", "dòng text 2", ...],  # ngoài bảng
    "kv_pairs": {"field": "value"},
    "table_count": N,
}
"""

import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional


# ── HTML Table Parser ─────────────────────────────────────────────────────────

class _HTMLTableParser(HTMLParser):
    """Parse một HTML table block thành headers + rows."""

    def __init__(self):
        super().__init__()
        self.in_thead = False
        self.in_tbody = False
        self.in_tr = False
        self.in_th = False
        self.in_td = False
        self.current_cell: List[str] = []
        self.current_row: List[str] = []
        self.headers: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "thead":   self.in_thead = True
        elif tag == "tbody": self.in_tbody = True
        elif tag == "tr":
            self.in_tr = True
            self.current_row = []
        elif tag == "th":
            self.in_th = True
            self.current_cell = []
        elif tag == "td":
            self.in_td = True
            self.current_cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            self.in_tr = False
            if self.current_row:
                if self.in_thead or (not self.in_tbody and not self.headers):
                    self.headers = self.current_row[:]
                else:
                    self.rows.append(self.current_row[:])
            self.current_row = []
        elif tag == "th":
            self.in_th = False
            self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = []
        elif tag == "td":
            self.in_td = False
            self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = []

    def handle_data(self, data):
        if self.in_th or self.in_td:
            self.current_cell.append(data)


def _parse_html_tables(text: str) -> List[Dict[str, Any]]:
    """Tìm và parse tất cả HTML table trong text."""
    tables = []
    pattern = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

    for match in pattern.finditer(text):
        parser = _HTMLTableParser()
        parser.feed(match.group(0))

        if not parser.headers:
            continue

        rows_as_dicts = []
        for raw_row in parser.rows:
            # Pad/trim để khớp số cột
            while len(raw_row) < len(parser.headers):
                raw_row.append("")
            raw_row = raw_row[: len(parser.headers)]
            row_dict = {parser.headers[i]: raw_row[i] for i in range(len(parser.headers))}
            rows_as_dicts.append(row_dict)

        if rows_as_dicts:
            tables.append({"headers": parser.headers, "rows": rows_as_dicts})

    return tables


# ── Markdown Table Parser (fallback) ─────────────────────────────────────────

_MD_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n[ \t]*\|[-:| \t]+\|\n(?:[ \t]*\|[^\n]+\|\n?)*)",
    re.MULTILINE,
)


def _parse_markdown_table_block(block: str) -> Optional[Dict[str, Any]]:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    headers = [c.strip() for c in lines[0].strip("|").split("|") if c.strip()]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[: len(headers)]
        rows.append({headers[i]: cells[i] for i in range(len(headers))})
    return {"headers": headers, "rows": rows} if headers and rows else None


def _parse_markdown_tables(text: str) -> List[Dict[str, Any]]:
    tables = []
    for m in _MD_TABLE_RE.finditer(text):
        parsed = _parse_markdown_table_block(m.group(0))
        if parsed:
            tables.append(parsed)
    return tables


# ── Non-table text line extractor ─────────────────────────────────────────────

def _extract_text_lines(text: str) -> List[str]:
    """
    Trả về các dòng text nằm ngoài bảng (HTML hoặc markdown).
    Dọn sạch HTML tags, LaTeX, markdown formatting.
    """
    # Xóa HTML tables
    cleaned = re.sub(r"<table[^>]*>.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Xóa markdown tables
    cleaned = _MD_TABLE_RE.sub("", cleaned)
    # Xóa HTML tags còn sót
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Xóa LaTeX inline ($...$, $$...$$)
    cleaned = re.sub(r"\$\$.*?\$\$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\$[^$\n]+\$", "", cleaned)
    # Xóa markdown headings (#, ##, ...)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # Xóa markdown bold/italic
    cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cleaned)

    lines = []
    for line in cleaned.splitlines():
        line = line.strip()
        # Bỏ qua dòng trống hoặc quá ngắn
        if line and len(line) > 1:
            lines.append(line)

    return lines


# ── Key-Value extraction ─────────────────────────────────────────────────────

_KV_RE = re.compile(r"^\*{0,2}([^|*\n<>]+?)\*{0,2}\s*[:\uff1a]\s*(.+)$", re.MULTILINE)


def _extract_kv_pairs(text: str) -> Dict[str, str]:
    # Strip HTML + table blocks trước để tránh false positive
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = _MD_TABLE_RE.sub("", cleaned)
    kv: Dict[str, str] = {}
    for m in _KV_RE.finditer(cleaned):
        key = m.group(1).strip(" *#>")
        value = m.group(2).strip()
        if key and len(key) < 80:
            kv[key] = value
    return kv


# ── Public API ────────────────────────────────────────────────────────────────

def extract_structured_data(filename: str, ocr_text: str) -> Dict[str, Any]:
    """
    Parse toàn bộ OCR text thành cấu trúc dữ liệu.

    Ưu tiên: HTML tables → Markdown tables → fallback raw lines.
    """
    # 1. Parse bảng (HTML ưu tiên, fallback markdown)
    tables = _parse_html_tables(ocr_text)
    if not tables:
        tables = _parse_markdown_tables(ocr_text)

    # 2. Các dòng text nằm ngoài bảng
    text_lines = _extract_text_lines(ocr_text)

    # 3. Key-Value pairs (header/metadata fields)
    kv_pairs = _extract_kv_pairs(ocr_text)

    return {
        "filename": filename,
        "ocr_text": ocr_text,
        "tables": tables,
        "text_lines": text_lines,
        "kv_pairs": kv_pairs,
        "table_count": len(tables),
    }
