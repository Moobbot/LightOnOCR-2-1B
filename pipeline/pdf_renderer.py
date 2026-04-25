"""
pipeline/pdf_renderer.py

Render trang PDF thành PIL Image (dùng chung cho run.py và demo.py).

Phụ thuộc: pypdfium2 (optional)
  - Nếu chưa cài: pip install pypdfium2
  - Nếu thiếu, các hàm trả về [] và in cảnh báo
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False


# =============================================================================
# INPUT
# =============================================================================
# pdf_path    : str — đường dẫn file PDF
# page_num    : int — số trang (1-indexed)
# max_res     : int — độ phân giải tối đa (pixels)
# scale       : float — hệ số phóng to khi render

# =============================================================================
# LOGIC
# =============================================================================

def _render_single_page(
    page,
    max_res: int = 1540,
    scale: float = 2.77,
) -> Image.Image:
    """Render một page object của pypdfium2 ra PIL Image."""
    w, h = page.get_size()
    resize_factor = min(1.0, max_res / (w * scale), max_res / (h * scale))
    return page.render(scale=scale * resize_factor, rev_byteorder=True).to_pil()


# =============================================================================
# OUTPUT
# =============================================================================

def render_pdf_page(
    pdf_path: str,
    page_num: int = 1,
    max_res: int = 1540,
    scale: float = 2.77,
) -> Tuple[Image.Image, int, int]:
    """
    INPUT  : đường dẫn PDF, số trang (1-indexed)
    OUTPUT : (image: PIL.Image, total_pages: int, actual_page: int)

    Dùng cho demo.py (render 1 trang để preview + OCR).
    """
    if not HAS_PDFIUM:
        raise ImportError("pypdfium2 chưa cài. Chạy: pip install pypdfium2")

    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    idx = min(max(int(page_num) - 1, 0), total - 1)
    image = _render_single_page(pdf[idx], max_res, scale)
    pdf.close()
    return image, total, idx + 1


def render_all_pages(
    pdf_path: str,
    max_res: int = 1540,
    scale: float = 2.77,
) -> List[Tuple[Image.Image, str]]:
    """
    INPUT  : đường dẫn PDF
    OUTPUT : list of (image: PIL.Image, label: str)
             label = "{stem}_page_001.pdf", "{stem}_page_002.pdf", ...

    Dùng cho run.py (xử lý batch toàn bộ trang).
    """
    if not HAS_PDFIUM:
        print("  [WARN] pypdfium2 chưa cài — bỏ qua PDF. Chạy: pip install pypdfium2")
        return []

    pdf = pdfium.PdfDocument(pdf_path)
    stem = Path(pdf_path).stem
    pages = []

    for i in range(len(pdf)):
        image = _render_single_page(pdf[i], max_res, scale)
        label = f"{stem}_page_{i + 1:03d}.pdf"
        pages.append((image, label))

    pdf.close()
    return pages
