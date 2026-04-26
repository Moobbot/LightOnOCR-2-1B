#!/usr/bin/env python3
"""
run.py — LightOnOCR-2-1B Image & PDF to JSON/Excel Pipeline

Usage:
    python run.py --input <file_or_folder> [options]

Examples:
    # Single image
    python run.py --input datasets/test/BCDKT-rang-dong-1.jpg

    # Single PDF (all pages)
    python run.py --input datasets/0218.pdf

    # Folder (ảnh + PDF, non-recursive)
    python run.py --input datasets/ --output-dir outputs/

    # Folder recursive
    python run.py --input datasets/ --recursive --output-name batch_01

    # Tăng token limit cho tài liệu dày
    python run.py --input datasets/ --max-tokens 16384
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from pipeline.model import get_model
from pipeline.ocr_engine import extract_text, is_blank_page
from pipeline.pdf_renderer import render_all_pages, HAS_PDFIUM
from pipeline.table_parser import extract_structured_data
from pipeline.exporter import save_json, json_to_excel

# --------------------------------------------------------------------------
# Supported file types
# --------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


# --------------------------------------------------------------------------
# File collection
# --------------------------------------------------------------------------


def collect_inputs(input_path: str, recursive: bool = False) -> List[str]:
    """
    Collect all valid image + PDF paths from a file or directory.
    Returns sorted list of absolute path strings.
    """
    p = Path(input_path).resolve()

    if p.is_file():
        if p.suffix.lower() in ALL_EXTENSIONS:
            return [str(p)]
        print(f"[ERROR] Unsupported file type: {p.suffix}")
        return []

    if p.is_dir():
        rglob = p.rglob if recursive else lambda pat: p.glob(pat)
        files = sorted(
            str(f)
            for f in (p.rglob("*") if recursive else p.iterdir())
            if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS
        )
        return files

    print(f"[ERROR] Path not found: {input_path}")
    return []


# --------------------------------------------------------------------------
# Per-file processing
# --------------------------------------------------------------------------


def process_image(
    image: Image.Image,
    label: str,
    model,
    processor,
    max_tokens: int,
    blank_threshold: float,
    skip_blank: bool,
) -> dict | None:
    """
    Run blank detection + OCR + structured parsing on a single PIL image.
    Returns a result dict, or None if the image should be skipped.
    """
    if skip_blank and is_blank_page(image, threshold=blank_threshold):
        print(f"         ↳ [SKIP] Blank page")
        return None

    t0 = time.time()
    try:
        ocr_text = extract_text(model, processor, image, max_tokens=max_tokens)
    except Exception as e:
        print(f"         ↳ [ERROR] OCR failed: {e}")
        return {
            "filename": label,
            "ocr_text": "",
            "tables": [],
            "kv_pairs": {},
            "table_count": 0,
            "error": str(e),
        }

    elapsed = time.time() - t0
    structured = extract_structured_data(label, ocr_text)
    n_tables = structured["table_count"]
    n_kv = len(structured.get("kv_pairs", {}))
    print(f"         ↳ {elapsed:.1f}s | tables={n_tables} | kv_pairs={n_kv}")
    return structured


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def print_summary(results: list, total_elapsed: float) -> None:
    total = len(results)
    tables = sum(1 for r in results if r.get("table_count", 0) > 0)
    kv = sum(1 for r in results if r.get("kv_pairs"))
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = sum(1 for r in results if r.get("error"))

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Pages/images processed : {total}")
    print(f"  With tables            : {tables}")
    print(f"  With KV pairs          : {kv}")
    print(f"  Errors                 : {errors}")
    print(f"  Total time             : {total_elapsed:.1f}s")
    print("=" * 55)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightOnOCR-2-1B — Trích xuất dữ liệu từ ảnh và PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="File ảnh, PDF, hoặc thư mục"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs",
        help="Thư mục lưu kết quả (default: ./outputs)",
    )
    parser.add_argument(
        "--output-name",
        "-n",
        default="result",
        help="Tên file output không có extension (default: result)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens sinh ra mỗi trang/ảnh (default: 4096)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Quét thư mục đệ quy"
    )
    parser.add_argument(
        "--blank-threshold",
        type=float,
        default=0.99,
        help="Ngưỡng tỉ lệ pixel trắng để bỏ qua trang trắng (default: 0.99)",
    )
    parser.add_argument(
        "--no-skip-blank", action="store_true", help="Không bỏ qua trang trắng"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    skip_blank = not args.no_skip_blank

    # ── Collect inputs ────────────────────────────────────────────────────
    input_files = collect_inputs(args.input, args.recursive)
    if not input_files:
        print("[ERROR] Không tìm thấy file hợp lệ. Thoát.")
        sys.exit(1)

    n_imgs = sum(1 for f in input_files if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
    n_pdfs = sum(1 for f in input_files if Path(f).suffix.lower() in PDF_EXTENSIONS)

    print(f"\n{'=' * 55}")
    print(f"  LightOnOCR-2-1B Pipeline")
    print(f"{'=' * 55}")
    print(f"  Input       : {args.input}")
    print(f"  Files       : {n_imgs} ảnh | {n_pdfs} PDF")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Output name : {args.output_name}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Skip blank  : {skip_blank}")
    if not HAS_PDFIUM and n_pdfs > 0:
        print(f"  [WARN] pypdfium2 chưa cài — PDF sẽ bị bỏ qua!")
    print(f"{'=' * 55}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model once ───────────────────────────────────────────────────
    model, processor = get_model()

    # ── Process all files ─────────────────────────────────────────────────
    results = []
    total_start = time.time()
    file_total = len(input_files)

    for file_idx, file_path in enumerate(input_files, start=1):
        fname = Path(file_path).name
        suffix = Path(file_path).suffix.lower()

        print(f"\n[{file_idx:>3}/{file_total}] {fname}")

        # ── PDF: render each page ─────────────────────────────────────────
        if suffix in PDF_EXTENSIONS:
            try:
                pages = render_all_pages(file_path)
            except Exception as e:
                print(f"         ↳ [ERROR] Không mở được PDF: {e}")
                continue

            print(f"         ↳ {len(pages)} trang")
            for page_img, page_label in pages:
                print(f"           • {page_label}")
                r = process_image(
                    page_img,
                    page_label,
                    model,
                    processor,
                    args.max_tokens,
                    args.blank_threshold,
                    skip_blank,
                )
                if r:
                    results.append(r)

        # ── Image ─────────────────────────────────────────────────────────
        else:
            try:
                image = Image.open(file_path).convert("RGB")
            except Exception as e:
                print(f"         ↳ [SKIP] Không mở được ảnh: {e}")
                results.append(
                    {
                        "filename": fname,
                        "ocr_text": "",
                        "tables": [],
                        "kv_pairs": {},
                        "table_count": 0,
                        "error": str(e),
                    }
                )
                continue

            r = process_image(
                image,
                fname,
                model,
                processor,
                args.max_tokens,
                args.blank_threshold,
                skip_blank,
            )
            if r:
                results.append(r)

    total_elapsed = time.time() - total_start

    # ── Export ────────────────────────────────────────────────────────────
    if not results:
        print("\n[WARNING] Không có kết quả để lưu.")
        sys.exit(0)

    json_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    excel_path = os.path.join(args.output_dir, f"{args.output_name}.xlsx")

    save_json(results, json_path)
    print(f"\n[✓] JSON  : {json_path}")

    json_to_excel(results, excel_path)
    print(f"[✓] Excel : {excel_path}")

    print_summary(results, total_elapsed)
    print()


if __name__ == "__main__":
    main()
