#!/usr/bin/env python3
"""
run.py — LightOnOCR-2-1B Image-to-JSON/Excel Pipeline

Usage:
    python run.py --input <image_or_folder> [options]

Examples:
    # Single image
    python run.py --input samples/invoice.jpg

    # Folder of images (non-recursive)
    python run.py --input samples/ --output-dir results/

    # Folder, recursive, custom output name
    python run.py --input data/ --recursive --output-name batch_01

    # Increase token limit for dense documents
    python run.py --input data/ --max-tokens 16384
"""

import argparse
import os
import sys
import time
from pathlib import Path

from PIL import Image

from pipeline.model import get_model
from pipeline.ocr_engine import extract_text, is_blank_page
from pipeline.table_parser import extract_structured_data
from pipeline.exporter import save_json, json_to_excel

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(input_path: str, recursive: bool = False) -> list:
    """
    Collect all valid image paths from a file or directory.
    Returns a sorted list of absolute path strings.
    """
    p = Path(input_path).resolve()

    if p.is_file():
        if p.suffix.lower() in VALID_EXTENSIONS:
            return [str(p)]
        print(f"[ERROR] Unsupported file type: {p.suffix}")
        return []

    if p.is_dir():
        if recursive:
            files = [
                str(f)
                for f in p.rglob("*")
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
        else:
            files = [
                str(f)
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
        return sorted(files)

    print(f"[ERROR] Path not found: {input_path}")
    return []


def print_summary(results: list) -> None:
    """Print a brief processing summary to stdout."""
    total = len(results)
    with_tables = sum(1 for r in results if r.get("table_count", 0) > 0)
    with_kv = sum(1 for r in results if r.get("kv_pairs"))
    errors = sum(1 for r in results if r.get("error"))

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Images processed : {total}")
    print(f"  With tables      : {with_tables}")
    print(f"  With KV pairs    : {with_kv}")
    print(f"  Errors           : {errors}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LightOnOCR-2-1B — Extract structured data from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Image file or folder containing images",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Directory to save JSON and Excel results (default: ./outputs)",
    )
    parser.add_argument(
        "--output-name", "-n",
        default="result",
        help="Base filename for output files, without extension (default: result)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens to generate per image (default: 8192)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan input directory recursively",
    )
    parser.add_argument(
        "--skip-blank",
        action="store_true",
        default=True,
        help="Skip near-white/blank images (default: True)",
    )
    parser.add_argument(
        "--blank-threshold",
        type=float,
        default=0.99,
        help="White pixel ratio threshold for blank detection (default: 0.99)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Collect images ──────────────────────────────────────────────────────
    image_paths = collect_images(args.input, args.recursive)
    if not image_paths:
        print("[ERROR] No valid images found. Exiting.")
        sys.exit(1)

    print(f"\n{'=' * 55}")
    print(f"  LightOnOCR-2-1B Pipeline")
    print(f"{'=' * 55}")
    print(f"  Input       : {args.input}")
    print(f"  Images      : {len(image_paths)}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Recursive   : {args.recursive}")
    print(f"{'=' * 55}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model once ──────────────────────────────────────────────────────
    model, processor = get_model()

    # ── Process images ───────────────────────────────────────────────────────
    results = []
    total_start = time.time()

    for idx, img_path in enumerate(image_paths, start=1):
        filename = Path(img_path).name
        print(f"[{idx:>3}/{len(image_paths)}] {filename}")

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"         ↳ [SKIP] Cannot open: {e}")
            results.append({
                "filename": filename,
                "ocr_text": "",
                "tables": [],
                "kv_pairs": {},
                "table_count": 0,
                "error": f"Cannot open image: {e}",
            })
            continue

        # Blank detection
        if args.skip_blank and is_blank_page(image, threshold=args.blank_threshold):
            print(f"         ↳ [SKIP] Blank image detected")
            continue

        # OCR inference
        t0 = time.time()
        try:
            ocr_text = extract_text(
                model, processor, image, max_tokens=args.max_tokens
            )
        except Exception as e:
            print(f"         ↳ [ERROR] OCR failed: {e}")
            results.append({
                "filename": filename,
                "ocr_text": "",
                "tables": [],
                "kv_pairs": {},
                "table_count": 0,
                "error": f"OCR failed: {e}",
            })
            continue

        elapsed = time.time() - t0

        # Parse structured data
        structured = extract_structured_data(filename, ocr_text)
        results.append(structured)

        n_tables = structured["table_count"]
        n_kv = len(structured.get("kv_pairs", {}))
        print(
            f"         ↳ Done in {elapsed:.1f}s "
            f"| tables={n_tables} | kv_pairs={n_kv}"
        )

    total_elapsed = time.time() - total_start

    # ── Nothing to save ──────────────────────────────────────────────────────
    if not results:
        print("\n[WARNING] No results to save.")
        sys.exit(0)

    # ── Export JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    save_json(results, json_path)
    print(f"\n[✓] JSON saved  : {json_path}")

    # ── Export Excel ──────────────────────────────────────────────────────────
    excel_path = os.path.join(args.output_dir, f"{args.output_name}.xlsx")
    json_to_excel(results, excel_path)
    print(f"[✓] Excel saved : {excel_path}")

    print_summary(results)
    print(f"\n  Total time: {total_elapsed:.1f}s for {len(results)} image(s)")
    print()


if __name__ == "__main__":
    main()
