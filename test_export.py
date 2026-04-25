"""
test_export.py — Test parser + exporter với JSON có sẵn (không cần OCR).

Dùng:
    python test_export.py <json_file> [output_xlsx]

Ví dụ:
    python test_export.py ..\datasets\Trang000001.json
    python test_export.py ..\datasets\Trang000001.json outputs\test_out.xlsx
"""

import sys
import json
import os

sys.path.insert(0, ".")
from pipeline.table_parser import extract_structured_data
from pipeline.exporter import save_json, json_to_excel

def reparse_json(json_path: str, out_xlsx: str = None):
    # Đọc JSON gốc
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        filename = item.get("filename", "unknown")
        ocr_text = item.get("ocr_text", "")

        r = extract_structured_data(filename, ocr_text)

        n_tables = r["table_count"]
        n_texts  = len(r["text_lines"])
        n_kv     = len(r["kv_pairs"])

        print(f"\n{'='*50}")
        print(f"  File     : {filename}")
        print(f"  Tables   : {n_tables}")
        print(f"  Text lines: {n_texts}")
        print(f"  KV pairs : {n_kv}")

        if r["tables"]:
            t0 = r["tables"][0]
            print(f"  Headers  : {t0['headers']}")
            print(f"  Rows     : {len(t0['rows'])}")
            print(f"  Row[0]   : {t0['rows'][0] if t0['rows'] else 'empty'}")

        if r["text_lines"]:
            print(f"  Texts    :")
            for line in r["text_lines"][:5]:
                print(f"    • {line}")

        results.append(r)

    # Output paths
    stem = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs("outputs", exist_ok=True)

    if out_xlsx is None:
        out_xlsx = os.path.join("outputs", f"{stem}_reparsed.xlsx")

    out_json = os.path.join("outputs", f"{stem}_reparsed.json")

    save_json(results, out_json)
    json_to_excel(results, out_xlsx)

    print(f"\n{'='*50}")
    print(f"[OK] JSON  : {out_json}")
    print(f"[OK] Excel : {out_xlsx}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_export.py <json_file> [output_xlsx]")
        sys.exit(1)

    json_path = sys.argv[1]
    out_xlsx  = sys.argv[2] if len(sys.argv) > 2 else None
    reparse_json(json_path, out_xlsx)
