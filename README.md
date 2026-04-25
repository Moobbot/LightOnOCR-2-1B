# LightOnOCR-2-1B — Local Pipeline

> Bản README này mô tả pipeline **local** được xây dựng trên nền model **LightOnOCR-2-1B** của [LightOn AI](https://lighton.ai).
> Model gốc: [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)

---

## Tổng quan

Pipeline OCR cục bộ dùng để:
- Trích xuất văn bản từ **ảnh** (JPG, PNG, BMP, WEBP) và **file PDF**
- Tự động parse **bảng HTML/Markdown** và **cặp Key-Value**
- Xuất kết quả ra **JSON** và **Excel (.xlsx)**
- Chạy hoàn toàn **offline** trên máy local (GPU CUDA hoặc CPU)

---

## Cấu trúc dự án

```
LightOnOCR-2-1B/
│
├── pipeline/                   # Thư viện xử lý lõi
│   ├── __init__.py
│   ├── model.py                # Load model (singleton, auto device)
│   ├── ocr_engine.py           # OCR inference + blank page detection
│   ├── pdf_renderer.py         # Render PDF → PIL Image (pypdfium2)
│   ├── table_parser.py         # Parse HTML table / Markdown table / text lines / KV
│   └── exporter.py             # Xuất JSON và Excel (4 bước rõ ràng)
│
├── demo.py                     # Giao diện web Gradio (chạy local)
├── run.py                      # CLI batch processor
├── test_export.py              # Test nhanh: re-parse JSON → Excel
│
├── setup_env.bat               # Cài đặt môi trường Conda (Windows)
├── requirements.txt            # Danh sách thư viện Python
│
├── app.py                      # (HuggingFace Spaces — không dùng local)
└── [Model files]               # model.safetensors, tokenizer.json, config.json...
```

---

## Cài đặt môi trường

### Phương án A — Docker (khuyến nghị cho production)

**Yêu cầu:** Docker + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Build image (chỉ copy app code, không copy model weights)
docker build -f docker/Dockerfile -t lightonocr:latest .

# Chạy Web Demo
docker compose up demo
# → Mở http://localhost:7860

# Chạy CLI batch
docker compose run --rm batch --input /data --output-name result
```

> **Model weights** được mount từ thư mục hiện tại (`.`) vào `/weights` bên trong container — không bake vào image.

---

### Phương án B — Conda (local development)

### Yêu cầu
- Python 3.10+
- Conda (Miniconda / Anaconda)
- GPU NVIDIA với CUDA 12.x *(khuyến nghị, CPU cũng chạy được nhưng chậm)*

### Bước 1 — Tạo môi trường

```powershell
conda create -n extract-pdf python=3.10 -y
conda activate extract-pdf
```

### Bước 2 — Cài PyTorch (CUDA 12.1)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 3 — Cài các thư viện còn lại

```powershell
pip install -r requirements.txt
```

### Kiểm tra

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Sử dụng

### 1. Web UI (Gradio)

```powershell
conda activate extract-pdf
cd LightOnOCR-2-1B
python demo.py
```

Mở trình duyệt tại **http://localhost:7860**

**Tính năng:**
- Upload ảnh hoặc PDF (có slider chọn trang)
- OCR → hiển thị văn bản rendered (Markdown/bảng/LaTeX)
- Raw text, JSON cấu trúc, tải xuống `.json` + `.xlsx`

---

### 2. CLI Batch (xử lý hàng loạt)

```powershell
conda activate extract-pdf
cd LightOnOCR-2-1B

# Một ảnh
python run.py --input ..\datasets\Trang000001.jpg

# Một PDF (tất cả trang)
python run.py --input ..\datasets\document.pdf

# Thư mục (không đệ quy)
python run.py --input ..\datasets\ --output-dir outputs --output-name batch_01

# Thư mục đệ quy
python run.py --input ..\datasets\ --recursive --output-name all_results

# Tùy chỉnh token limit
python run.py --input ..\datasets\ --max-tokens 16384
```

**Output:** `outputs/<output-name>.json` và `outputs/<output-name>.xlsx`

#### Tham số CLI

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--input` | *(bắt buộc)* | File ảnh, PDF, hoặc thư mục |
| `--output-dir` | `outputs` | Thư mục lưu kết quả |
| `--output-name` | `result` | Tên file output (không có extension) |
| `--max-tokens` | `8192` | Giới hạn token mỗi trang |
| `--recursive` | False | Quét thư mục đệ quy |
| `--blank-threshold` | `0.99` | Ngưỡng phát hiện trang trắng |
| `--no-skip-blank` | False | Không bỏ qua trang trắng |

---

### 3. Test export (không cần OCR lại)

Dùng khi đã có JSON và muốn kiểm tra/xuất lại Excel:

```powershell
python test_export.py ..\datasets\Trang000001.json
python test_export.py ..\datasets\Trang000001.json outputs\custom_output.xlsx
```

---

## Cấu trúc dữ liệu output

### JSON (`OcrResult`)

```json
[
  {
    "filename": "Trang000001.jpg",
    "ocr_text": "<văn bản OCR thô>",
    "tables": [
      {
        "headers": ["SĐT", "HỌ TÊN", "NGÀY SINH", "..."],
        "rows": [
          { "SĐT": "1", "HỌ TÊN": "Nguyễn Văn A", "NGÀY SINH": "01/01/1990", "...": "" }
        ]
      }
    ],
    "text_lines": [
      "DANH SÁCH CẤP BẰNG TỐT NGHIỆP",
      "Số SVTN: 51 SV"
    ],
    "kv_pairs": {
      "Số SVTN": "51 SV"
    },
    "table_count": 1
  }
]
```

### Excel (`.xlsx`)

Mỗi **cấu trúc bảng duy nhất** → 1 sheet riêng:

| Kiểu dòng | Màu nền | Nội dung |
|---|---|---|
| **Header** | Xanh đậm (#1F4E79) | Tên các cột |
| **Text metadata** | Xanh nhạt (#EBF3FB, in nghiêng) | Dòng text ngoài bảng (tiêu đề, ghi chú...) |
| **Dữ liệu bảng** | Trắng | Giá trị từng ô trong bảng |

Ảnh không parse được bảng → sheet `OCR_Raw`.

---

## Kiến trúc pipeline

```
PIL Image
    │
    ▼  pipeline.ocr_engine.extract_text()
    │   ├── _prepare_inputs()    — tokenize + move to device
    │   └── model.generate()    — LightOnOCR inference
    │
    ▼  pipeline.table_parser.extract_structured_data()
    │   ├── _parse_html_tables()  — parse <table> HTML
    │   ├── _parse_markdown_tables() — fallback markdown
    │   ├── _extract_text_lines() — dòng ngoài bảng
    │   └── _extract_kv_pairs()   — key: value pairs
    │
    ├──▶ pipeline.exporter.save_json()       → .json
    └──▶ pipeline.exporter.json_to_excel()   → .xlsx
         ├── Bước 1: _group_by_structure()
         ├── Bước 2: _build_rows()
         ├── Bước 3: _write_sheet_to_workbook()
         └── Bước 4: wb.save()
```

---

## Lưu ý

- **Model chạy local hoàn toàn** — không cần internet sau khi tải weights
- **`app.py`** là bản gốc cho HuggingFace Spaces (dùng `spaces`, load từ HF Hub) — không dùng cho local
- PDF cần cài `pypdfium2`: `pip install pypdfium2`
- Temperature mặc định `0.2` trong demo UI, `0.0` (deterministic) trong CLI batch