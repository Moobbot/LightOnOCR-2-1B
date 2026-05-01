# LightOnOCR-2-1B — Local Pipeline

> Pipeline OCR cục bộ xây dựng trên model **LightOnOCR-2-1B** của [LightOn AI](https://lighton.ai).  
> Model gốc: [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)

---

## Tổng quan

Pipeline OCR cục bộ, chạy **hoàn toàn offline**, dùng để:

- Trích xuất văn bản từ **ảnh** (JPG, PNG, BMP, WEBP) và **file PDF**
- Tự động parse **bảng HTML/Markdown** và **cặp Key-Value**
- Xuất kết quả ra **JSON** và **Excel (.xlsx)**
- Chạy trên **GPU NVIDIA (CUDA)** hoặc **CPU**

---

## Cấu trúc dự án

```
LightOnOCR-2-1B/
│
├── pipeline/                   # Thư viện xử lý lõi
│   ├── __init__.py
│   ├── model.py                # Load model singleton (env-driven device/dtype)
│   ├── ocr_engine.py           # OCR inference + blank page detection
│   ├── pdf_renderer.py         # Render PDF → PIL Image (pypdfium2)
│   ├── lightonocr_common.py    # Shared helpers (load, run, export)
│   ├── table_parser.py         # Parse HTML/Markdown table, text lines, KV
│   └── exporter.py             # Xuất JSON và Excel
│
├── api.py                      # FastAPI REST API server (port 7861)
├── demo.py                     # Gradio web UI (port 7860)
├── run.py                      # CLI batch processor
├── test_export.py              # Test re-parse JSON → Excel
│
├── docker/
│   ├── Dockerfile              # Image definition
│   ├── entrypoint.sh           # Auto-download model weights nếu chưa có
│   └── requirements-docker.txt
│
├── docker-compose.yml          # Orchestration (api / demo / batch)
├── .env.example                # Mẫu biến môi trường
├── requirements.txt            # Thư viện cho chạy local (Conda)
├── setup_env.bat               # Cài đặt môi trường Conda (Windows)
├── setup_env.sh                # Cài đặt môi trường Conda (Linux)
└── [Model files]               # model.safetensors, tokenizer.json, config.json...
```

---

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MODEL_PATH` | _(thư mục hiện tại)_ | Đường dẫn tới model weights |
| `LIGHTONOCR_DEVICE` | `auto` | `cpu` / `gpu` / `auto` |
| `LIGHTONOCR_DTYPE` | `auto` | `float32` / `bfloat16` / `auto` |
| `API_HOST` | `0.0.0.0` | Host bind của FastAPI |
| `API_PORT` | `7861` | Port của FastAPI |
| `CORS_ALLOW_ORIGINS` | `*` | Origin được phép CORS |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

> **Ghi chú về device/dtype:**
> - `LIGHTONOCR_DEVICE=auto` → tự chọn GPU nếu có, ngược lại CPU.
> - `LIGHTONOCR_DTYPE=auto` → `bfloat16` trên GPU (tiết kiệm VRAM ~50%), `float32` trên CPU.
> - Khi chạy CPU, model chiếm khoảng **~4.5 GB RAM** (float32). Đảm bảo Docker Desktop được cấp đủ RAM.

Sao chép file `.env.example` thành `.env` và chỉnh sửa theo nhu cầu:

```bash
cp .env.example .env
```

---

## Cài đặt

### Phương án A — Docker (khuyến nghị)

**Yêu cầu:** Docker Desktop (hoặc Docker Engine + Compose plugin).

Dự án cung cấp **hai file Compose** tương ứng với hai chế độ chạy:

| File | Chế độ | Yêu cầu |
|---|---|---|
| `docker-compose.yml` | **GPU** (mặc định) | nvidia-container-toolkit |
| `docker-compose.cpu.yml` | **CPU** (override) | Không cần GPU |

---

#### 🟢 Chạy với GPU (mặc định)

Yêu cầu [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) đã cài trên host.

```bash
# API server
docker compose up -d --build

# Gradio UI
docker compose --profile demo up -d demo

# Batch CLI
docker compose --profile batch run --rm batch \
  --input /data --output-name result
```

> Model chiếm ~2.3 GB VRAM (bfloat16). Nhanh hơn CPU ~5–10 lần.

---

#### 🔵 Chạy với CPU (không cần GPU)

```bash
# API server
docker compose -f docker-compose.cpu.yml up -d --build

# Gradio UI
docker compose -f docker-compose.cpu.yml --profile demo up -d demo

# Batch CLI
docker compose -f docker-compose.cpu.yml \
  --profile batch run --rm batch \
  --input /data --output-name result
```

> ⚠️ **Yêu cầu tài nguyên bắt buộc (CPU mode)**
>
> Model LightOnOCR-2-1B chiếm RAM rất lớn khi chạy trên CPU:
>
> | Thành phần | RAM |
> |---|---|
> | Model weights (float32) | ~4.6 GB |
> | KV cache (4096 tokens) | ~1.8 GB |
> | Input image tensors | ~0.3 GB |
> | OS + Docker overhead | ~0.5 GB |
> | **Tổng cần tối thiểu** | **~7.2 GB** |
>
> → Docker Desktop **phải được cấp ≥ 12 GB RAM**. Nếu thiếu, container sẽ **tự động restart** (OOM Killer) ngay khi bắt đầu xử lý ảnh, dù khởi động thành công.

#### Cách cấp đủ RAM cho Docker (Windows — WSL2 backend)

**Bước 1** — Tạo hoặc sửa file `C:\Users\<tên_user>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=4
swap=8GB
```

**Bước 2** — Áp dụng cấu hình:

```powershell
wsl --shutdown
# Sau đó mở lại Docker Desktop, chờ khởi động xong
```

**Bước 3** — Xác nhận RAM đã được tăng:

```powershell
docker info --format "{{.MemTotal}}"
# Kết quả mong đợi: ≥ 12548165632 (~12 GB)
```

---

#### Kiểm tra & Logs

```bash
docker compose ps
docker compose logs -f api
```

API sẵn sàng tại:
- Health check: `http://localhost:7861/`
- OCR endpoint: `POST http://localhost:7861/extract`
- API docs: `http://localhost:7861/docs`

#### Tắt hệ thống

```bash
docker compose down
```

---

### Phương án B — Conda (local development)

**Yêu cầu:**
- Python 3.10+
- Conda (Miniconda / Anaconda)
- GPU NVIDIA CUDA 12.x _(khuyến nghị — CPU cũng chạy được nhưng chậm)_

#### Bước 1 — Tạo môi trường

```powershell
conda create -n extract-pdf python=3.10 -y
conda activate extract-pdf
```

#### Bước 2 — Cài PyTorch (CUDA 12.1)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Bước 3 — Cài thư viện và tải model

**Windows:**
```powershell
.\setup_env.bat
```

**Linux/macOS:**
```bash
bash setup_env.sh
```

Script sẽ tự động:
- Cài đặt các thư viện Python từ `requirements.txt`
- Tải model weights (~2 GB) từ GitHub Releases nếu chưa có

Hoặc cài thủ công:
```bash
pip install -r requirements.txt
```

#### Kiểm tra

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Sử dụng

### 1. REST API Server (FastAPI)

```bash
conda activate extract-pdf
cd LightOnOCR-2-1B
python api.py
```

Server chạy tại `http://localhost:7861`

**Endpoints:**

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/` | Health check + thông tin device |
| `POST` | `/extract` | OCR file ảnh hoặc PDF |
| `POST` | `/download` | Tải file output (JSON/Excel) |

**Ví dụ gọi `/extract`:**

```bash
curl -X POST http://localhost:7861/extract \
  -F "file=@image.jpg" \
  -F "page_num=1" \
  -F "max_tokens=4096"
```

**Response:**

```json
{
  "status": "OK | source=image.jpg | tables=1 | text_lines=5 | kv_pairs=3",
  "rendered_text": "...",
  "raw_text": "...",
  "data": { "tables": [...], "text_lines": [...], "kv_pairs": {...} },
  "json_path": "/tmp/.../image.json",
  "excel_path": "/tmp/.../image.xlsx",
  "file_info": {
    "source_name": "image.jpg",
    "page_info": "image.jpg",
    "total_pages": 1,
    "actual_page": 1,
    "is_pdf": false
  }
}
```

#### Kết nối từ extract-pdf

Trong file `extract-pdf/ui-config.json`, cấu hình profile:

```json
{
  "id": "lightonocr-2-1b",
  "agent": "lightonocr",
  "base_url": "http://127.0.0.1:7861/extract"
}
```

Hoặc trong `extract-pdf/.env`:

```env
LOCAL_HTTP_BASE_URL=http://lightonocr:7861/extract  # Docker network
# LOCAL_HTTP_BASE_URL=http://127.0.0.1:7861/extract  # Local
```

---

### 2. Gradio Web UI

```bash
conda activate extract-pdf
cd LightOnOCR-2-1B
python demo.py
# → http://localhost:7860
```

---

### 3. CLI Batch

```bash
conda activate extract-pdf
cd LightOnOCR-2-1B

# Một ảnh
python run.py --input ..\datasets\Trang000001.jpg

# Một PDF (tất cả trang)
python run.py --input ..\datasets\document.pdf

# Thư mục
python run.py --input ..\datasets\ --output-dir outputs --output-name batch_01

# Thư mục đệ quy
python run.py --input ..\datasets\ --recursive --output-name all_results

# Tùy chỉnh token limit
python run.py --input ..\datasets\ --max-tokens 8192
```

**Output:** `outputs/<output-name>.json` và `outputs/<output-name>.xlsx`

**Tham số CLI:**

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--input` | _(bắt buộc)_ | File ảnh, PDF, hoặc thư mục |
| `--output-dir` | `outputs` | Thư mục lưu kết quả |
| `--output-name` | `result` | Tên file output (không có extension) |
| `--max-tokens` | `8192` | Giới hạn token mỗi trang |
| `--recursive` | False | Quét thư mục đệ quy |
| `--blank-threshold` | `0.99` | Ngưỡng phát hiện trang trắng |
| `--no-skip-blank` | False | Không bỏ qua trang trắng |

---

### 4. Test export (không cần OCR lại)

```bash
python test_export.py ..\datasets\Trang000001.json
python test_export.py ..\datasets\Trang000001.json outputs\custom_output.xlsx
```

---

## Cấu trúc dữ liệu output

### JSON

```json
[
  {
    "filename": "Trang000001.jpg",
    "ocr_text": "<văn bản OCR thô>",
    "tables": [
      {
        "headers": ["SĐT", "HỌ TÊN", "NGÀY SINH"],
        "rows": [
          { "SĐT": "1", "HỌ TÊN": "Nguyễn Văn A", "NGÀY SINH": "01/01/1990" }
        ]
      }
    ],
    "text_lines": ["DANH SÁCH CẤP BẰNG TỐT NGHIỆP"],
    "kv_pairs": { "Số SVTN": "51 SV" },
    "table_count": 1
  }
]
```

### Excel (`.xlsx`)

Mỗi cấu trúc bảng duy nhất → 1 sheet riêng:

| Kiểu dòng | Màu nền | Nội dung |
|---|---|---|
| **Header** | Xanh đậm (`#1F4E79`) | Tên các cột |
| **Text metadata** | Xanh nhạt (`#EBF3FB`, in nghiêng) | Dòng text ngoài bảng |
| **Dữ liệu bảng** | Trắng | Giá trị từng ô |

Ảnh không parse được bảng → sheet `OCR_Raw`.

---

## Kiến trúc pipeline

```
File (ảnh / PDF)
    │
    ▼  pipeline.lightonocr_common.load_uploaded_document()
    │   └── pdf_renderer.render_pdf_page()   — nếu là PDF
    │
    ▼  pipeline.ocr_engine.extract_text()
    │   ├── _prepare_inputs()    — tokenize + move to device/dtype
    │   └── model.generate()     — LightOnOCR inference
    │
    ▼  pipeline.table_parser.extract_structured_data()
    │   ├── _parse_html_tables()
    │   ├── _parse_markdown_tables()
    │   ├── _extract_text_lines()
    │   └── _extract_kv_pairs()
    │
    ├──▶ pipeline.exporter.save_json()      → .json
    └──▶ pipeline.exporter.json_to_excel()  → .xlsx
```

---

## Lưu ý

- **Model chạy hoàn toàn offline** sau khi tải weights về.
- **CPU mode:** Model chiếm ~4.5 GB RAM (float32). Trên Docker, đảm bảo cấp đủ RAM (khuyến nghị ≥ 6 GB).
- **GPU mode:** Model chiếm ~2.3 GB VRAM (bfloat16). Nhanh hơn CPU ~5–10 lần.
- PDF cần `pypdfium2`: `pip install pypdfium2`
- Temperature mặc định `0.2` trong API/demo, `0.0` (greedy) trong CLI batch.
