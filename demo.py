#!/usr/bin/env python3
"""
demo.py — LightOnOCR-2-1B Gradio Demo (local, simple)

Chạy:
    conda activate extract-pdf
    python demo.py
Mở trình duyệt tại http://localhost:7860
"""

import sys, json, os, tempfile, time, warnings
warnings.filterwarnings("ignore")

import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, ".")
from pipeline.model import get_model, DEVICE, ATTN_IMPLEMENTATION, LOCAL_MODEL_PATH, DTYPE
from pipeline.ocr_engine import clean_output_text, is_blank_page
from pipeline.table_parser import extract_structured_data
from pipeline.exporter import save_json, json_to_excel

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False

_model = None
_processor = None

def _ensure_model():
    global _model, _processor
    if _model is None:
        _model, _processor = get_model()
    return _model, _processor


# ── PDF ───────────────────────────────────────────────────────────────────────

def render_pdf_page(pdf_path, page_num=1, max_res=1540, scale=2.77):
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    idx = min(max(int(page_num) - 1, 0), total - 1)
    page = pdf[idx]
    w, h = page.get_size()
    rf = min(1.0, max_res / (w * scale), max_res / (h * scale))
    img = page.render(scale=scale * rf, rev_byteorder=True).to_pil()
    pdf.close()
    return img, total, idx + 1


def update_file_preview(file_input):
    if file_input is None:
        return gr.update(maximum=20, value=1, visible=False), None
    path = file_input if isinstance(file_input, str) else file_input.name
    if path.lower().endswith(".pdf") and HAS_PDFIUM:
        try:
            pdf = pdfium.PdfDocument(path)
            n = len(pdf)
            preview = pdf[0].render(scale=2).to_pil()
            pdf.close()
            return gr.update(maximum=n, value=1, visible=True), preview
        except Exception:
            return gr.update(maximum=20, value=1, visible=True), None
    else:
        try:
            return gr.update(maximum=1, value=1, visible=False), Image.open(path)
        except Exception:
            return gr.update(maximum=1, value=1, visible=False), None


# ── Core OCR (không streaming) ────────────────────────────────────────────────

def run_ocr(file_input, page_slider, temperature, max_tokens):
    """
    Chạy OCR đầy đủ, trả về:
      (status, rendered_md, raw_text, json_str, json_file, excel_file)
    """
    if file_input is None:
        return "⚠️ Vui lòng upload ảnh hoặc PDF.", "", "", "{}", None, None

    path = file_input if isinstance(file_input, str) else file_input.name

    # Load ảnh / PDF
    if path.lower().endswith(".pdf"):
        if not HAS_PDFIUM:
            return "❌ pypdfium2 chưa cài. Chạy: pip install pypdfium2", "", "", "{}", None, None
        try:
            image, total_pages, actual_page = render_pdf_page(path, int(page_slider))
            page_info = f"Trang {actual_page}/{total_pages}"
        except Exception as e:
            return f"❌ Lỗi đọc PDF: {e}", "", "", "{}", None, None
    else:
        try:
            image = Image.open(path).convert("RGB")
            page_info = os.path.basename(path)
        except Exception as e:
            return f"❌ Lỗi mở ảnh: {e}", "", "", "{}", None, None

    if is_blank_page(image):
        return "⚠️ Ảnh trắng/rỗng — bỏ qua.", "", "", "{}", None, None

    # Load model
    try:
        model, processor = _ensure_model()
    except Exception as e:
        return f"❌ Lỗi load model: {e}", "", "", "{}", None, None

    # Tokenize
    chat = [{"role": "user", "content": [{"type": "image", "url": image}]}]
    inputs = processor.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {
        k: (v.to(device=DEVICE, dtype=DTYPE)
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    do_sample = float(temperature) > 0
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature) if do_sample else 1.0,
        use_cache=True,
        do_sample=do_sample,
    )

    # Generate
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    full_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    full_text = clean_output_text(full_text)
    elapsed = time.time() - t0

    # Parse
    fname = os.path.basename(path)
    structured = extract_structured_data(fname, full_text)
    n_tables = structured["table_count"]
    n_texts  = len(structured.get("text_lines", []))
    n_kv     = len(structured.get("kv_pairs", {}))

    json_preview = {
        "tables":     structured["tables"],
        "text_lines": structured["text_lines"],
        "kv_pairs":   structured["kv_pairs"],
    }
    json_str = json.dumps(json_preview, ensure_ascii=False, indent=2)

    # Export files
    tmp = tempfile.mkdtemp()
    json_path  = os.path.join(tmp, f"{os.path.splitext(fname)[0]}.json")
    excel_path = os.path.join(tmp, f"{os.path.splitext(fname)[0]}.xlsx")
    save_json([structured], json_path)
    json_to_excel([structured], excel_path)

    status = f"✅ {page_info} | {elapsed:.1f}s | bảng={n_tables} | text={n_texts} | kv={n_kv}"
    return status, full_text, full_text, json_str, json_path, excel_path


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="LightOnOCR-2-1B",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(f"""
# 🔍 LightOnOCR-2-1B — Demo local
**Device:** `{DEVICE.upper()}` &nbsp;|&nbsp; **Attn:** `{ATTN_IMPLEMENTATION}`
""")

    with gr.Row():
        # ── Cột trái ─────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📂 Upload ảnh hoặc PDF",
                file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"],
                type="filepath",
            )
            preview_img = gr.Image(
                label="🖼 Preview", type="pil", height=320, interactive=False,
            )
            page_slider = gr.Slider(
                minimum=1, maximum=20, value=1, step=1,
                label="📄 Trang PDF", visible=False,
            )
            temperature = gr.Slider(
                0.0, 1.0, value=0.2, step=0.05,
                label="Temperature (0 = deterministic)",
            )
            max_tokens = gr.Slider(
                256, 16384, value=4096, step=256, label="Max tokens",
            )
            with gr.Row():
                run_btn   = gr.Button("🚀 Chạy OCR", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Xóa", size="lg")

        # ── Cột phải ──────────────────────────────────────────────────────────
        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="ℹ️ Trạng thái", value="", interactive=False, lines=1,
            )
            output_md = gr.Markdown(
                value="*Kết quả OCR sẽ hiển thị ở đây...*",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$",  "right": "$",  "display": False},
                ],
                label="📄 Văn bản (Rendered)",
            )
            output_raw = gr.Textbox(
                label="📝 Raw Text",
                lines=10,
                max_lines=20,
                interactive=False,
            )
            json_out = gr.Code(
                label="🗂️ JSON (Bảng + KV)",
                language="json",
                lines=10,
            )

    # ── Nút tải xuống (cuối trang) ────────────────────────────────────────────
    gr.Markdown("---\n### 💾 Tải xuống kết quả")
    with gr.Row():
        download_json  = gr.File(label="📥 result.json")
        download_excel = gr.File(label="📥 result.xlsx")

    # ── Ảnh mẫu ───────────────────────────────────────────────────────────────
    example_files = []
    for search_dir in [r"..\datasets\exemples", r"..\datasets\test"]:
        if os.path.isdir(search_dir):
            for f in sorted(os.listdir(search_dir))[:6]:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    example_files.append([os.path.join(search_dir, f)])
            if example_files:
                break

    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=[file_input],
            label="📁 Ảnh mẫu",
        )

    # ── Events ────────────────────────────────────────────────────────────────

    file_input.change(
        fn=update_file_preview,
        inputs=[file_input],
        outputs=[page_slider, preview_img],
    )

    run_btn.click(
        fn=run_ocr,
        inputs=[file_input, page_slider, temperature, max_tokens],
        outputs=[status_box, output_md, output_raw, json_out, download_json, download_excel],
    )

    clear_btn.click(
        fn=lambda: (None, None, gr.update(visible=False), "", "*Kết quả OCR sẽ hiển thị ở đây...*", "", "{}", None, None),
        outputs=[file_input, preview_img, page_slider, status_box, output_md, output_raw, json_out, download_json, download_excel],
    )


if __name__ == "__main__":
    print(f"\n{'=' * 50}")
    print(f"  LightOnOCR-2-1B — Demo")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  URL    : http://localhost:7860")
    print(f"{'=' * 50}\n")

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        inbrowser=True,
        ssr_mode=False,
    )
