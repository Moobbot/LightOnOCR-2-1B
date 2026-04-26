#!/usr/bin/env python3
"""LightOnOCR-2-1B minimal Gradio demo.

Flow: upload image/PDF -> preview -> click Extract -> show OCR text.
"""

from __future__ import annotations

import os
import socket
import time
import warnings

import gradio as gr

from pipeline.lightonocr_common import load_uploaded_document, process_uploaded_document
from pipeline.model import DEVICE

warnings.filterwarnings("ignore")


def update_preview(file_input):
    if file_input is None:
        return gr.update(value=1, maximum=1, visible=False), None, ""

    try:
        loaded = load_uploaded_document(file_input, page_num=1)
        return (
            gr.update(
                value=loaded.actual_page,
                maximum=loaded.total_pages,
                visible=loaded.is_pdf,
            ),
            loaded.image,
            loaded.page_info,
        )
    except Exception as exc:
        return (
            gr.update(value=1, maximum=1, visible=False),
            None,
            f"❌ Preview failed: {exc}",
        )


def run_extract(file_input, page_num):
    if file_input is None:
        return "⚠️ Vui lòng upload ảnh hoặc PDF.", "", None, None

    started = time.time()
    try:
        _loaded, bundle = process_uploaded_document(
            file_input=file_input,
            page_num=int(page_num or 1),
            prompt="Extract all text and tables from this image.",
            temperature=0.2,
            max_tokens=4096,
        )
        elapsed = time.time() - started
        status = f"{bundle.status} | {elapsed:.1f}s"
        return (
            status,
            bundle.rendered_text or bundle.raw_text,
            bundle.json_path,
            bundle.excel_path,
        )
    except Exception as exc:
        return f"❌ OCR failed: {exc}", "", None, None


def _find_free_port(start_port: int = 7860, end_port: int = 7870) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


with gr.Blocks(title="LightOnOCR-2-1B") as demo:
    gr.Markdown(
        f"""
# LightOnOCR-2-1B
Upload ảnh/PDF rồi bấm Extract.

**Device:** {DEVICE.upper()}
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload ảnh hoặc PDF",
                file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"],
                type="filepath",
            )
            page_num = gr.Slider(
                1, 200, value=1, step=1, label="PDF page", visible=False
            )
            preview = gr.Image(
                label="Preview", type="pil", interactive=False, height=320
            )
            status = gr.Textbox(label="Status", interactive=False, lines=1)
            extract_btn = gr.Button("Extract", variant="primary")

        with gr.Column(scale=2):
            result = gr.Markdown(
                label="Markdown Preview",
                value="*Kết quả Markdown sẽ hiển thị ở đây...*",
            )
            json_file = gr.File(label="JSON")
            excel_file = gr.File(label="Excel")

    file_input.change(
        fn=update_preview,
        inputs=[file_input],
        outputs=[page_num, preview, status],
    )

    extract_btn.click(
        fn=run_extract,
        inputs=[file_input, page_num],
        outputs=[status, result, json_file, excel_file],
    )


if __name__ == "__main__":
    host = os.environ.get("GRADIO_HOST", "localhost")
    port = int(os.environ.get("GRADIO_PORT") or _find_free_port())

    print(f"\n{'=' * 50}")
    print("  LightOnOCR-2-1B — Demo")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  URL    : http://{host}:{port}")
    print(f"{'=' * 50}\n")

    demo.launch(
        server_name=host, server_port=port, share=False, inbrowser=host != "0.0.0.0"
    )
