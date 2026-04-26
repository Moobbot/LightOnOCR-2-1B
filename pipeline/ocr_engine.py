"""
pipeline/ocr_engine.py

OCR inference engine cho LightOnOCR-2-1B.

INPUT  : model, processor (từ pipeline.model.get_model)
         image: PIL.Image
         tham số sinh text: max_tokens, temperature, do_sample
OUTPUT : str — văn bản OCR đã làm sạch

Hàm hỗ trợ:
  - is_blank_page   : phát hiện trang trắng
  - clean_output_text: loại bỏ template markers
"""

from __future__ import annotations

import torch
from PIL import Image

from .model import DEVICE, DTYPE


# =============================================================================
# LOGIC — Tiền xử lý / hậu xử lý
# =============================================================================


def is_blank_page(image: Image.Image, threshold: float = 0.99) -> bool:
    """
    INPUT  : PIL Image, ngưỡng tỉ lệ pixel trắng
    OUTPUT : True nếu trang trắng/rỗng (cần bỏ qua)

    Logic  : tính tỉ lệ pixel sáng (≥ 250) trên toàn ảnh grayscale
    """
    gray = image.convert("L")
    min_val, max_val = gray.getextrema()

    # Màu đồng nhất (ảnh hoàn toàn trắng hoặc màu đặc)
    if min_val == max_val:
        return max_val > 250

    histogram = gray.histogram()
    white_pixels = sum(histogram[250:])
    total_pixels = gray.width * gray.height
    return (white_pixels / total_pixels) > threshold


def clean_output_text(text: str) -> str:
    """
    INPUT  : raw output từ model (có thể chứa role markers)
    OUTPUT : văn bản sạch

    Logic  : loại dòng chứa 'system'/'user'/'assistant';
             nếu có 'assistant', lấy phần sau nó
    """
    markers = {"system", "user", "assistant"}
    lines = text.split("\n")
    cleaned_lines = [l for l in lines if l.strip().lower() not in markers]
    cleaned = "\n".join(cleaned_lines).strip()

    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()

    return cleaned


def _prepare_inputs(processor, image: Image.Image, prompt: str) -> dict:
    """
    INPUT  : processor, PIL Image
    OUTPUT : dict of tensors đã chuyển về đúng device/dtype

    Logic  : Process image + text qua processor
             (LightOnOcrProcessor không có apply_chat_template)
    """
    if hasattr(processor, "apply_chat_template"):
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

    # Move to device + convert dtype
    return {
        k: (
            v.to(device=DEVICE, dtype=DTYPE)
            if isinstance(v, torch.Tensor)
            and v.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }


# =============================================================================
# OUTPUT — Chạy inference
# =============================================================================


def extract_text(
    model,
    processor,
    image: Image.Image,
    prompt: str = "Extract all text and tables from this image.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False,
) -> str:
    """
    INPUT  : model, processor (từ get_model())
             image       — PIL Image cần OCR
             max_tokens  — giới hạn token sinh ra
             temperature — độ ngẫu nhiên (0 = deterministic)
             top_p       — nucleus sampling (dùng khi do_sample=True, khuyến nghị 0.9)
             do_sample   — True để dùng sampling, False để greedy
    OUTPUT : str — văn bản OCR đã làm sạch

    Logic  : tokenize → generate → decode → clean
    """
    inputs = _prepare_inputs(processor, image, prompt)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            use_cache=True,
            do_sample=do_sample,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    return clean_output_text(raw_text)
