"""
pipeline/ocr_engine.py

OCR inference engine cho LightOnOCR-2-1B.

INPUT  : model, processor (từ pipeline.model.get_model)
         image: PIL.Image
         tham số sinh text: max_tokens, temperature, do_sample
OUTPUT : str — văn bản OCR đã làm sạch

Hàm hỗ trợ:
  is_blank_page    : phát hiện trang trắng
  clean_output_text: loại bỏ template markers từ output của model
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from .model import DEVICE, DTYPE

logger = logging.getLogger("lightonocr.ocr_engine")


# ---------------------------------------------------------------------------
# Tiền / hậu xử lý
# ---------------------------------------------------------------------------


def is_blank_page(image: Image.Image, threshold: float = 0.99) -> bool:
    """Kiểm tra trang trắng/rỗng.

    Args:
        image: Ảnh PIL cần kiểm tra.
        threshold: Tỉ lệ pixel sáng (≥ 250) để coi là trang trắng (mặc định 0.99).

    Returns:
        True nếu trang được coi là trắng/rỗng.
    """
    gray = image.convert("L")
    min_val, max_val = gray.getextrema()

    # Ảnh đồng màu hoàn toàn
    if min_val == max_val:
        return max_val > 250

    histogram = gray.histogram()
    white_pixels = sum(histogram[250:])
    total_pixels = gray.width * gray.height
    return (white_pixels / total_pixels) > threshold


def clean_output_text(text: str) -> str:
    """Làm sạch output của model, loại bỏ role markers.

    Args:
        text: Văn bản thô từ model (có thể chứa 'system'/'user'/'assistant').

    Returns:
        Văn bản đã làm sạch.
    """
    markers = {"system", "user", "assistant"}
    lines = text.split("\n")
    cleaned_lines = [ln for ln in lines if ln.strip().lower() not in markers]
    cleaned = "\n".join(cleaned_lines).strip()

    # Nếu có 'assistant', lấy phần nội dung sau nó
    lower_text = text.lower()
    if "assistant" in lower_text:
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()

    return cleaned


# ---------------------------------------------------------------------------
# Chuẩn bị đầu vào
# ---------------------------------------------------------------------------


def _prepare_inputs(processor, image: Image.Image, prompt: str) -> dict:
    """Chuẩn bị tensor đầu vào cho model.

    Args:
        processor: LightOnOcrProcessor.
        image: Ảnh PIL.
        prompt: Câu lệnh OCR.

    Returns:
        dict các tensors đã chuyển sang đúng device / dtype.
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

    _float_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    return {
        k: (
            v.to(device=DEVICE, dtype=DTYPE)
            if isinstance(v, torch.Tensor) and v.dtype in _float_dtypes
            else v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


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
    """Chạy OCR inference trên một ảnh.

    Args:
        model: LightOnOcrForConditionalGeneration đã load.
        processor: LightOnOcrProcessor đã load.
        image: Ảnh PIL cần OCR.
        prompt: Câu lệnh hướng dẫn model.
        max_tokens: Giới hạn số token sinh ra.
        temperature: Độ ngẫu nhiên (0 = greedy/deterministic).
        top_p: Ngưỡng nucleus sampling (chỉ dùng khi do_sample=True).
        do_sample: True để dùng sampling, False để greedy decoding.

    Returns:
        Văn bản OCR đã làm sạch.
    """
    logger.debug(
        "extract_text | prompt_len=%d | max_tokens=%d | temperature=%.2f | do_sample=%s",
        len(prompt),
        max_tokens,
        temperature,
        do_sample,
    )

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

    logger.debug("extract_text | output_tokens=%d", len(generated_ids))
    return clean_output_text(raw_text)
