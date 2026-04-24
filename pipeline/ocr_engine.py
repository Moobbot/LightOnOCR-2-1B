"""
pipeline/ocr_engine.py

OCR inference using LightOnOCR model.
- extract_text: run inference on a PIL Image
- is_blank_page: detect near-white/blank images to skip
- clean_output_text: strip chat template artifacts from output
"""

import torch
from PIL import Image

from .model import DEVICE, DTYPE


def is_blank_page(image: Image.Image, threshold: float = 0.99) -> bool:
    """
    Return True if the image is mostly blank (white).
    Uses pixel histogram to detect white ratio above `threshold`.
    """
    gray = image.convert("L")
    min_val, max_val = gray.getextrema()

    # Pure solid color shortcut
    if min_val == max_val:
        return max_val > 250  # pure white or near-white

    histogram = gray.histogram()
    white_pixels = sum(histogram[250:])  # pixels with brightness >= 250
    total_pixels = gray.width * gray.height
    return (white_pixels / total_pixels) > threshold


def clean_output_text(text: str) -> str:
    """
    Remove chat template role markers from model output.
    The model sometimes echoes 'user' / 'assistant' at the start.
    """
    markers_to_remove = {"system", "user", "assistant"}
    lines = text.split("\n")
    cleaned_lines = [
        line for line in lines if line.strip().lower() not in markers_to_remove
    ]
    cleaned = "\n".join(cleaned_lines).strip()

    # If 'assistant' marker appears, take only what follows it
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()

    return cleaned


def extract_text(
    model,
    processor,
    image: Image.Image,
    max_tokens: int = 8192,
) -> str:
    """
    Run OCR inference on a single PIL Image.
    Returns cleaned markdown text.
    """
    chat = [
        {
            "role": "user",
            "content": [{"type": "image", "url": image}],
        }
    ]

    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move tensors to correct device + dtype
    inputs = {
        k: (
            v.to(device=DEVICE, dtype=DTYPE)
            if isinstance(v, torch.Tensor)
            and v.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else v.to(DEVICE)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,   # deterministic
            top_p=0.9,
            use_cache=True,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    return clean_output_text(output_text)
