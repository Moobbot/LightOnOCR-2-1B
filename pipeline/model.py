"""
pipeline/model.py

Singleton loader cho LightOnOCR-2-1B model.
Model files được đọc từ thư mục cha của package này, hoặc từ MODEL_PATH env.

Environment variables:
  MODEL_PATH         : Đường dẫn tới model weights (mặc định: thư mục cha của package)
  LIGHTONOCR_DEVICE  : cpu | gpu | cuda | auto (mặc định: auto)
  LIGHTONOCR_DTYPE   : float32 | bfloat16 | float16 | auto (mặc định: auto)
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

logger = logging.getLogger("lightonocr.model")


# ---------------------------------------------------------------------------
# .env loader (chỉ dùng khi chạy trực tiếp ngoài Docker)
# ---------------------------------------------------------------------------

def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE từ .env nếu file tồn tại (không ghi đè biến đã có)."""
    if not env_path.is_file():
        return
    with env_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # Strip inline comments (chỉ khi value không được bao bởi quotes)
            if "#" in value and not (value.startswith(('"', "'"))):
                value = value.split("#", 1)[0].rstrip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# Ưu tiên .env khi chạy local (không ghi đè env đã được Docker inject)
_load_env_file(Path(__file__).resolve().parents[1] / ".env")


# ---------------------------------------------------------------------------
# Cấu hình device / dtype từ env
# ---------------------------------------------------------------------------

LOCAL_MODEL_PATH: str = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent.parent.resolve()),
)

_device_env = os.environ.get("LIGHTONOCR_DEVICE", "auto").strip().lower()
if _device_env == "cpu":
    DEVICE = "cpu"
elif _device_env in ("gpu", "cuda"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if _device_env != "cpu" and DEVICE == "cpu":
        logger.warning(
            "LIGHTONOCR_DEVICE=%s nhưng CUDA không có sẵn — fallback sang CPU.",
            _device_env,
        )
else:  # auto
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dtype: trên GPU dùng bfloat16 để tiết kiệm VRAM; CPU dùng float32 vì
# bfloat16/float16 trên CPU thường chậm hơn và không tiết kiệm RAM thực.
_dtype_env = os.environ.get("LIGHTONOCR_DTYPE", "auto").strip().lower()
if _dtype_env == "float32":
    DTYPE = torch.float32
elif _dtype_env in ("bfloat16", "bf16"):
    DTYPE = torch.bfloat16
elif _dtype_env in ("float16", "fp16"):
    DTYPE = torch.float16
else:  # auto
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Attention: sdpa hiệu quả hơn trên GPU; eager an toàn nhất trên CPU
ATTN_IMPLEMENTATION = "sdpa" if DEVICE == "cuda" else "eager"

# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_model: LightOnOcrForConditionalGeneration | None = None
_processor: LightOnOcrProcessor | None = None


def get_model() -> Tuple[LightOnOcrForConditionalGeneration, LightOnOcrProcessor]:
    """Load và cache model + processor. Trả về (model, processor).

    An toàn khi gọi nhiều lần — chỉ load 1 lần duy nhất.
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    # Giải phóng bộ nhớ trước khi load
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Loading model | path=%s | device=%s | dtype=%s | attn=%s",
        LOCAL_MODEL_PATH,
        DEVICE.upper(),
        DTYPE,
        ATTN_IMPLEMENTATION,
    )

    start = time.perf_counter()

    try:
        _model = (
            LightOnOcrForConditionalGeneration.from_pretrained(
                LOCAL_MODEL_PATH,
                attn_implementation=ATTN_IMPLEMENTATION,
                torch_dtype=DTYPE,
                trust_remote_code=True,
                local_files_only=True,
                ignore_mismatched_sizes=True,
            )
            .to(DEVICE)
            .eval()
        )
    except Exception:
        logger.exception("Không thể load model từ '%s'.", LOCAL_MODEL_PATH)
        raise

    try:
        _processor = LightOnOcrProcessor.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    except Exception:
        logger.exception("Không thể load processor từ '%s'.", LOCAL_MODEL_PATH)
        _model = None  # reset để tránh trạng thái nửa vời
        raise

    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2fs", elapsed)
    return _model, _processor
