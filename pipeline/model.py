"""
pipeline/model.py

Singleton loader for the local LightOnOCR-2-1B model.
Model files are expected in the parent directory of this package
(i.e., d:/Work/Clients/A_Giap/extract-pdf/LightOnOCR-2-1B/).
"""

import gc
import os
import time
import torch
from pathlib import Path
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

# Local model path — có thể override bằng env var MODEL_PATH (dùng trong Docker)
LOCAL_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent.parent.resolve()),
)

# Device & dtype auto-detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
ATTN_IMPLEMENTATION = "sdpa" if DEVICE == "cuda" else "eager"

# Singleton cache
_model = None
_processor = None


def get_model():
    """Load and cache the model + processor. Returns (model, processor)."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    # Free memory before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Model] Loading from: {LOCAL_MODEL_PATH}")
    print(f"[Model] Device={DEVICE.upper()}, dtype={DTYPE}, attn={ATTN_IMPLEMENTATION}")

    start = time.time()

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

    _processor = LightOnOcrProcessor.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    print(f"[Model] Loaded in {time.time() - start:.2f}s")
    return _model, _processor
