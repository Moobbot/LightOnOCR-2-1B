#!/usr/bin/env python3
import warnings

# Suppress FutureWarning from spaces library about torch.distributed.reduce_op
warnings.filterwarnings("ignore", category=FutureWarning, module="spaces")

import base64
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from io import BytesIO

import gradio as gr
import pypdfium2 as pdfium
import spaces
import torch
from openai import OpenAI
from PIL import Image
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    TextIteratorStreamer,
)

# vLLM endpoint configuration from environment variables
VLLM_ENDPOINT_OCR = os.environ.get("VLLM_ENDPOINT_OCR")
VLLM_ENDPOINT_BBOX = os.environ.get("VLLM_ENDPOINT_BBOX")

# Streaming configuration
STREAM_YIELD_INTERVAL = 0.5  # Yield every N seconds to reduce UI overhead

# Model Registry with all supported models
MODEL_REGISTRY = {
    "LightOnOCR-2-1B (Best OCR)": {
        "model_id": "lightonai/LightOnOCR-2-1B",
        "has_bbox": False,
        "description": "Best overall OCR performance",
        "vllm_endpoint": VLLM_ENDPOINT_OCR,
    },
    "LightOnOCR-2-1B-bbox (Best Bbox)": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox",
        "has_bbox": True,
        "description": "Best bounding box detection",
        "vllm_endpoint": VLLM_ENDPOINT_BBOX,
    },
    "LightOnOCR-2-1B-base": {
        "model_id": "lightonai/LightOnOCR-2-1B-base",
        "has_bbox": False,
        "description": "Base OCR model",
    },
    "LightOnOCR-2-1B-bbox-base": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-base",
        "has_bbox": True,
        "description": "Base bounding box model",
    },
    "LightOnOCR-2-1B-ocr-soup": {
        "model_id": "lightonai/LightOnOCR-2-1B-ocr-soup",
        "has_bbox": False,
        "description": "OCR soup variant",
    },
    "LightOnOCR-2-1B-bbox-soup": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-soup",
        "has_bbox": True,
        "description": "Bounding box soup variant",
    },
}

DEFAULT_MODEL = "LightOnOCR-2-1B (Best OCR)"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose best attention implementation based on device
if device == "cuda":
    attn_implementation = "sdpa"
    dtype = torch.bfloat16
    print("Using sdpa for GPU")
else:
    attn_implementation = "eager"  # Best for CPU
    dtype = torch.float32
    print("Using eager attention for CPU")


class ModelManager:
    """Manages model loading with LRU caching and GPU memory management."""

    def __init__(self, max_cached=2):
        self._cache = OrderedDict()  # {model_id: (model, processor)}
        self._max_cached = max_cached

    def get_model(self, model_name):
        """Get model and processor, loading if necessary."""
        config = MODEL_REGISTRY.get(model_name)
        if config is None:
            raise ValueError(f"Unknown model: {model_name}")

        model_id = config["model_id"]

        # Check cache
        if model_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(model_id)
            print(f"Using cached model: {model_name}")
            return self._cache[model_id]

        # Evict oldest if cache is full
        while len(self._cache) >= self._max_cached:
            evicted_id, (evicted_model, _) = self._cache.popitem(last=False)
            print(f"Evicting model from cache: {evicted_id}")
            del evicted_model
            if device == "cuda":
                torch.cuda.empty_cache()

        # Load new model
        print(f"Loading model: {model_name} ({model_id})...")
        model = (
            LightOnOcrForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )

        processor = LightOnOcrProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

        # Add to cache
        self._cache[model_id] = (model, processor)
        print(f"Model loaded successfully: {model_name}")

        return model, processor

    def get_model_info(self, model_name):
        """Get model info without loading."""
        return MODEL_REGISTRY.get(model_name)


# Initialize model manager
model_manager = ModelManager(max_cached=2)
print("Model manager initialized. Models will be loaded on first use.")


def render_pdf_page(page, max_resolution=1540, scale=2.77):
    """Render a PDF page to PIL Image."""
    width, height = page.get_size()
    pixel_width = width * scale
    pixel_height = height * scale
    resize_factor = min(1, max_resolution / pixel_width, max_resolution / pixel_height)
    target_scale = scale * resize_factor
    return page.render(scale=target_scale, rev_byteorder=True).to_pil()


def process_pdf(pdf_path, page_num=1):
    """Extract a specific page from PDF."""
    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)
    page_idx = min(max(int(page_num) - 1, 0), total_pages - 1)

    page = pdf[page_idx]
    img = render_pdf_page(page)

    pdf.close()
    return img, total_pages, page_idx + 1


def clean_output_text(text):
    """Remove chat template artifacts from output."""
    # Remove common chat template markers
    markers_to_remove = ["system", "user", "assistant"]

    # Split by lines and filter
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip lines that are just template markers
        if stripped.lower() not in markers_to_remove:
            cleaned_lines.append(line)

    # Join back and strip leading/trailing whitespace
    cleaned = "\n".join(cleaned_lines).strip()

    # Alternative approach: if there's an "assistant" marker, take everything after it
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()

    return cleaned


# Bbox parsing pattern: ![image](image_N.png)x1,y1,x2,y2 (no space between)
BBOX_PATTERN = r"!\[image\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)"


def parse_bbox_output(text):
    """Parse bbox output and return cleaned text with list of detections."""
    detections = []
    for match in re.finditer(BBOX_PATTERN, text):
        image_ref, x1, y1, x2, y2 = match.groups()
        detections.append(
            {"ref": image_ref, "coords": (int(x1), int(y1), int(x2), int(y2))}
        )
    # Clean text: remove coordinates, keep markdown image refs
    cleaned = re.sub(BBOX_PATTERN, r"![image](\1)", text)
    return cleaned, detections


def crop_from_bbox(source_image, bbox, padding=5):
    """Crop region from image based on normalized [0,1000] coords."""
    w, h = source_image.size
    x1, y1, x2, y2 = bbox["coords"]

    # Convert to pixel coordinates (coords are normalized to 0-1000)
    px1 = int(x1 * w / 1000)
    py1 = int(y1 * h / 1000)
    px2 = int(x2 * w / 1000)
    py2 = int(y2 * h / 1000)

    # Add padding, clamp to bounds
    px1, py1 = max(0, px1 - padding), max(0, py1 - padding)
    px2, py2 = min(w, px2 + padding), min(h, py2 + padding)

    return source_image.crop((px1, py1, px2, py2))


def image_to_data_uri(image):
    """Convert PIL image to base64 data URI for markdown embedding."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def extract_text_via_vllm(image, model_name, temperature=0.2, stream=False, max_tokens=2048):
    """Extract text from image using vLLM endpoint."""
    config = MODEL_REGISTRY.get(model_name)
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")

    endpoint = config.get("vllm_endpoint")
    if endpoint is None:
        raise ValueError(f"Model {model_name} does not have a vLLM endpoint")

    model_id = config["model_id"]

    # Convert image to base64 data URI
    if isinstance(image, Image.Image):
        image_uri = image_to_data_uri(image)
    else:
        # Assume it's already a data URI or URL
        image_uri = image

    # Create OpenAI client pointing to vLLM endpoint
    client = OpenAI(base_url=endpoint, api_key="not-needed")

    # Prepare the message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
            ],
        }
    ]

    if stream:
        # Streaming response
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.9,
            stream=True,
        )

        full_text = ""
        last_yield_time = time.time()
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                # Batch yields to reduce UI overhead
                if time.time() - last_yield_time > STREAM_YIELD_INTERVAL:
                    yield clean_output_text(full_text)
                    last_yield_time = time.time()
        # Final yield with cleaned text
        yield clean_output_text(full_text)
    else:
        # Non-streaming response
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.9,
            stream=False,
        )

        output_text = response.choices[0].message.content
        cleaned_text = clean_output_text(output_text)
        yield cleaned_text


def render_bbox_with_crops(raw_output, source_image):
    """Replace markdown image placeholders with actual cropped images."""
    cleaned, detections = parse_bbox_output(raw_output)

    for bbox in detections:
        try:
            cropped = crop_from_bbox(source_image, bbox)
            data_uri = image_to_data_uri(cropped)
            # Replace ![image](image_N.png) with ![Cropped](data:...)
            cleaned = cleaned.replace(
                f"![image]({bbox['ref']})", f"![Cropped region]({data_uri})"
            )
        except Exception as e:
            print(f"Error cropping bbox {bbox}: {e}")
            # Keep original reference if cropping fails
            continue

    return cleaned


@spaces.GPU
def extract_text_from_image(image, model_name, temperature=0.2, stream=False, max_tokens=2048):
    """Extract text from image using LightOnOCR model."""
    # Check if model has a vLLM endpoint configured
    config = MODEL_REGISTRY.get(model_name, {})
    if config.get("vllm_endpoint"):
        # Use vLLM endpoint instead of local model
        yield from extract_text_via_vllm(image, model_name, temperature, stream, max_tokens)
        return

    # Get model and processor from cache or load
    model, processor = model_manager.get_model(model_name)

    # Prepare the chat format
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
            ],
        }
    ]

    # Apply chat template and tokenize
    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to device AND convert to the correct dtype
    inputs = {
        k: v.to(device=device, dtype=dtype)
        if isinstance(v, torch.Tensor)
        and v.dtype in [torch.float32, torch.float16, torch.bfloat16]
        else v.to(device)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=0.9,
        top_k=0,
        use_cache=True,
        do_sample=temperature > 0,
    )

    if stream:
        # Setup streamer for streaming generation
        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer

        # Run generation in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield chunks as they arrive
        full_text = ""
        last_yield_time = time.time()
        for new_text in streamer:
            full_text += new_text
            # Batch yields to reduce UI overhead
            if time.time() - last_yield_time > STREAM_YIELD_INTERVAL:
                yield clean_output_text(full_text)
                last_yield_time = time.time()

        thread.join()
        # Final yield with cleaned text
        yield clean_output_text(full_text)
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        # Decode the output
        output_text = processor.decode(outputs[0], skip_special_tokens=True)

        # Clean the output
        cleaned_text = clean_output_text(output_text)

        yield cleaned_text


def process_input(file_input, model_name, temperature, page_num, enable_streaming, max_output_tokens):
    """Process uploaded file (image or PDF) and extract text with optional streaming."""
    if file_input is None:
        yield "Please upload an image or PDF first.", "", "", None, gr.update()
        return

    image_to_process = None
    page_info = ""

    file_path = file_input if isinstance(file_input, str) else file_input.name

    # Handle PDF files
    if file_path.lower().endswith(".pdf"):
        try:
            image_to_process, total_pages, actual_page = process_pdf(
                file_path, int(page_num)
            )
            page_info = f"Processing page {actual_page} of {total_pages}"
        except Exception as e:
            yield f"Error processing PDF: {str(e)}", "", "", None, gr.update()
            return
    # Handle image files
    else:
        try:
            image_to_process = Image.open(file_path)
            page_info = "Processing image"
        except Exception as e:
            yield f"Error opening image: {str(e)}", "", "", None, gr.update()
            return

    # Check if model has bbox capability
    model_info = MODEL_REGISTRY.get(model_name, {})
    has_bbox = model_info.get("has_bbox", False)

    try:
        # Extract text using LightOnOCR with optional streaming
        for extracted_text in extract_text_from_image(
            image_to_process, model_name, temperature, stream=enable_streaming, max_tokens=max_output_tokens
        ):
            # For bbox models, render cropped images inline
            if has_bbox:
                rendered_text = render_bbox_with_crops(extracted_text, image_to_process)
            else:
                rendered_text = extracted_text
            yield (
                rendered_text,
                extracted_text,
                page_info,
                image_to_process,
                gr.update(),
            )

    except Exception as e:
        error_msg = f"Error during text extraction: {str(e)}"
        yield error_msg, error_msg, page_info, image_to_process, gr.update()


def update_slider_and_preview(file_input):
    """Update page slider and preview image based on uploaded file."""
    if file_input is None:
        return gr.update(maximum=20, value=1), None

    file_path = file_input if isinstance(file_input, str) else file_input.name

    if file_path.lower().endswith(".pdf"):
        try:
            pdf = pdfium.PdfDocument(file_path)
            total_pages = len(pdf)
            # Render first page for preview
            page = pdf[0]
            preview_image = page.render(scale=2).to_pil()
            pdf.close()
            return gr.update(maximum=total_pages, value=1), preview_image
        except:
            return gr.update(maximum=20, value=1), None
    else:
        # It's an image file
        try:
            preview_image = Image.open(file_path)
            return gr.update(maximum=1, value=1), preview_image
        except:
            return gr.update(maximum=1, value=1), None


# Helper function to get model info text
def get_model_info_text(model_name):
    """Return formatted model info string."""
    info = MODEL_REGISTRY.get(model_name, {})
    has_bbox = (
        "Yes - will show cropped regions inline"
        if info.get("has_bbox", False)
        else "No"
    )
    return f"**Description:** {info.get('description', 'N/A')}\n**Bounding Box Detection:** {has_bbox}"


# Create Gradio interface
with gr.Blocks(title="LightOnOCR-2 Multi-Model OCR") as demo:
    gr.Markdown(f"""
# LightOnOCR-2 — Efficient 1B VLM for OCR

State-of-the-art OCR on OlmOCR-Bench, ~9× smaller and faster than competitors. Handles tables, forms, math, multi-column layouts.

⚡ **3.3× faster** than Chandra, **1.7× faster** than OlmOCR | 💸 **<$0.01/1k pages** | 🧠 End-to-end differentiable | 📍 Bbox variants for image detection

📄 [Paper](https://arxiv.org/pdf/2601.14251) | 📝 [Blog](https://huggingface.co/blog/lightonai/lightonocr-2) | 📊 [Dataset](https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126) | 📓 [Finetuning](https://colab.research.google.com/drive/1WjbsFJZ4vOAAlKtcCauFLn_evo5UBRNa?usp=sharing)

---

**How to use:** Select a model → Upload image/PDF → Click "Extract Text" | **Device:** {device.upper()} | **Attention:** {attn_implementation}
""")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                value=DEFAULT_MODEL,
                label="Model",
                info="Select OCR model variant",
            )
            model_info = gr.Markdown(
                value=get_model_info_text(DEFAULT_MODEL), label="Model Info"
            )
            file_input = gr.File(
                label="Upload Image or PDF",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath",
            )
            rendered_image = gr.Image(
                label="Preview", type="pil", height=400, interactive=False
            )
            num_pages = gr.Slider(
                minimum=1,
                maximum=20,
                value=1,
                step=1,
                label="PDF: Page Number",
                info="Select which page to extract",
            )
            page_info = gr.Textbox(label="Processing Info", value="", interactive=False)
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.05,
                label="Temperature",
                info="0.0 = deterministic, Higher = more varied",
            )
            enable_streaming = gr.Checkbox(
                label="Enable Streaming",
                value=True,
                info="Show text progressively as it's generated",
            )
            max_output_tokens = gr.Slider(
                minimum=256,
                maximum=8192,
                value=2048,
                step=256,
                label="Max Output Tokens",
                info="Maximum number of tokens to generate",
            )
            submit_btn = gr.Button("Extract Text", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):
            output_text = gr.Markdown(
                label="📄 Extracted Text (Rendered)",
                value="*Extracted text will appear here...*",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )

    # Example inputs with image previews
    EXAMPLE_IMAGES = [
        "examples/example_1.png",
        "examples/example_2.png",
        "examples/example_3.png",
        "examples/example_4.png",
        "examples/example_5.png",
        "examples/example_6.png",
        "examples/example_7.png",
        "examples/example_8.png",
        "examples/example_9.png",
    ]

    with gr.Accordion("📁 Example Documents (click an image to load)", open=True):
        example_gallery = gr.Gallery(
            value=EXAMPLE_IMAGES,
            columns=5,
            rows=2,
            height="auto",
            object_fit="contain",
            show_label=False,
            allow_preview=False,
        )

    def load_example_image(evt: gr.SelectData):
        """Load selected example image into file input."""
        return EXAMPLE_IMAGES[evt.index]

    example_gallery.select(
        fn=load_example_image,
        outputs=[file_input],
    )

    with gr.Row():
        with gr.Column():
            raw_output = gr.Textbox(
                label="Raw Markdown Output",
                placeholder="Raw text will appear here...",
                lines=20,
                max_lines=30,
            )

    # Event handlers
    submit_btn.click(
        fn=process_input,
        inputs=[file_input, model_selector, temperature, num_pages, enable_streaming, max_output_tokens],
        outputs=[output_text, raw_output, page_info, rendered_image, num_pages],
    )

    file_input.change(
        fn=update_slider_and_preview,
        inputs=[file_input],
        outputs=[num_pages, rendered_image],
    )

    model_selector.change(
        fn=get_model_info_text, inputs=[model_selector], outputs=[model_info]
    )

    clear_btn.click(
        fn=lambda: (
            None,
            DEFAULT_MODEL,
            get_model_info_text(DEFAULT_MODEL),
            "*Extracted text will appear here...*",
            "",
            "",
            None,
            1,
            2048,
        ),
        outputs=[
            file_input,
            model_selector,
            model_info,
            output_text,
            raw_output,
            page_info,
            rendered_image,
            num_pages,
            max_output_tokens,
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)
