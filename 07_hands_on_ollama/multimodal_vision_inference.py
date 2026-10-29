#!/usr/bin/env python3
"""
Multimodal Vision Inference with Ollama.
Demonstrates how Vision-Language Models (VLMs) process image patches into tokens,
measure vision prefill latency, and generate responses from visual inputs.
"""

import argparse
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib


def generate_test_image_png(width: int = 128, height: int = 128, color=(0, 180, 255)) -> bytes:
    """Generates an in-memory test PNG (pure Python, zero dependencies)."""
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # Filter type none
        for x in range(width):
            # Draw a solid cyan rectangle with a darker border
            if x < 4 or x >= width - 4 or y < 4 or y >= height - 4:
                raw_data.extend((0, 80, 150))
            else:
                raw_data.extend(color)
    compressed = zlib.compress(bytes(raw_data))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    png.extend(struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc))
    # IDAT
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    png.extend(struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc))
    # IEND
    iend_crc = zlib.crc32(b"IEND")
    png.extend(struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc))
    return bytes(png)


def run_multimodal_inference(
    model: str = "bonsai-2:latest",
    image_path: str = None,
    prompt: str = "What colors and shapes are present in this image?",
):
    print("=" * 82)
    print(f"  MULTIMODAL VISION INFERENCE BENCHMARK ({model})")
    print("=" * 82)

    if image_path:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        print(f"Loaded user image from: {image_path} ({len(img_bytes):,} bytes)")
    else:
        print("Synthesizing in-memory cyan test image (128x128 PNG)...")
        img_bytes = generate_test_image_png(128, 128, (0, 180, 255))

    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "options": {
            "num_predict": 250,
            "temperature": 0.2,
        }
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print("Sending prompt + image tokens to Ollama vision projector...")
    start_t = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start_t

    prefill_tokens = res.get("prompt_eval_count", 0)
    prefill_ms = res.get("prompt_eval_duration", 1) / 1e6
    decode_tokens = res.get("eval_count", 0)
    decode_ms = res.get("eval_duration", 1) / 1e6
    decode_speed = (decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0

    print("\n--- INFERENCE TELEMETRY ---")
    print(f"Total Prefill Tokens (Text + Image Patches): {prefill_tokens} tokens")
    print(f"Vision Projector + Prefill Latency:           {prefill_ms:.2f} ms")
    print(f"Decode Generated Tokens:                     {decode_tokens} tokens")
    print(f"Decode Throughput:                           {decode_speed:.2f} tokens/sec")
    print(f"Total Turnaround Time:                       {elapsed:.2f} s")
    print("-" * 82)

    thinking = res.get("thinking", "").strip()
    if thinking:
        print(f"\nInternal Visual Reasoning:\n{thinking[:300]}...\n")

    print(f"Final Model Response:\n{res.get('response', '').strip()}")
    print("=" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Vision Inference")
    parser.add_argument("--model", type=str, default="bonsai-2:latest", help="VLM Model name")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (optional)")
    parser.add_argument("--prompt", type=str, default="What colors and shapes are present in this image?", help="Prompt")
    args = parser.parse_args()

    run_multimodal_inference(model=args.model, image_path=args.image, prompt=args.prompt)
