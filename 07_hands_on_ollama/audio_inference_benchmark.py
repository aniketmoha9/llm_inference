#!/usr/bin/env python3
"""
Multimodal Audio Inference Benchmark with nemotron-3.5-bolt:e4b-mxfp8.
Demonstrates how the native sound_encoder in Gemma 4 / Nemotron processes
audio waveforms, computes audio prefill tokens, and performs speech reasoning.
"""

import argparse
import base64
import json
import math
import os
import shutil
import struct
import subprocess
import time
import urllib.request
import wave


def synthesize_spoken_audio(text: str, output_wav: str = "spoken_test.wav") -> str:
    """Synthesizes audio: uses macOS 'say' if available, otherwise generates a clean 16kHz 16-bit PCM WAV."""
    if shutil.which("say"):
        print(f"Synthesizing spoken audio via macOS TTS: '{text}'...")
        cmd = [
            "say", text,
            "-o", output_wav,
            "--data-format=LEI16@16000"
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"macOS 'say' not found; generating synthetic 16kHz PCM WAV waveform...")
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)
        with wave.open(output_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            frames = bytearray()
            for i in range(num_samples):
                val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                frames.extend(struct.pack("<h", val))
            w.writeframes(frames)

    size_kb = os.path.getsize(output_wav) / 1024
    print(f"Created WAV file: {output_wav} ({size_kb:.1f} KB, 16kHz 16-bit mono)")
    return output_wav


def run_audio_benchmark(
    audio_file: str,
    prompt: str = "Transcribe the audio exactly.",
    model: str = "nemotron-3.5-bolt:e4b-mxfp8"
):
    print("=" * 84)
    print(f"  MULTIMODAL AUDIO INFERENCE BENCHMARK ({model})")
    print("=" * 84)

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_dur_sec = len(audio_bytes) / (16000 * 2)  # 16kHz 16-bit mono

    print(f"Audio File:        {audio_file}")
    print(f"Audio Duration:    ~{audio_dur_sec:.2f} seconds")
    print(f"User Prompt:       '{prompt}'\n")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_audio],  # Ollama passes multimodal blobs via images array
        "stream": False,
        "options": {
            "num_predict": 1024,
            "temperature": 0.2,
        }
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print("Sending audio waveform to sound encoder & transformer...")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_sec = time.perf_counter() - t0

    prefill_tokens = data.get("prompt_eval_count", 0)
    prefill_ms = data.get("prompt_eval_duration", 1) / 1e6
    decode_tokens = data.get("eval_count", 0)
    decode_ms = data.get("eval_duration", 1) / 1e6
    decode_tps = (decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0
    tpot_ms = (decode_ms / decode_tokens) if decode_tokens > 0 else 0

    print("\n--- INFERENCE TELEMETRY ---")
    print(f"Audio + Prompt Tokens (Prefill): {prefill_tokens} tokens")
    print(f"Audio Encoder + Prefill Latency: {prefill_ms:.2f} ms ({prefill_ms/1000.0:.2f} s)")
    print(f"Generated Decode Tokens:         {decode_tokens} tokens")
    print(f"Decode Throughput:               {decode_tps:.1f} tokens/sec")
    print(f"TPOT (Inter-Token Latency):      {tpot_ms:.1f} ms/token")
    print(f"Total Wall Clock Time:           {wall_sec:.2f} s")
    print("-" * 84)

    thinking = data.get("thinking", "").strip()
    if thinking:
        print(f"Internal Audio Reasoning:\n{thinking[:250]}...\n")

    print(f"Final Model Response:\n{data.get('response', '').strip()}")
    print("=" * 84)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Audio Inference Benchmark")
    parser.add_argument("--file", type=str, default=None, help="Path to WAV audio file")
    parser.add_argument("--prompt", type=str, default="Transcribe the speech in this audio.", help="Prompt")
    parser.add_argument("--synthesize", type=str, default="The server is running on Apple M5 with sixteen gigabytes of unified memory.", help="Text to speak")
    args = parser.parse_args()

    audio_path = args.file
    if not audio_path:
        audio_path = synthesize_spoken_audio(args.synthesize)

    run_audio_benchmark(audio_file=audio_path, prompt=args.prompt)
