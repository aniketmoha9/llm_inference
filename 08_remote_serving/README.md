# Module 08: Remote LLM Hosting & Access from Anywhere

This guide explains how to turn your Apple M5 Mac into a dedicated, self-hosted LLM server and access your models (`bonsai-2`, `nemotron-3.5`, etc.) from your iPhone, Android, or secondary laptop from anywhere in the world.

---

## 🎛️ Four Unified Interfaces

| Interface | URL | Purpose | Key Features |
| :--- | :--- | :--- | :--- |
| **Mobile Chat UI** | `http://<IP>:8000/` | Fast conversation | ChatGPT-style clean UI, auto-downscaled photos, audio attachments, SSE token streaming |
| **LLM Playground** | `http://<IP>:8000/playground` | Parameter exploration & Arena | System prompt tuning, sliders (Temp, Top-P, Max Tokens, Context), presets, **Side-by-Side Model Arena**, 1-click code export (cURL/Python) |
| **Telemetry Dashboard** | `http://<IP>:8000/dashboard` | Hardware & serving metrics | Live decode speed (tok/s), TPOT (ms), TTFT (ms), RAM/VRAM gauge, connected devices (Tailscale vs LAN) |
| **API Docs & Testing Console** | `http://<IP>:8000/docs` | Drop-in API integration | Interactive web tester, OpenAI SDK / cURL / Python / JS snippets, Cursor & Continue IDE setup |

---

## 🔌 Drop-in OpenAI-Compatible API Endpoints

The server exposes drop-in OpenAI-compatible endpoints so you can connect your locally hosted models directly into external applications, scripts, agents, and IDEs (Cursor, Continue.dev, Aider, Open WebUI) from your LAN or anywhere via Tailscale.

### Base URL:
```text
http://<your-mac-ip>:8000/v1
```
*(e.g., `http://192.168.1.11:8000/v1` locally or `http://<tailscale-ip>:8000/v1` over VPN)*

### Supported Endpoints:

| Method | Endpoint | Description | Compatible With |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | Multi-turn chat completion (`stream: true` or `false`) | Official OpenAI SDK, LangChain, LlamaIndex, Cursor, Continue |
| `GET` | `/v1/models` | List all available local models | OpenAI client model discovery |
| `POST` | `/api/generate` | Single prompt generation with raw token stats | Ollama native scripts, automation webhooks |
| `GET` | `/api/metrics` | Real-time hardware telemetry and request history | Custom monitoring dashboards, Prometheus scrapers |
| `GET` | `/docs` | Interactive Swagger-style documentation & testing console | Any web browser |

---

### 💻 Client Code Examples

#### 1. Official OpenAI Python SDK
Install: `pip install openai`

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.11:8000/v1",  # Or your Tailscale IP
    api_key="not-needed"                     # Local server requires no API key
)

# Streaming Response
stream = client.chat.completions.create(
    model="bonsai-2:latest",
    messages=[
        {"role": "system", "content": "You are a concise engineering assistant."},
        {"role": "user", "content": "Explain KV caching in two sentences."}
    ],
    temperature=0.2,
    stream=True
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

#### 2. cURL (Terminal / Bash)

```bash
# Non-streaming
curl -X POST http://192.168.1.11:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-2:latest",
    "messages": [{"role": "user", "content": "What is 25 * 4?"}],
    "temperature": 0.0
  }'

# Real-time SSE Streaming
curl -N -X POST http://192.168.1.11:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-2:latest",
    "messages": [{"role": "user", "content": "List 3 advantages of quantization."}],
    "stream": true
  }'
```

#### 3. IDE Integration: Cursor & Continue.dev

You can use your Apple Silicon Mac as your dedicated AI coding copilot in Cursor or VSCode:

- **Continue.dev (`~/.continue/config.json`)**:
  ```json
  {
    "models": [
      {
        "title": "Local Apple M5 (Bonsai-2)",
        "provider": "openai",
        "model": "bonsai-2:latest",
        "apiBase": "http://192.168.1.11:8000/v1",
        "apiKey": "none"
      }
    ]
  }
  ```
- **Cursor Settings**:
  - Open **Cursor Settings > Models > OpenAI API Key**.
  - Enable **Override OpenAI Base URL**: Set to `http://192.168.1.11:8000/v1`.
  - Add your model name (e.g. `bonsai-2:latest`).

---

---

## 🎛️ LLM Playground Features (`/playground`)

The Playground provides a simple, intuitive developer interface for testing and comparing models:

1. **Single Model Exploration**:
   * **System Prompt / Persona**: Collapsible instructions panel for setting behavior (e.g. Code Architect, JSON Extractor).
   * **Inference Sliders**: Direct real-time control over `Temperature` ($0.0 \to 1.5$), `Top-P` ($0.05 \to 1.0$), `Max Tokens` ($128 \to 4096$), `Context Window` ($4\text{k} \to 32\text{k}$), and `Stop Sequences`.
   * **Preset Templates**: 1-click presets for coding, reasoning, summarization, creative writing, JSON extraction, and audio analysis.
   * **Live Telemetry Banner**: Displays real-time decode speed ($\text{tok/s}$), TTFT ($\text{ms}$), context tokens, output tokens, and stop reasons.
   * **Export Code**: Generates clean copy-pasteable `cURL` commands and `Python` scripts matching your active playground settings.

2. **⚔️ Side-by-Side Model Arena**:
   * Send the same prompt concurrently to **Model A** (e.g. `bonsai-2:latest` - 2.5 GB Q2_0) and **Model B** (e.g. `nemotron-3.5-bolt:e4b-mxfp8` - 11 GB mxfp8).
   * Watch both models stream in parallel and compare:
     * **Time-to-First-Token (TTFT)**
     * **Decode throughput (tokens/sec)**
     * **Response quality and nuance**

---

## 📊 Real-Time Inference Telemetry Dashboard (`/dashboard`)

A dedicated, remotely accessible dashboard is built directly into the server at `/dashboard`:

- **On Home Wi-Fi**: `http://192.168.1.11:8000/dashboard`
- **From Anywhere (Tailscale)**: `http://<your-tailscale-ip>:8000/dashboard`
- **JSON Metrics API**: `http://192.168.1.11:8000/api/metrics`

### What the Dashboard Monitors in Real Time (Auto-refresh every 2s):
1. **Decode Throughput**: Token generation speed (e.g. `38.3 tok/s`, memory-bandwidth bound GEMV).
2. **TPOT (Inter-Token Latency)**: Time per output token (e.g. `26.1 ms/token`).
3. **TTFT (Time-To-First-Token)**: Prefill compute time (e.g. `204 ms`, compute-bound GEMM).
4. **Unified RAM & VRAM**: macOS unified memory allocation gauge (`used / total GB`).
5. **Active Loaded Model**: Model name, resident VRAM footprint (`2.5 GB`), and runner (`llamacpp Metal GPU` vs `mlx`).
6. **Network & Connected Clients**: Distinguishes between **Local Wi-Fi** clients and **Remote Tailscale 5G/VPN** clients.
7. **Live Activity Stream**: Real-time log of the last 30 requests with prompts, modality (`🎵 Audio`, `🖼️ Vision`, or `📝 Text`), and performance breakdown.
8. **Automated On-the-Fly Audio Transcoding**: Ingests mobile audio recordings (`.m4a`, `.caf`, `.mp3`, `.ogg`, `.wav`) and downmixes to 16kHz mono WAV in ~15 ms before forwarding to multimodal speech models (`nemotron-3.5-bolt:e4b-mxfp8`).

---

## 🏗️ Architecture Overview

```
 [ Your iPhone / Android ]            [ Another Laptop (Coffee Shop) ]
            │                                       │
            │ (Cellular 5G / Remote Wi-Fi)          │
            ▼                                       ▼
  ┌──────────────────────────────────────────────────────────┐
  │   Secure Networking Layer (Tailscale WireGuard Mesh)     │
  │   • Encrypted peer-to-peer connection                    │
  │   • No port forwarding, no public IP exposure            │
  └────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │        Your Apple M5 Mac (Host Server)       │
        │                                              │
        │  ┌────────────────────────────────────────┐  │
        │  │ Web Chat UI (Port 8000)                │  │
        │  │ • Mobile-responsive browser interface  │  │
        │  │ • Token streaming & model selector     │  │
        │  └───────────────────┬────────────────────┘  │
        │                      │                       │
        │  ┌───────────────────▼────────────────────┐  │
        │  │ Ollama Inference Engine (Port 11434)   │  │
        │  │ • Bound to 0.0.0.0                     │  │
        │  │ • Running bonsai-2, nemotron, etc.     │  │
        │  └────────────────────────────────────────┘  │
        └──────────────────────────────────────────────┘
```

---

## 🚀 Step 1: Configure Ollama to Accept Remote Connections

By default, Ollama only listens on `127.0.0.1` (localhost only). To allow other devices to connect, it must bind to `0.0.0.0`.

### If launching from Terminal:
```bash
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_ORIGINS="*"
ollama serve
```

### If using the macOS Menu Bar Ollama App:
Configure macOS system-level launchctl variables:
```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
launchctl setenv OLLAMA_ORIGINS="*"
```
Then quit Ollama from the menu bar and reopen it.

---

## 🔒 Step 2: Access from Anywhere via Tailscale (Recommended)

**Tailscale** creates an encrypted, zero-configuration WireGuard VPN mesh between your devices. It is free, takes 2 minutes to set up, and requires **no router port forwarding**.

1. **Install Tailscale on your Mac**:
   - Download from the [Mac App Store](https://apps.apple.com/app/tailscale/id1475387142) or [tailscale.com](https://tailscale.com).
   - Sign in (Google / Apple ID / GitHub).
   - Your Mac will get a unique private IP (e.g. `100.85.120.45`) and a name (e.g. `my-mac`).

2. **Install Tailscale on your Phone / Other Laptop**:
   - Install Tailscale from the iOS App Store or Google Play Store.
   - Sign in with the **same account**.

3. **Connect**:
   - Your phone is now securely linked to your Mac over WireGuard, whether you are on mobile data (5G), airport Wi-Fi, or home.
   - From your phone's browser or apps, you can reach your Mac at `http://<your-tailscale-ip>:11434` or `http://<your-tailscale-ip>:8000`.

---

## 📱 Step 3: Choose Your Client Interface

You have two great options to interact with your models from your phone or other devices:

### Option A: Native Mobile Apps (Zero web setup)
- **iOS (iPhone/iPad)**:
  - **Enchanted** (Free & Open Source on the iOS App Store): Built specifically for Ollama!
  - In Enchanted settings, enter your Mac's IP/Tailscale IP (`http://100.x.y.z:11434`).
  - It automatically detects your models (`bonsai-2:latest`, `nemotron-3.5-bolt`, etc.) and provides a native, beautiful iOS chat UI with voice input and image attachment support!
- **Android / Cross-Platform**:
  - **Chatbox** (Available on iOS, Android, macOS, Windows).
  - Point the host to `http://100.x.y.z:11434`.

### Option B: The Built-in Web Chat Server (`web_chat_server.py`)
This repository includes a lightweight, zero-dependency mobile-friendly Web UI.

Start the web server on your Mac:
```bash
cd /Users/eigenket/vibe_code/llm_inference
source .venv/bin/activate
python3 08_remote_serving/web_chat_server.py
```

Now, open any web browser on your phone or other laptop:
- **On Home Wi-Fi**: `http://192.168.1.11:8000`
- **From Anywhere (via Tailscale)**: `http://<your-tailscale-ip>:8000`

---

## ⚡ Step 4: Keep Your Mac Awake (Clamshell / Server Mode)

To ensure your Mac continues serving while you are away or when the lid is closed:
1. Open **System Settings > Energy Saver** (or Battery).
2. Enable **"Prevent automatic sleeping on power adapter when the display is off"**.
3. Plug in the power adapter if closing the lid.
4. Alternatively, run the built-in macOS `caffeinate` command in terminal:
   ```bash
   caffeinate -s
   ```
