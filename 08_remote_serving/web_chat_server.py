#!/usr/bin/env python3
"""
High-Performance Streaming Web Chat Server & Real-Time Telemetry Dashboard.
Serves:
1. /           -> Mobile-first ChatGPT-style UI with SSE streaming & image attachment
2. /dashboard  -> Remotely accessible live telemetry & inference engineering dashboard
3. /api/metrics-> Real-time JSON metrics endpoint (hardware, latency, throughput, clients)
"""

import base64
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
import wave

PORT = 8000
OLLAMA_BASE_URL = "http://localhost:11434"
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")


# ==============================================================================
# Global In-Memory Metrics Store
# ==============================================================================
class MetricsTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.multimodal_requests = 0
        self.latest_decode_tps = 0.0
        self.latest_prefill_ms = 0.0
        self.latest_tpot_ms = 0.0
        self.recent_requests = []  # Max 30 requests
        self.clients = {}  # ip -> {"count": int, "last_seen": str, "type": str}
        self.server_start_time = datetime.now()

    def record_request(self, client_ip: str, model: str, prompt: str, media_type: str = "text"):
        with self.lock:
            self.total_requests += 1
            if media_type in ("image", "audio"):
                self.multimodal_requests += 1

            client_type = "Local Mac"
            if client_ip.startswith("100."):
                client_type = "Tailscale Remote (5G/VPN)"
            elif client_ip.startswith("192.168.") or client_ip.startswith("10.") or client_ip.startswith("172."):
                client_type = "Home Wi-Fi (LAN)"

            if client_ip not in self.clients:
                self.clients[client_ip] = {"count": 0, "type": client_type, "last_seen": ""}
            self.clients[client_ip]["count"] += 1
            self.clients[client_ip]["last_seen"] = datetime.now().strftime("%H:%M:%S")

    def record_completion(self, client_ip: str, model: str, prompt: str, media_type: str,
                          p_cnt: int, p_ms: float, d_cnt: int, d_tps: float, tpot: float):
        with self.lock:
            self.total_prefill_tokens += p_cnt
            self.total_decode_tokens += d_cnt
            self.latest_decode_tps = d_tps
            self.latest_prefill_ms = p_ms
            self.latest_tpot_ms = tpot

            client_type = "Local"
            if client_ip.startswith("100."):
                client_type = "Tailscale"
            elif client_ip.startswith("192.168."):
                client_type = "Wi-Fi"

            req_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "client_ip": client_ip,
                "client_type": client_type,
                "model": model,
                "prompt": (prompt[:45] + "...") if len(prompt) > 45 else prompt,
                "media_type": media_type,
                "has_image": (media_type == "image"),
                "prefill_tokens": p_cnt,
                "prefill_ms": round(p_ms, 1),
                "decode_tokens": d_cnt,
                "decode_tps": round(d_tps, 1),
                "tpot_ms": round(tpot, 1),
            }
            self.recent_requests.insert(0, req_entry)
            if len(self.recent_requests) > 30:
                self.recent_requests.pop()

    def get_snapshot(self) -> dict:
        with self.lock:
            uptime = str(datetime.now() - self.server_start_time).split(".")[0]
            return {
                "uptime": uptime,
                "total_requests": self.total_requests,
                "total_prefill_tokens": self.total_prefill_tokens,
                "total_decode_tokens": self.total_decode_tokens,
                "multimodal_requests": self.multimodal_requests,
                "latest_decode_tps": self.latest_decode_tps,
                "latest_prefill_ms": self.latest_prefill_ms,
                "latest_tpot_ms": self.latest_tpot_ms,
                "recent_requests": list(self.recent_requests),
                "clients": dict(self.clients),
            }


METRICS = MetricsTracker()


def log_event(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def get_hardware_telemetry() -> dict:
    """Gathers real-time macOS unified memory and Ollama runner status."""
    hw = {
        "memory": {"used_gb": 0.0, "free_gb": 16.0, "total_gb": 16.0, "percent": 0.0},
        "loaded_model": {"name": "Idle / On-Demand", "size": "0 GB", "runner": "N/A", "processor": "Idle"}
    }

    # 1. macOS Unified Memory via vm_stat
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
        stats = {}
        for line in vm.stdout.split("\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    stats[k.strip()] = int(v.strip().rstrip("."))
                except ValueError:
                    pass
        page_size = 16384  # macOS arm64 standard page size
        active_b = stats.get("Pages active", 0) * page_size
        wired_b = stats.get("Pages wired down", 0) * page_size
        comp_b = stats.get("Pages occupied by compressor", 0) * page_size
        used_b = active_b + wired_b + comp_b
        total_b = 16 * 1024 * 1024 * 1024

        hw["memory"]["used_gb"] = round(used_b / (1024**3), 2)
        hw["memory"]["free_gb"] = round(max(0, total_b - used_b) / (1024**3), 2)
        hw["memory"]["percent"] = round((used_b / total_b) * 100.0, 1)
    except Exception:
        pass

    # 2. Ollama Loaded Models via 'ollama ps'
    try:
        res = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=2)
        lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
        if len(lines) > 1:
            p = lines[1].split()
            if len(p) >= 6:
                hw["loaded_model"] = {
                    "name": p[0],
                    "size": p[2] + " " + p[3] if len(p) > 6 else p[2],
                    "processor": p[4] if len(p) > 6 else p[3],
                    "runner": p[5] if len(p) > 6 else p[5],
                }
    except Exception:
        pass

    return hw


def ensure_compatible_audio(b64_data: str) -> str:
    """
    Transcodes any incoming mobile audio (M4A, AAC, MP3, CAF, AIFF, or non-16kHz WAV)
    into standard 16kHz 16-bit mono PCM WAV using macOS afconvert.
    Ollama / Gemma 4 / Nemotron sound_encoder strictly requires PCM WAV.
    """
    in_path, out_path = None, None
    try:
        raw = base64.b64decode(b64_data)
        # Fast path: check if already 16kHz 16-bit mono WAV
        try:
            with wave.open(io.BytesIO(raw), "rb") as w:
                if w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2:
                    return b64_data
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".in_audio", delete=False) as f_in:
            f_in.write(raw)
            in_path = f_in.name

        out_path = in_path + ".wav"
        cmd = ["afconvert", in_path, "-o", out_path, "-d", "LEI16@16000", "-c", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f_out:
                converted_b64 = base64.b64encode(f_out.read()).decode("utf-8")
            log_event(f"[Audio Transcoder] Transcoded {os.path.basename(in_path)} -> 16kHz mono WAV ({len(converted_b64)} b64 chars)")
            return converted_b64
        else:
            log_event(f"[Audio Transcoder Warning] afconvert failed: {res.stderr.strip()}")
            return b64_data
    except Exception as exc:
        log_event(f"[Audio Transcoder Error] {exc}")
        return b64_data
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ==============================================================================
# HTML Templates
# ==============================================================================
CHAT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Apple M5 Private LLM</title>
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #151e2e;
            --bg-tertiary: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
            --user-bubble: #2563eb;
            --bot-bubble: #151e2e;
            --thinking-bg: #0f172a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-primary); height: 100vh; height: 100dvh; display: flex; flex-direction: column; overflow: hidden; }
        
        header { background: var(--bg-secondary); border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; z-index: 10; gap: 8px; }
        .logo-group { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
        .pulse-dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e; }
        h1 { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.02em; }
        .header-actions { display: flex; align-items: center; gap: 8px; }
        select { background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); padding: 6px 10px; border-radius: 8px; font-size: 0.85rem; outline: none; cursor: pointer; max-width: 150px; }
        .dash-link-btn { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 6px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 500; text-decoration: none; display: flex; align-items: center; gap: 4px; transition: all 0.2s; white-space: nowrap; }
        .dash-link-btn:hover { background: #38bdf8; color: #0f172a; }
        
        #chat-container { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth; }
        .message-wrapper { display: flex; flex-direction: column; max-width: 88%; animation: fadeIn 0.2s ease; }
        .message-wrapper.user { align-self: flex-end; }
        .message-wrapper.assistant { align-self: flex-start; width: 88%; }
        .bubble { padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 0.95rem; word-break: break-word; white-space: pre-wrap; }
        .user .bubble { background: var(--user-bubble); color: white; border-bottom-right-radius: 4px; }
        .assistant .bubble { background: var(--bot-bubble); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
        .bubble img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
        
        details.thinking-box { background: var(--thinking-bg); border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.82rem; color: var(--text-secondary); }
        details.thinking-box summary { cursor: pointer; font-weight: 500; color: #38bdf8; outline: none; user-select: none; }
        details.thinking-box .thinking-content { margin-top: 8px; white-space: pre-wrap; max-height: 180px; overflow-y: auto; font-family: ui-monospace, Menlo, Monaco, monospace; line-height: 1.4; color: #cbd5e1; }
        
        .telemetry-tag { font-size: 0.72rem; color: var(--text-secondary); margin-top: 4px; padding: 0 4px; }
        
        #preview-bar { padding: 6px 16px; background: var(--bg-secondary); border-top: 1px solid var(--border); display: none; align-items: center; gap: 10px; }
        #preview-bar img { width: 44px; height: 44px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
        .remove-img-btn { background: #ef4444; color: white; border: none; border-radius: 50%; width: 20px; height: 20px; font-size: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center; }

        footer { background: var(--bg-secondary); border-top: 1px solid var(--border); padding: 10px 14px; display: flex; align-items: center; gap: 10px; position: sticky; bottom: 0; }
        .icon-btn { background: none; border: none; color: var(--text-secondary); cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 8px; border-radius: 50%; transition: background 0.2s; }
        .icon-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }
        textarea { flex: 1; background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); border-radius: 20px; padding: 10px 16px; font-size: 0.95rem; resize: none; max-height: 120px; height: 42px; outline: none; line-height: 1.4; }
        textarea::placeholder { color: var(--text-secondary); }
        .send-btn { background: var(--accent); color: var(--bg-primary); border: none; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-weight: bold; transition: transform 0.1s, background 0.2s; flex-shrink: 0; }
        .send-btn:hover { background: var(--accent-hover); }
        .send-btn:active { transform: scale(0.95); }
        .send-btn:disabled { background: var(--bg-tertiary); color: var(--text-secondary); cursor: not-allowed; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <div class="pulse-dot"></div>
            <h1>Apple M5</h1>
        </div>
        <div class="header-actions">
            <select id="model-select">
                <option value="bonsai-2:latest">bonsai-2:latest</option>
            </select>
            <a href="/playground" class="dash-link-btn" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(168, 85, 247, 0.4);" title="Open LLM Playground">🎛️ Playground</a>
            <a href="/dashboard" class="dash-link-btn" title="View Live Telemetry Dashboard">📊 Metrics</a>
        </div>
    </header>

    <div id="chat-container">
        <div class="message-wrapper assistant">
            <div class="bubble">Connected to Apple M5 Server. Ready to chat or analyze images! Real-time streaming active.</div>
        </div>
    </div>

    <div id="preview-bar">
        <img id="attached-preview" src="" alt="preview" style="display:none;">
        <span id="attached-label" style="font-size: 0.82rem; color: var(--text-secondary);">Media attached</span>
        <button class="remove-img-btn" onclick="clearMedia()">✕</button>
    </div>

    <footer>
        <input type="file" id="media-upload" accept="image/*,audio/*" style="display: none;" onchange="handleMedia(event)">
        <button class="icon-btn" onclick="document.getElementById('media-upload').click()" title="Attach Photo or Audio">
            <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path></svg>
        </button>
        <textarea id="user-input" placeholder="Type a message..." rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <button id="send-button" class="send-btn" onclick="sendMessage()">
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7"></path></svg>
        </button>
    </footer>

    <script>
        let attachedBase64 = null;
        let attachedType = null; // 'image' or 'audio'

        async function loadModels() {
            try {
                const res = await fetch('/api/tags');
                const data = await res.json();
                const select = document.getElementById('model-select');
                select.innerHTML = '';
                
                const validModels = data.models.filter(m => {
                    const caps = m.capabilities || [];
                    return caps.includes('completion') || m.name.includes('bonsai') || m.name.includes('nemotron');
                });
                const modelsToShow = validModels.length > 0 ? validModels : data.models;

                modelsToShow.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.name;
                    if (m.name.includes('bonsai-2')) opt.selected = true;
                    select.appendChild(opt);
                });
            } catch (e) {
                console.log('Error loading models', e);
            }
        }
        loadModels();

        function handleMedia(e) {
            const file = e.target.files[0];
            if (!file) return;

            const fname = file.name.toLowerCase();
            const isAudio = file.type.startsWith('audio/') || 
                            fname.endsWith('.wav') || fname.endsWith('.mp3') || 
                            fname.endsWith('.m4a') || fname.endsWith('.ogg') || 
                            fname.endsWith('.caf') || fname.endsWith('.aac') || 
                            fname.endsWith('.flac') || fname.endsWith('.aif') || 
                            fname.endsWith('.aiff');

            if (isAudio) {
                const reader = new FileReader();
                reader.onload = function(evt) {
                    const dataUrl = evt.target.result;
                    attachedBase64 = dataUrl.split(',')[1];
                    attachedType = 'audio';
                    document.getElementById('attached-preview').style.display = 'none';
                    document.getElementById('attached-label').innerHTML = `🎵 Audio: <strong>${escapeHtml(file.name)}</strong>`;
                    document.getElementById('preview-bar').style.display = 'flex';

                    // Auto-select nemotron model for audio analysis
                    const select = document.getElementById('model-select');
                    for (let i = 0; i < select.options.length; i++) {
                        if (select.options[i].value.includes('nemotron')) {
                            select.selectedIndex = i;
                            break;
                        }
                    }
                };
                reader.readAsDataURL(file);
            } else {
                const reader = new FileReader();
                reader.onload = function(evt) {
                    const img = new Image();
                    img.onload = function() {
                        const canvas = document.createElement('canvas');
                        let w = img.width;
                        let h = img.height;
                        const maxDim = 1024;
                        if (w > maxDim || h > maxDim) {
                            if (w > h) {
                                h = Math.round((h * maxDim) / w);
                                w = maxDim;
                            } else {
                                w = Math.round((w * maxDim) / h);
                                h = maxDim;
                            }
                        }
                        canvas.width = w;
                        canvas.height = h;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, w, h);
                        const resizedDataUrl = canvas.toDataURL('image/jpeg', 0.85);
                        attachedBase64 = resizedDataUrl.split(',')[1];
                        attachedType = 'image';
                        const pImg = document.getElementById('attached-preview');
                        pImg.src = resizedDataUrl;
                        pImg.style.display = 'block';
                        document.getElementById('attached-label').textContent = 'Image attached';
                        document.getElementById('preview-bar').style.display = 'flex';
                    };
                    img.src = evt.target.result;
                };
                reader.readAsDataURL(file);
            }
        }

        function clearMedia() {
            attachedBase64 = null;
            attachedType = null;
            document.getElementById('media-upload').value = '';
            document.getElementById('preview-bar').style.display = 'none';
        }

        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 120) + 'px';
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const prompt = input.value.trim();
            if (!prompt && !attachedBase64) return;

            const chat = document.getElementById('chat-container');
            const model = document.getElementById('model-select').value;
            const sendBtn = document.getElementById('send-button');

            // Render User Message
            const userWrap = document.createElement('div');
            userWrap.className = 'message-wrapper user';
            let userContent = escapeHtml(prompt);
            if (attachedBase64) {
                if (attachedType === 'audio') {
                    userContent += `<br><audio controls src="data:audio/wav;base64,${attachedBase64}" style="margin-top:8px; max-width: 100%; height: 36px;"></audio>`;
                } else {
                    userContent += `<br><img src="data:image/jpeg;base64,${attachedBase64}">`;
                }
            }
            userWrap.innerHTML = `<div class="bubble">${userContent}</div>`;
            chat.appendChild(userWrap);

            const mediaToSend = attachedBase64;
            const mediaTypeToSend = attachedType;
            clearMedia();
            input.value = '';
            input.style.height = '42px';
            sendBtn.disabled = true;

            // Render Assistant Placeholder
            const botWrap = document.createElement('div');
            botWrap.className = 'message-wrapper assistant';

            const thinkBox = document.createElement('details');
            thinkBox.className = 'thinking-box';
            thinkBox.style.display = 'none';
            thinkBox.innerHTML = '<summary>Thinking...</summary><div class="thinking-content"></div>';
            botWrap.appendChild(thinkBox);

            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = '...';
            botWrap.appendChild(bubble);

            const telemetry = document.createElement('div');
            telemetry.className = 'telemetry-tag';
            botWrap.appendChild(telemetry);

            chat.appendChild(botWrap);
            chat.scrollTop = chat.scrollHeight;

            let thinkingContent = '';
            let responseContent = '';

            try {
                const response = await fetch('/api/stream_chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        prompt: prompt,
                        image: mediaToSend,
                        media_type: mediaTypeToSend
                    })
                });

                if (!response.ok) {
                    const errText = await response.text();
                    bubble.textContent = 'Server Error: ' + errText;
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const jsonStr = line.slice(6);
                            if (!jsonStr) continue;
                            try {
                                const payload = JSON.parse(jsonStr);

                                if (payload.type === 'thinking') {
                                    thinkingContent += payload.text;
                                    thinkBox.style.display = 'block';
                                    thinkBox.querySelector('.thinking-content').textContent = thinkingContent;
                                    chat.scrollTop = chat.scrollHeight;
                                } else if (payload.type === 'token') {
                                    if (responseContent === '') bubble.textContent = '';
                                    responseContent += payload.text;
                                    bubble.textContent = responseContent;
                                    chat.scrollTop = chat.scrollHeight;
                                } else if (payload.type === 'done') {
                                    const meta = payload.meta;
                                    const reason = payload.done_reason;
                                    if (meta) {
                                        let note = reason === 'length' ? ' | <span style="color:#f87171;">⚠️ Truncated by token limit</span>' : '';
                                        telemetry.innerHTML = `⚡ Speed: ${meta.decode_tps.toFixed(1)} tok/s | Prefill: ${meta.prefill_ms.toFixed(0)} ms | TPOT: ${meta.tpot_ms.toFixed(1)} ms${note}`;
                                    }
                                } else if (payload.type === 'error') {
                                    bubble.textContent = 'Error: ' + payload.text;
                                }
                            } catch (e) {}
                        }
                    }
                }

                if (!responseContent && thinkingContent) {
                    bubble.innerHTML = "<span style='color: #f87171;'>⚠️ The model was cut off during internal reasoning before emitting the answer (context/token limit reached). Try asking a shorter prompt or zooming in on the image.</span>";
                }

            } catch (err) {
                bubble.textContent = "Network Error: " + err.message;
            } finally {
                sendBtn.disabled = false;
                chat.scrollTop = chat.scrollHeight;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Inference Telemetry Dashboard</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #131b2e;
            --bg-card-hover: #1b2640;
            --border: #23314f;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --cyan: #38bdf8;
            --green: #22c55e;
            --purple: #a855f7;
            --orange: #f97316;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); padding: 16px; min-height: 100vh; }
        
        .container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        
        /* Header */
        header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .brand { display: flex; align-items: center; gap: 12px; }
        .status-badge { display: flex; align-items: center; gap: 6px; background: rgba(34, 197, 94, 0.15); color: var(--green); border: 1px solid rgba(34, 197, 94, 0.3); padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
        .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
        h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
        .header-actions { display: flex; align-items: center; gap: 10px; }
        .btn { background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: 7px 14px; border-radius: 8px; font-size: 0.85rem; font-weight: 500; cursor: pointer; text-decoration: none; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }
        .btn:hover { background: var(--bg-card-hover); border-color: var(--cyan); }
        .btn-primary { background: var(--cyan); color: #0a0e17; border-color: var(--cyan); font-weight: 600; }
        .btn-primary:hover { background: #0ea5e9; }
        
        /* KPI Grid */
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; gap: 10px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; color: var(--text-muted); font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
        .card-value { font-size: 2rem; font-weight: 700; color: #fff; line-height: 1.1; }
        .card-sub { font-size: 0.82rem; color: var(--text-muted); display: flex; align-items: center; justify-content: space-between; }
        .tag { font-size: 0.72rem; padding: 2px 8px; border-radius: 6px; font-weight: 600; }
        .tag-cyan { background: rgba(56, 189, 248, 0.15); color: var(--cyan); }
        .tag-green { background: rgba(34, 197, 94, 0.15); color: var(--green); }
        .tag-purple { background: rgba(168, 85, 247, 0.15); color: var(--purple); }
        .tag-orange { background: rgba(249, 115, 22, 0.15); color: var(--orange); }

        /* Progress Bar */
        .progress-bar-bg { background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 6px; }
        .progress-bar-fill { background: linear-gradient(90deg, var(--cyan), var(--purple)); height: 100%; width: 0%; transition: width 0.4s ease; }

        /* Tables & Lists */
        .section-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
        .table-wrap { overflow-x: auto; background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; }
        th { background: #1a233a; color: var(--text-muted); padding: 12px 14px; font-weight: 600; border-bottom: 1px solid var(--border); }
        td { padding: 12px 14px; border-bottom: 1px solid rgba(51, 65, 85, 0.4); color: #cbd5e1; white-space: nowrap; }
        tr:last-child td { border-bottom: none; }
        tr:hover td { background: rgba(255, 255, 255, 0.02); }

        .client-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 500; }
        .client-tailscale { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
        .client-lan { background: rgba(56, 189, 248, 0.18); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.3); }
        .client-local { background: rgba(148, 163, 184, 0.18); color: #cbd5e1; }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <div class="status-badge"><div class="pulse"></div> LIVE</div>
                <div>
                    <h1>Apple M5 Inference Server</h1>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Unified Memory Architecture &bull; Uptime: <span id="uptime-val">--</span></span>
                </div>
            </div>
            <div class="header-actions">
                <button class="btn" onclick="fetchMetrics()">🔄 Refresh</button>
                <a href="/playground" class="btn" style="border-color: rgba(168, 85, 247, 0.4); color: #c084fc;">🎛️ Playground</a>
                <a href="/" class="btn btn-primary">💬 Open Chat UI</a>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="grid-4">
            <div class="card">
                <div class="card-header">
                    <span>Decode Speed</span>
                    <span class="tag tag-cyan">Memory-Bound</span>
                </div>
                <div class="card-value" id="decode-tps">-- <span style="font-size: 1.1rem; color: var(--text-muted);">tok/s</span></div>
                <div class="card-sub">
                    <span>TPOT (Inter-Token):</span>
                    <strong id="tpot-val" style="color: #fff;">-- ms</strong>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>Time To First Token</span>
                    <span class="tag tag-green">Compute-Bound</span>
                </div>
                <div class="card-value" id="prefill-ms">-- <span style="font-size: 1.1rem; color: var(--text-muted);">ms</span></div>
                <div class="card-sub">
                    <span>Prompt Evaluation:</span>
                    <span id="prefill-sub" style="color: #fff;">GEMM phase</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>Loaded Model</span>
                    <span class="tag tag-purple" id="runner-tag">Metal GPU</span>
                </div>
                <div class="card-value" id="model-name" style="font-size: 1.45rem;">--</div>
                <div class="card-sub">
                    <span>Resident VRAM:</span>
                    <strong id="model-vram" style="color: #fff;">--</strong>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>Volume Handled</span>
                    <span class="tag tag-orange">Throughput</span>
                </div>
                <div class="card-value" id="total-reqs">--</div>
                <div class="card-sub">
                    <span>Tokens Streamed:</span>
                    <strong id="total-tokens" style="color: #fff;">-- tokens</strong>
                </div>
            </div>
        </div>

        <!-- Hardware & Clients Row -->
        <div class="grid-4" style="grid-template-columns: 1fr 1fr;">
            <!-- Memory Gauge Card -->
            <div class="card">
                <div class="card-header">
                    <span>Unified Memory (RAM / VRAM)</span>
                    <span class="tag tag-green" id="mem-status-tag">Optimal</span>
                </div>
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 1.1rem; font-weight: 600;">
                        <span id="mem-used">-- GB</span>
                        <span style="color: var(--text-muted);">16.00 GB Total</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="mem-bar" class="progress-bar-fill"></div>
                    </div>
                </div>
                <div class="card-sub">
                    <span>Free Unified Memory:</span>
                    <strong id="mem-free" style="color: #fff;">-- GB</strong>
                </div>
            </div>

            <!-- Connected Clients Summary -->
            <div class="card">
                <div class="card-header">
                    <span>Connected Clients</span>
                    <span class="tag tag-cyan" id="client-count-tag">-- Devices</span>
                </div>
                <div id="clients-list" style="display: flex; flex-direction: column; gap: 8px; margin-top: 4px;">
                    <span style="color: var(--text-muted); font-size: 0.85rem;">No clients logged yet.</span>
                </div>
            </div>
        </div>

        <!-- Live Activity Stream Table -->
        <div>
            <div class="section-title">
                <span>Recent Inference Stream</span>
                <span style="font-size: 0.8rem; color: var(--text-muted);">Auto-refreshing every 2s</span>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Client IP</th>
                            <th>Type</th>
                            <th>Model</th>
                            <th>Prompt Preview</th>
                            <th>Modality</th>
                            <th>Prefill</th>
                            <th>Decode Speed</th>
                            <th>TPOT</th>
                        </tr>
                    </thead>
                    <tbody id="activity-tbody">
                        <tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 24px;">No requests recorded yet. Send a message to see live logs!</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        async function fetchMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();
                renderMetrics(data);
            } catch (err) {
                console.error("Failed to load metrics", err);
            }
        }

        function renderMetrics(data) {
            const m = data.metrics;
            const hw = data.hardware;

            document.getElementById('uptime-val').textContent = m.uptime;

            // KPIs
            document.getElementById('decode-tps').innerHTML = `${m.latest_decode_tps.toFixed(1)} <span style="font-size: 1.1rem; color: var(--text-muted);">tok/s</span>`;
            document.getElementById('tpot-val').textContent = `${m.latest_tpot_ms.toFixed(1)} ms`;
            document.getElementById('prefill-ms').innerHTML = `${m.latest_prefill_ms.toFixed(0)} <span style="font-size: 1.1rem; color: var(--text-muted);">ms</span>`;
            document.getElementById('total-reqs').textContent = `${m.total_requests} reqs`;
            document.getElementById('total-tokens').textContent = `${(m.total_prefill_tokens + m.total_decode_tokens).toLocaleString()} tokens`;

            // Model
            const lm = hw.loaded_model;
            document.getElementById('model-name').textContent = lm.name;
            document.getElementById('model-vram').textContent = lm.size;
            document.getElementById('runner-tag').textContent = lm.runner || 'Metal';

            // Memory
            const mem = hw.memory;
            document.getElementById('mem-used').textContent = `${mem.used_gb.toFixed(2)} GB (${mem.percent}%)`;
            document.getElementById('mem-free').textContent = `${mem.free_gb.toFixed(2)} GB`;
            document.getElementById('mem-bar').style.width = `${mem.percent}%`;

            // Clients List
            const clientsDiv = document.getElementById('clients-list');
            const clientKeys = Object.keys(m.clients);
            document.getElementById('client-count-tag').textContent = `${clientKeys.length} Device${clientKeys.length === 1 ? '' : 's'}`;

            if (clientKeys.length > 0) {
                clientsDiv.innerHTML = clientKeys.map(ip => {
                    const c = m.clients[ip];
                    let badgeClass = 'client-local';
                    if (c.type.includes('Tailscale')) badgeClass = 'client-tailscale';
                    else if (c.type.includes('Wi-Fi')) badgeClass = 'client-lan';
                    return `
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <span style="font-family: ui-monospace, monospace;">${ip}</span>
                            <span class="client-badge ${badgeClass}">${c.type}</span>
                            <span style="color: var(--text-muted); font-size: 0.78rem;">${c.count} reqs &bull; ${c.last_seen}</span>
                        </div>
                    `;
                }).join('');
            }

            // Recent Requests Table
            const tbody = document.getElementById('activity-tbody');
            if (m.recent_requests.length > 0) {
                tbody.innerHTML = m.recent_requests.map(r => {
                    const badge = r.client_type === 'Tailscale' ? 'client-tailscale' : (r.client_type === 'Wi-Fi' ? 'client-lan' : 'client-local');
                    const modIcon = r.media_type === 'audio' ? '🎵 Audio' : ((r.media_type === 'image' || r.has_image) ? '🖼️ Vision' : '📝 Text');
                    return `
                        <tr>
                            <td>${r.timestamp}</td>
                            <td style="font-family: ui-monospace, monospace;">${r.client_ip}</td>
                            <td><span class="client-badge ${badge}">${r.client_type}</span></td>
                            <td style="font-weight: 500; color: #38bdf8;">${r.model}</td>
                            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(r.prompt)}</td>
                            <td>${modIcon}</td>
                            <td>${r.prefill_ms} ms</td>
                            <td><strong>${r.decode_tps} tok/s</strong> (${r.decode_tokens} tok)</td>
                            <td>${r.tpot_ms} ms</td>
                        </tr>
                    `;
                }).join('');
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Fetch metrics on page load and every 2 seconds
        fetchMetrics();
        setInterval(fetchMetrics, 2000);
    </script>
</body>
</html>
"""


PLAYGROUND_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>LLM Inference Playground</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #131b2e;
            --bg-card-hover: #1b2640;
            --border: #23314f;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --cyan: #38bdf8;
            --cyan-hover: #0ea5e9;
            --purple: #c084fc;
            --purple-bg: rgba(168, 85, 247, 0.15);
            --green: #22c55e;
            --red: #f87171;
            --thinking-bg: #0b1120;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }
        
        /* Top Navigation Header */
        header { background: var(--bg-card); border-bottom: 1px solid var(--border); padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; gap: 12px; z-index: 20; }
        .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; }
        .brand .badge { background: linear-gradient(135deg, #38bdf8, #818cf8); color: #0a0e17; padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }
        
        .header-actions { display: flex; align-items: center; gap: 10px; }
        .btn-nav { background: rgba(255, 255, 255, 0.05); color: var(--text-main); border: 1px solid var(--border); padding: 6px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 500; text-decoration: none; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }
        .btn-nav:hover { background: var(--bg-card-hover); border-color: var(--cyan); color: #fff; }
        
        /* Subnav / Controls */
        .subnav { background: #0f1524; border-bottom: 1px solid var(--border); padding: 8px 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
        .mode-tabs { display: flex; gap: 4px; background: rgba(0,0,0,0.3); padding: 3px; border-radius: 8px; border: 1px solid var(--border); }
        .mode-tab { background: transparent; border: none; color: var(--text-muted); padding: 6px 14px; border-radius: 6px; font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .mode-tab.active { background: var(--bg-card); color: var(--cyan); box-shadow: 0 1px 4px rgba(0,0,0,0.3); }
        .mode-tab.arena-tab.active { color: var(--purple); }
        
        .preset-group { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--text-muted); }
        select.preset-select { background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: 5px 10px; border-radius: 6px; font-size: 0.82rem; outline: none; cursor: pointer; }

        /* Main Workspace Container */
        .playground-container { flex: 1; display: flex; padding: 16px 20px; gap: 16px; width: 100%; max-width: 1600px; margin: 0 auto; }
        
        /* Two Column Layout (Single Mode) */
        .workspace-single { flex: 1; display: flex; gap: 16px; width: 100%; }
        .main-col { flex: 1; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
        .sidebar-col { width: 310px; flex-shrink: 0; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 14px; max-height: calc(100vh - 130px); overflow-y: auto; }

        /* Boxes */
        .box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
        .box-title { font-size: 0.82rem; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; }
        
        /* System Prompt Details */
        details.system-box summary { font-size: 0.82rem; font-weight: 600; color: var(--text-muted); cursor: pointer; user-select: none; outline: none; display: flex; align-items: center; gap: 6px; }
        details.system-box summary:hover { color: var(--cyan); }
        textarea.system-input { margin-top: 8px; width: 100%; min-height: 56px; max-height: 120px; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; color: #cbd5e1; font-size: 0.85rem; font-family: ui-monospace, Menlo, monospace; outline: none; resize: vertical; }

        /* User Prompt */
        textarea.prompt-input { width: 100%; min-height: 100px; max-height: 240px; background: rgba(0,0,0,0.25); border: 1px solid var(--border); border-radius: 8px; padding: 12px; color: #fff; font-size: 0.95rem; line-height: 1.5; outline: none; resize: vertical; }
        textarea.prompt-input:focus { border-color: var(--cyan); }

        /* Attachment chip */
        .preview-chip { display: none; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 8px; font-size: 0.82rem; width: fit-content; }
        .preview-chip img { width: 28px; height: 28px; object-fit: cover; border-radius: 4px; }
        .btn-remove-media { background: transparent; border: none; color: var(--red); font-size: 14px; cursor: pointer; font-weight: bold; }

        /* Action Row */
        .action-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
        .action-left { display: flex; align-items: center; gap: 8px; }
        .action-right { display: flex; align-items: center; gap: 8px; }

        .btn-run { background: linear-gradient(135deg, #0284c7, #2563eb); color: white; border: none; padding: 8px 18px; border-radius: 8px; font-size: 0.88rem; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35); transition: all 0.2s; }
        .btn-run:hover { opacity: 0.95; transform: translateY(-1px); }
        .btn-run.running { background: #ef4444; box-shadow: 0 2px 8px rgba(239, 68, 68, 0.35); }

        .btn-secondary { background: rgba(255, 255, 255, 0.05); color: var(--text-main); border: 1px solid var(--border); padding: 7px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 500; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }
        .btn-secondary:hover { background: var(--bg-card-hover); border-color: var(--cyan); }

        /* Output Box */
        .output-box { flex: 1; min-height: 250px; display: flex; flex-direction: column; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        .output-header { background: #0f1524; border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; font-size: 0.85rem; }
        .status-pill { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; font-size: 0.78rem; color: var(--text-muted); }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); }
        .status-dot.active { background: var(--green); box-shadow: 0 0 6px var(--green); }
        .status-dot.running { background: var(--cyan); animation: pulse 1s infinite; }

        .output-body { flex: 1; padding: 16px; overflow-y: auto; font-size: 0.95rem; line-height: 1.6; color: #e2e8f0; white-space: pre-wrap; word-break: break-word; }
        .output-body pre { background: rgba(0,0,0,0.4); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin: 10px 0; overflow-x: auto; font-family: ui-monospace, Menlo, Monaco, monospace; font-size: 0.88rem; color: #38bdf8; }
        .output-body code { font-family: ui-monospace, Menlo, monospace; font-size: 0.88rem; }

        /* Thinking Process */
        details.thinking-box { background: var(--thinking-bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.82rem; color: var(--text-muted); display: none; }
        details.thinking-box summary { cursor: pointer; font-weight: 600; color: var(--cyan); outline: none; }
        details.thinking-box .thinking-content { margin-top: 8px; white-space: pre-wrap; font-family: ui-monospace, monospace; line-height: 1.4; color: #94a3b8; max-height: 180px; overflow-y: auto; }

        /* Live Telemetry Banner */
        .telemetry-footer { background: #0f1524; border-top: 1px solid var(--border); padding: 8px 16px; display: flex; align-items: center; justify-content: space-between; font-size: 0.78rem; color: var(--text-muted); flex-wrap: wrap; gap: 12px; }
        .tele-stat { display: inline-flex; align-items: center; gap: 4px; }
        .tele-stat strong { color: var(--cyan); font-family: ui-monospace, monospace; font-weight: 600; }

        /* Sidebar Control Styling */
        .param-group { display: flex; flex-direction: column; gap: 6px; }
        .param-label { display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; font-weight: 600; color: var(--text-main); }
        .param-badge { font-family: ui-monospace, monospace; background: rgba(56, 189, 248, 0.15); color: var(--cyan); padding: 1px 6px; border-radius: 4px; font-size: 0.78rem; font-weight: 600; }
        .param-desc { font-size: 0.72rem; color: var(--text-muted); line-height: 1.3; }

        input[type="range"] { -webkit-appearance: none; width: 100%; height: 5px; border-radius: 5px; background: #1e293b; outline: none; margin: 4px 0; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; background: var(--cyan); cursor: pointer; transition: transform 0.1s; }
        input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.15); }

        select.control-select { width: 100%; background: #0f1524; color: var(--text-main); border: 1px solid var(--border); padding: 8px 10px; border-radius: 8px; font-size: 0.88rem; outline: none; cursor: pointer; }
        input.control-input { width: 100%; background: #0f1524; color: var(--text-main); border: 1px solid var(--border); padding: 7px 10px; border-radius: 8px; font-size: 0.85rem; outline: none; }
        input.control-input:focus, select.control-select:focus { border-color: var(--cyan); }

        .ctx-pills { display: flex; gap: 6px; }
        .ctx-pill { flex: 1; background: #0f1524; border: 1px solid var(--border); color: var(--text-muted); padding: 5px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; cursor: pointer; text-align: center; }
        .ctx-pill.active { background: rgba(56, 189, 248, 0.15); border-color: var(--cyan); color: var(--cyan); }

        /* Arena Mode */
        .workspace-arena { display: none; flex-direction: column; gap: 14px; width: 100%; }
        .arena-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; flex: 1; }
        .arena-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; min-height: 380px; }
        .arena-card-header { background: #0f1524; border-bottom: 1px solid var(--border); padding: 10px 14px; display: flex; align-items: center; justify-content: space-between; gap: 8px; }

        /* Modal */
        .modal-overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(4px); display: none; align-items: center; justify-content: center; z-index: 100; }
        .modal { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; width: 90%; max-width: 640px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .modal-header { padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; font-weight: 600; }
        .modal-tabs { display: flex; border-bottom: 1px solid var(--border); background: #0f1524; }
        .modal-tab { flex: 1; padding: 10px; text-align: center; border: none; background: transparent; color: var(--text-muted); font-size: 0.85rem; font-weight: 600; cursor: pointer; }
        .modal-tab.active { color: var(--cyan); border-bottom: 2px solid var(--cyan); background: var(--bg-card); }
        .modal-body { padding: 16px; }
        .modal-code { background: #080c14; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: ui-monospace, Menlo, monospace; font-size: 0.82rem; color: #38bdf8; max-height: 280px; overflow-y: auto; white-space: pre-wrap; }
        .modal-footer { padding: 12px 18px; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; gap: 8px; }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        @media (max-width: 900px) {
            .playground-container { flex-direction: column; padding: 12px; }
            .workspace-single { flex-direction: column; }
            .sidebar-col { width: 100%; max-height: none; }
            .arena-columns { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <!-- Top Navigation Header -->
    <header>
        <div class="brand">
            <span>⚡ LLM Playground</span>
            <span class="badge">Inference Lab</span>
        </div>
        <div class="header-actions">
            <a href="/" class="btn-nav">💬 Chat UI</a>
            <a href="/dashboard" class="btn-nav">📊 Metrics Dashboard</a>
        </div>
    </header>

    <!-- Subnav: Presets & Mode Selector -->
    <div class="subnav">
        <div class="mode-tabs">
            <button class="mode-tab active" id="tab-single" onclick="switchMode('single')">🎛️ Single Model</button>
            <button class="mode-tab arena-tab" id="tab-arena" onclick="switchMode('arena')">⚔️ Model Arena (Side-by-Side)</button>
        </div>
        <div class="preset-group">
            <span>Templates:</span>
            <select class="preset-select" id="preset-select" onchange="applyPreset(this.value)">
                <option value="custom">-- Custom Prompt --</option>
                <option value="code">Code Architect (Python / Rust)</option>
                <option value="reasoning">Step-by-Step Reasoner</option>
                <option value="concise">Concise Executive Summary</option>
                <option value="creative">Creative Storyteller</option>
                <option value="json">Structured JSON Extractor</option>
                <option value="audio">Audio & Speech Analyst</option>
            </select>
        </div>
    </div>

    <!-- Main Workspace Container -->
    <div class="playground-container">
        
        <!-- ================= Single Model Mode ================= -->
        <div class="workspace-single" id="single-view">
            <div class="main-col">
                <!-- System Instructions Box (Collapsible) -->
                <div class="box">
                    <details class="system-box" id="system-details" open>
                        <summary>⚙️ System Instructions (Persona / Rules)</summary>
                        <textarea class="system-input" id="system-prompt" placeholder="Set model behavior (e.g. 'You are an expert systems engineer. Answer concisely with code.')"></textarea>
                    </details>
                </div>

                <!-- User Prompt Box -->
                <div class="box">
                    <div class="box-title">
                        <span>User Prompt</span>
                        <span style="font-size: 0.72rem; color: var(--text-muted);">Press <strong style="color:#fff;">Cmd+Enter</strong> or <strong style="color:#fff;">Ctrl+Enter</strong> to run</span>
                    </div>
                    <textarea class="prompt-input" id="user-prompt" placeholder="Type your prompt here..."></textarea>
                    
                    <!-- Media Attachment Preview Chip -->
                    <div class="preview-chip" id="media-chip">
                        <img id="media-preview-img" style="display:none;">
                        <span id="media-chip-label">Attachment</span>
                        <button class="btn-remove-media" onclick="clearMedia()">✕</button>
                    </div>

                    <input type="file" id="media-file-input" style="display: none;" accept="image/*,audio/*" onchange="handleMediaUpload(event)">

                    <div class="action-row">
                        <div class="action-left">
                            <button class="btn-run" id="btn-run-single" onclick="toggleRunSingle()">
                                <span id="run-single-icon">▶</span> <span id="run-single-text">Run Generation</span>
                            </button>
                            <button class="btn-secondary" onclick="document.getElementById('media-file-input').click()">
                                📎 Attach Media
                            </button>
                        </div>
                        <div class="action-right">
                            <button class="btn-secondary" onclick="clearPrompt()">Clear</button>
                            <button class="btn-secondary" onclick="openExportModal()">📋 Export Code</button>
                        </div>
                    </div>
                </div>

                <!-- Output Box -->
                <div class="output-box">
                    <div class="output-header">
                        <div class="status-pill">
                            <div class="status-dot" id="output-dot"></div>
                            <span id="output-status-text">Ready</span>
                        </div>
                        <button class="btn-secondary" style="padding: 4px 10px; font-size: 0.75rem;" onclick="copyOutputText()">📋 Copy Text</button>
                    </div>
                    
                    <div class="output-body" id="output-container">
                        <details class="thinking-box" id="thinking-box">
                            <summary>🧠 Model Reasoning (<span id="thinking-tok-cnt">0</span> tokens)</summary>
                            <div class="thinking-content" id="thinking-content"></div>
                        </details>
                        <div id="response-content" style="color: var(--text-muted); font-style: italic;">Run generation to inspect output and telemetry...</div>
                    </div>

                    <div class="telemetry-footer" id="telemetry-bar">
                        <span class="tele-stat">⚡ Decode: <strong id="tele-speed">0.0 tok/s</strong></span>
                        <span class="tele-stat">⏱️ TTFT: <strong id="tele-ttft">0 ms</strong></span>
                        <span class="tele-stat">📦 Prompt: <strong id="tele-prompt-tok">0</strong> tok</span>
                        <span class="tele-stat">📝 Output: <strong id="tele-gen-tok">0</strong> tok</span>
                        <span class="tele-stat">🛑 Reason: <strong id="tele-reason">--</strong></span>
                    </div>
                </div>
            </div>

            <!-- Sidebar Controls (Single Mode) -->
            <div class="sidebar-col">
                <div class="param-group">
                    <label class="param-label">
                        <span>Model</span>
                    </label>
                    <select class="control-select" id="param-model"></select>
                </div>

                <div class="param-group">
                    <div class="param-label">
                        <span>Temperature</span>
                        <span class="param-badge" id="temp-badge">0.70</span>
                    </div>
                    <input type="range" id="param-temp" min="0.0" max="1.5" step="0.05" value="0.70" oninput="document.getElementById('temp-badge').textContent = parseFloat(this.value).toFixed(2)">
                    <span class="param-desc">0.0 = Deterministic & Factual | 1.0+ = Creative</span>
                </div>

                <div class="param-group">
                    <div class="param-label">
                        <span>Top P (Nucleus)</span>
                        <span class="param-badge" id="topp-badge">0.90</span>
                    </div>
                    <input type="range" id="param-topp" min="0.05" max="1.0" step="0.05" value="0.90" oninput="document.getElementById('topp-badge').textContent = parseFloat(this.value).toFixed(2)">
                    <span class="param-desc">Probability mass filter for next token sample</span>
                </div>

                <div class="param-group">
                    <div class="param-label">
                        <span>Max Output Tokens</span>
                        <span class="param-badge" id="tokens-badge">1024</span>
                    </div>
                    <input type="range" id="param-tokens" min="128" max="4096" step="128" value="1024" oninput="document.getElementById('tokens-badge').textContent = this.value">
                    <span class="param-desc">Upper bound on generated completion tokens</span>
                </div>

                <div class="param-group">
                    <label class="param-label">Context Window (num_ctx)</label>
                    <div class="ctx-pills">
                        <div class="ctx-pill" onclick="setCtx(4096, this)">4k</div>
                        <div class="ctx-pill" onclick="setCtx(8192, this)">8k</div>
                        <div class="ctx-pill active" onclick="setCtx(16384, this)">16k</div>
                        <div class="ctx-pill" onclick="setCtx(32768, this)">32k</div>
                    </div>
                    <input type="hidden" id="param-ctx" value="16384">
                </div>

                <div class="param-group">
                    <label class="param-label">Stop Sequences</label>
                    <input type="text" class="control-input" id="param-stop" placeholder="e.g. User:, ###">
                    <span class="param-desc">Comma-separated tokens that halt generation</span>
                </div>
            </div>
        </div>

        <!-- ================= Model Arena Mode ================= -->
        <div class="workspace-arena" id="arena-view">
            <!-- Shared Prompt Box -->
            <div class="box">
                <div class="box-title">
                    <span>Arena Shared Prompt</span>
                    <span style="font-size: 0.75rem; color: var(--purple);">Concurrent Head-to-Head Model Evaluation</span>
                </div>
                <textarea class="prompt-input" id="arena-prompt" style="min-height: 80px;" placeholder="Enter a prompt to race both models side-by-side..."></textarea>
                <div class="action-row">
                    <button class="btn-run" style="background: linear-gradient(135deg, #7c3aed, #a855f7);" id="btn-run-arena" onclick="toggleRunArena()">
                        <span id="run-arena-icon">⚔️</span> <span id="run-arena-text">Run Head-to-Head Race</span>
                    </button>
                    <button class="btn-secondary" onclick="document.getElementById('arena-prompt').value = ''">Clear</button>
                </div>
            </div>

            <!-- Dual Columns -->
            <div class="arena-columns">
                <!-- Model A Card -->
                <div class="arena-card">
                    <div class="arena-card-header">
                        <div style="display:flex; align-items:center; gap: 8px;">
                            <span style="font-weight:700; color:#38bdf8;">Model A:</span>
                            <select class="control-select" id="arena-model-a" style="padding: 4px 8px; font-size: 0.8rem;"></select>
                        </div>
                        <span id="arena-status-a" style="font-size: 0.75rem; color: var(--text-muted);">Ready</span>
                    </div>
                    <div class="output-body" id="arena-output-a" style="color: var(--text-muted); font-style: italic;">Awaiting race launch...</div>
                    <div class="telemetry-footer" id="arena-tele-a">
                        <span>Speed: <strong id="arena-speed-a" style="color:#38bdf8;">-- tok/s</strong></span>
                        <span>TTFT: <strong id="arena-ttft-a">-- ms</strong></span>
                        <span>Tokens: <strong id="arena-tok-a">--</strong></span>
                    </div>
                </div>

                <!-- Model B Card -->
                <div class="arena-card">
                    <div class="arena-card-header">
                        <div style="display:flex; align-items:center; gap: 8px;">
                            <span style="font-weight:700; color:#c084fc;">Model B:</span>
                            <select class="control-select" id="arena-model-b" style="padding: 4px 8px; font-size: 0.8rem;"></select>
                        </div>
                        <span id="arena-status-b" style="font-size: 0.75rem; color: var(--text-muted);">Ready</span>
                    </div>
                    <div class="output-body" id="arena-output-b" style="color: var(--text-muted); font-style: italic;">Awaiting race launch...</div>
                    <div class="telemetry-footer" id="arena-tele-b">
                        <span>Speed: <strong id="arena-speed-b" style="color:#c084fc;">-- tok/s</strong></span>
                        <span>TTFT: <strong id="arena-ttft-b">-- ms</strong></span>
                        <span>Tokens: <strong id="arena-tok-b">--</strong></span>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <!-- Export Code Modal -->
    <div class="modal-overlay" id="export-modal" onclick="if(event.target===this) closeExportModal()">
        <div class="modal">
            <div class="modal-header">
                <span>📋 Export Playground Code</span>
                <button class="btn-remove-media" onclick="closeExportModal()">✕</button>
            </div>
            <div class="modal-tabs">
                <button class="modal-tab active" id="tab-curl" onclick="switchExportTab('curl')">cURL</button>
                <button class="modal-tab" id="tab-python" onclick="switchExportTab('python')">Python (Requests)</button>
            </div>
            <div class="modal-body">
                <div class="modal-code" id="modal-code-block"></div>
            </div>
            <div class="modal-footer">
                <button class="btn-secondary" onclick="closeExportModal()">Close</button>
                <button class="btn-run" onclick="copyExportCode()" id="btn-copy-code">Copy to Clipboard</button>
            </div>
        </div>
    </div>

    <!-- JavaScript Logic -->
    <script>
        let currentMode = 'single';
        let attachedBase64 = null;
        let attachedType = null;
        let activeAbortController = null;
        let arenaAbortA = null;
        let arenaAbortB = null;
        let exportTab = 'curl';

        // Preset Definitions
        const PRESETS = {
            custom: {
                system: "",
                prompt: "",
                temp: 0.7,
                top_p: 0.9,
                tokens: 1024
            },
            code: {
                system: "You are an elite software architect and systems programmer. Provide robust, idiomatic code with clean comments and concise explanations.",
                prompt: "Write a high-performance thread-safe token bucket rate limiter in Python with zero external dependencies.",
                temp: 0.2,
                top_p: 0.95,
                tokens: 1536
            },
            reasoning: {
                system: "You are a mathematical and logical reasoning specialist. Think step-by-step and write out each logical deduction clearly.",
                prompt: "A train leaves station A traveling at 60 mph. Two hours later, a second train leaves station A on a parallel track at 90 mph. How long until the second train catches up, and at what distance from station A?",
                temp: 0.4,
                top_p: 0.9,
                tokens: 1024
            },
            concise: {
                system: "You are a concise executive assistant. Summarize information into high-impact, actionable bullet points.",
                prompt: "Explain the architectural differences and trade-offs between FlashAttention and PagedAttention in three crisp bullet points.",
                temp: 0.3,
                top_p: 0.9,
                tokens: 512
            },
            creative: {
                system: "You are an imaginative science fiction author. Write vivid, atmospheric prose with rich sensory details.",
                prompt: "Describe an ancient derelict spacecraft orbiting a neutron star, its quantum computers still whispering corrupted telemetry into the void.",
                temp: 1.0,
                top_p: 0.95,
                tokens: 1024
            },
            json: {
                system: "You are an exact JSON extraction engine. Output ONLY valid, parseable JSON matching the requested schema without any markdown formatting or surrounding commentary.",
                prompt: "Extract the entity name, role, and skills from: 'Dr. Elena Rostova is the Principal Quantum Physicist leading superconducting qubit research at DeepSpace Labs.' Schema: { name: str, role: str, skills: [str], company: str }",
                temp: 0.1,
                top_p: 0.8,
                tokens: 512
            },
            audio: {
                system: "You are a multimodal acoustic and speech analysis expert. Analyze spoken words, speaker tone, cadence, and meaning directly.",
                prompt: "Listen to the attached audio recording. Summarize the main message, identify the speaker's emotional tone, and extract any key decisions made.",
                temp: 0.2,
                top_p: 0.9,
                tokens: 1024
            }
        };

        // Initialize Models
        async function loadModels() {
            try {
                const res = await fetch('/api/tags');
                const data = await res.json();
                const models = (data.models || []).filter(m => !m.details || m.details.family !== 'bert');
                
                const selSingle = document.getElementById('param-model');
                const selArenaA = document.getElementById('arena-model-a');
                const selArenaB = document.getElementById('arena-model-b');

                selSingle.innerHTML = '';
                selArenaA.innerHTML = '';
                selArenaB.innerHTML = '';

                models.forEach((m, idx) => {
                    const opt1 = new Option(m.name, m.name);
                    const opt2 = new Option(m.name, m.name);
                    const opt3 = new Option(m.name, m.name);
                    selSingle.add(opt1);
                    selArenaA.add(opt2);
                    selArenaB.add(opt3);
                });

                // Set default models
                if (models.length > 0) {
                    selSingle.selectedIndex = 0;
                    selArenaA.selectedIndex = 0;
                    selArenaB.selectedIndex = models.length > 1 ? 1 : 0;
                }
            } catch (err) {
                console.error("Error loading models", err);
            }
        }
        loadModels();

        // Switch Modes
        function switchMode(mode) {
            currentMode = mode;
            if (mode === 'single') {
                document.getElementById('tab-single').classList.add('active');
                document.getElementById('tab-arena').classList.remove('active');
                document.getElementById('single-view').style.display = 'flex';
                document.getElementById('arena-view').style.display = 'none';
            } else {
                document.getElementById('tab-single').classList.remove('active');
                document.getElementById('tab-arena').classList.add('active');
                document.getElementById('single-view').style.display = 'none';
                document.getElementById('arena-view').style.display = 'flex';
                // Sync prompt to arena
                const singlePrompt = document.getElementById('user-prompt').value;
                if (singlePrompt && !document.getElementById('arena-prompt').value) {
                    document.getElementById('arena-prompt').value = singlePrompt;
                }
            }
        }

        // Apply Presets
        function applyPreset(key) {
            const p = PRESETS[key];
            if (!p) return;
            if (p.system !== undefined) document.getElementById('system-prompt').value = p.system;
            if (p.prompt) document.getElementById('user-prompt').value = p.prompt;
            if (p.temp !== undefined) {
                document.getElementById('param-temp').value = p.temp;
                document.getElementById('temp-badge').textContent = p.temp.toFixed(2);
            }
            if (p.top_p !== undefined) {
                document.getElementById('param-topp').value = p.top_p;
                document.getElementById('topp-badge').textContent = p.top_p.toFixed(2);
            }
            if (p.tokens !== undefined) {
                document.getElementById('param-tokens').value = p.tokens;
                document.getElementById('tokens-badge').textContent = p.tokens;
            }
        }

        // Context Pills
        function setCtx(val, el) {
            document.querySelectorAll('.ctx-pill').forEach(p => p.classList.remove('active'));
            el.classList.add('active');
            document.getElementById('param-ctx').value = val;
        }

        // Media Upload
        function handleMediaUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            const isAudio = file.type.startsWith('audio/') || 
                            file.name.endsWith('.wav') || file.name.endsWith('.mp3') || 
                            file.name.endsWith('.m4a') || file.name.endsWith('.caf');

            const reader = new FileReader();
            if (isAudio) {
                reader.onload = function(evt) {
                    attachedBase64 = evt.target.result.split(',')[1];
                    attachedType = 'audio';
                    document.getElementById('media-preview-img').style.display = 'none';
                    document.getElementById('media-chip-label').textContent = `🎵 ${file.name}`;
                    document.getElementById('media-chip').style.display = 'inline-flex';
                };
                reader.readAsDataURL(file);
            } else {
                reader.onload = function(evt) {
                    const img = new Image();
                    img.onload = function() {
                        const canvas = document.createElement('canvas');
                        let w = img.width, h = img.height;
                        const maxDim = 1024;
                        if (w > maxDim || h > maxDim) {
                            if (w > h) { h = Math.round((h * maxDim) / w); w = maxDim; }
                            else { w = Math.round((w * maxDim) / h); h = maxDim; }
                        }
                        canvas.width = w; canvas.height = h;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, w, h);
                        const resizedUrl = canvas.toDataURL('image/jpeg', 0.85);
                        attachedBase64 = resizedUrl.split(',')[1];
                        attachedType = 'image';
                        const pImg = document.getElementById('media-preview-img');
                        pImg.src = resizedUrl;
                        pImg.style.display = 'block';
                        document.getElementById('media-chip-label').textContent = file.name;
                        document.getElementById('media-chip').style.display = 'inline-flex';
                    };
                    img.src = evt.target.result;
                };
                reader.readAsDataURL(file);
            }
        }

        function clearMedia() {
            attachedBase64 = null;
            attachedType = null;
            document.getElementById('media-file-input').value = '';
            document.getElementById('media-chip').style.display = 'none';
        }

        function clearPrompt() {
            document.getElementById('user-prompt').value = '';
            clearMedia();
        }

        // ================= Run Single Model Generation =================
        function toggleRunSingle() {
            if (activeAbortController) {
                activeAbortController.abort();
                activeAbortController = null;
                resetRunSingleBtn();
                document.getElementById('output-status-text').textContent = 'Stopped';
                document.getElementById('output-dot').className = 'status-dot';
                return;
            }
            runSingle();
        }

        function resetRunSingleBtn() {
            const btn = document.getElementById('btn-run-single');
            btn.classList.remove('running');
            document.getElementById('run-single-icon').textContent = '▶';
            document.getElementById('run-single-text').textContent = 'Run Generation';
        }

        async function runSingle() {
            const prompt = document.getElementById('user-prompt').value.trim();
            if (!prompt && !attachedBase64) return;

            const model = document.getElementById('param-model').value;
            const system = document.getElementById('system-prompt').value.trim();
            const temp = parseFloat(document.getElementById('param-temp').value);
            const top_p = parseFloat(document.getElementById('param-topp').value);
            const num_predict = parseInt(document.getElementById('param-tokens').value);
            const num_ctx = parseInt(document.getElementById('param-ctx').value);
            const stop = document.getElementById('param-stop').value.trim();

            const btn = document.getElementById('btn-run-single');
            btn.classList.add('running');
            document.getElementById('run-single-icon').textContent = '⏹';
            document.getElementById('run-single-text').textContent = 'Stop';

            const statusDot = document.getElementById('output-dot');
            const statusText = document.getElementById('output-status-text');
            statusDot.className = 'status-dot running';
            statusText.textContent = 'Generating...';

            const thinkBox = document.getElementById('thinking-box');
            const thinkContent = document.getElementById('thinking-content');
            const thinkCnt = document.getElementById('thinking-tok-cnt');
            const respBox = document.getElementById('response-content');

            thinkBox.style.display = 'none';
            thinkContent.textContent = '';
            thinkCnt.textContent = '0';
            respBox.style.color = '#fff';
            respBox.style.fontStyle = 'normal';
            respBox.textContent = '';

            // Reset Telemetry
            document.getElementById('tele-speed').textContent = '-- tok/s';
            document.getElementById('tele-ttft').textContent = '-- ms';
            document.getElementById('tele-prompt-tok').textContent = '--';
            document.getElementById('tele-gen-tok').textContent = '--';
            document.getElementById('tele-reason').textContent = '--';

            activeAbortController = new AbortController();
            const t0 = performance.now();
            let firstTokenTime = null;
            let fullThinking = '';
            let fullResponse = '';

            try {
                const response = await fetch('/api/stream_chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    signal: activeAbortController.signal,
                    body: JSON.stringify({
                        model: model,
                        prompt: prompt,
                        system: system,
                        temperature: temp,
                        top_p: top_p,
                        num_predict: num_predict,
                        num_ctx: num_ctx,
                        stop: stop,
                        image: attachedBase64,
                        media_type: attachedType
                    })
                });

                if (!response.ok) {
                    const err = await response.text();
                    respBox.textContent = 'Server Error: ' + err;
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const jsonStr = line.slice(6);
                            if (!jsonStr) continue;
                            try {
                                const payload = JSON.parse(jsonStr);

                                if (payload.type === 'thinking') {
                                    if (!firstTokenTime) {
                                        firstTokenTime = performance.now();
                                        document.getElementById('tele-ttft').textContent = Math.round(firstTokenTime - t0) + ' ms';
                                    }
                                    fullThinking += payload.text;
                                    thinkBox.style.display = 'block';
                                    thinkContent.textContent = fullThinking;
                                    thinkCnt.textContent = fullThinking.split(/\\s+/).length;
                                } else if (payload.type === 'token') {
                                    if (!firstTokenTime) {
                                        firstTokenTime = performance.now();
                                        document.getElementById('tele-ttft').textContent = Math.round(firstTokenTime - t0) + ' ms';
                                    }
                                    fullResponse += payload.text;
                                    respBox.textContent = fullResponse;
                                } else if (payload.type === 'done') {
                                    const meta = payload.meta;
                                    if (meta) {
                                        document.getElementById('tele-speed').textContent = `${meta.decode_tps.toFixed(1)} tok/s`;
                                        document.getElementById('tele-ttft').textContent = `${meta.prefill_ms.toFixed(0)} ms`;
                                        document.getElementById('tele-prompt-tok').textContent = meta.prefill_tokens;
                                        document.getElementById('tele-gen-tok').textContent = meta.decode_tokens;
                                    }
                                    document.getElementById('tele-reason').textContent = payload.done_reason || 'stop';
                                    statusDot.className = 'status-dot active';
                                    statusText.textContent = 'Completed';
                                } else if (payload.type === 'error') {
                                    respBox.textContent = 'Error: ' + payload.text;
                                    statusDot.className = 'status-dot';
                                    statusText.textContent = 'Error';
                                }
                            } catch (e) {}
                        }
                    }
                }

                if (!fullResponse && fullThinking) {
                    respBox.innerHTML = "<span style='color:#f87171;'>⚠️ Model reached max token limit during internal reasoning. Increase 'Max Output Tokens' in sidebar.</span>";
                }

            } catch (err) {
                if (err.name !== 'AbortError') {
                    respBox.textContent = 'Network Error: ' + err.message;
                }
            } finally {
                activeAbortController = null;
                resetRunSingleBtn();
            }
        }

        // ================= Run Arena Battle =================
        function toggleRunArena() {
            if (arenaAbortA || arenaAbortB) {
                if (arenaAbortA) arenaAbortA.abort();
                if (arenaAbortB) arenaAbortB.abort();
                arenaAbortA = null;
                arenaAbortB = null;
                resetRunArenaBtn();
                return;
            }
            runArena();
        }

        function resetRunArenaBtn() {
            const btn = document.getElementById('btn-run-arena');
            btn.classList.remove('running');
            document.getElementById('run-arena-icon').textContent = '⚔️';
            document.getElementById('run-arena-text').textContent = 'Run Head-to-Head Race';
        }

        async function runArena() {
            const prompt = document.getElementById('arena-prompt').value.trim();
            if (!prompt) return;

            const modelA = document.getElementById('arena-model-a').value;
            const modelB = document.getElementById('arena-model-b').value;
            const system = document.getElementById('system-prompt').value.trim();

            const btn = document.getElementById('btn-run-arena');
            btn.classList.add('running');
            document.getElementById('run-arena-icon').textContent = '⏹';
            document.getElementById('run-arena-text').textContent = 'Stop Arena Race';

            document.getElementById('arena-status-a').textContent = 'Generating...';
            document.getElementById('arena-status-b').textContent = 'Generating...';

            const boxA = document.getElementById('arena-output-a');
            const boxB = document.getElementById('arena-output-b');
            boxA.textContent = ''; boxA.style.fontStyle = 'normal'; boxA.style.color = '#fff';
            boxB.textContent = ''; boxB.style.fontStyle = 'normal'; boxB.style.color = '#fff';

            arenaAbortA = new AbortController();
            arenaAbortB = new AbortController();

            const runStream = async (model, targetBox, speedEl, ttftEl, tokEl, statusEl, abortCtrl) => {
                const t0 = performance.now();
                let ttftSet = false;
                let full = '';
                try {
                    const res = await fetch('/api/stream_chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        signal: abortCtrl.signal,
                        body: JSON.stringify({
                            model: model,
                            prompt: prompt,
                            system: system,
                            temperature: 0.7,
                            num_predict: 1024
                        })
                    });
                    const reader = res.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    let buf = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buf += decoder.decode(value, { stream: true });
                        const lines = buf.split('\\n\\n');
                        buf = lines.pop();

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const p = JSON.parse(line.slice(6));
                                    if (p.type === 'token') {
                                        if (!ttftSet) {
                                            ttftSet = true;
                                            ttftEl.textContent = Math.round(performance.now() - t0) + ' ms';
                                        }
                                        full += p.text;
                                        targetBox.textContent = full;
                                    } else if (p.type === 'done' && p.meta) {
                                        speedEl.textContent = `${p.meta.decode_tps.toFixed(1)} tok/s`;
                                        ttftEl.textContent = `${p.meta.prefill_ms.toFixed(0)} ms`;
                                        tokEl.textContent = p.meta.decode_tokens;
                                        statusEl.textContent = 'Finished';
                                    }
                                } catch (e) {}
                            }
                        }
                    }
                } catch (e) {
                    if (e.name !== 'AbortError') targetBox.textContent = 'Error: ' + e.message;
                }
            };

            await Promise.all([
                runStream(modelA, boxA, document.getElementById('arena-speed-a'), document.getElementById('arena-ttft-a'), document.getElementById('arena-tok-a'), document.getElementById('arena-status-a'), arenaAbortA),
                runStream(modelB, boxB, document.getElementById('arena-speed-b'), document.getElementById('arena-ttft-b'), document.getElementById('arena-tok-b'), document.getElementById('arena-status-b'), arenaAbortB)
            ]);

            arenaAbortA = null;
            arenaAbortB = null;
            resetRunArenaBtn();
        }

        // ================= Code Export & Copy =================
        function copyOutputText() {
            const text = document.getElementById('response-content').textContent;
            if (!text) return;
            navigator.clipboard.writeText(text);
            alert("Response text copied to clipboard!");
        }

        function openExportModal() {
            const prompt = document.getElementById('user-prompt').value || "Hello!";
            const model = document.getElementById('param-model').value || "bonsai-2:latest";
            const system = document.getElementById('system-prompt').value;
            const temp = parseFloat(document.getElementById('param-temp').value);
            const top_p = parseFloat(document.getElementById('param-topp').value);
            const num_predict = parseInt(document.getElementById('param-tokens').value);

            renderExportSnippet(model, prompt, system, temp, top_p, num_predict);
            document.getElementById('export-modal').style.display = 'flex';
        }

        function closeExportModal() {
            document.getElementById('export-modal').style.display = 'none';
        }

        function switchExportTab(tab) {
            exportTab = tab;
            document.getElementById('tab-curl').className = 'modal-tab ' + (tab === 'curl' ? 'active' : '');
            document.getElementById('tab-python').className = 'modal-tab ' + (tab === 'python' ? 'active' : '');
            
            const prompt = document.getElementById('user-prompt').value || "Hello!";
            const model = document.getElementById('param-model').value || "bonsai-2:latest";
            const system = document.getElementById('system-prompt').value;
            const temp = parseFloat(document.getElementById('param-temp').value);
            const top_p = parseFloat(document.getElementById('param-topp').value);
            const num_predict = parseInt(document.getElementById('param-tokens').value);

            renderExportSnippet(model, prompt, system, temp, top_p, num_predict);
        }

        function renderExportSnippet(model, prompt, system, temp, top_p, num_predict) {
            const block = document.getElementById('modal-code-block');
            const host = window.location.host;

            if (exportTab === 'curl') {
                const payload = {
                    model: model,
                    prompt: prompt,
                    system: system || undefined,
                    temperature: temp,
                    top_p: top_p,
                    num_predict: num_predict
                };
                block.textContent = `curl -N http://${host}/api/stream_chat \\\\
  -H "Content-Type: application/json" \\\\
  -d '${JSON.stringify(payload, null, 2)}'`;
            } else {
                block.textContent = `import urllib.request
import json

payload = {
    "model": ${JSON.stringify(model)},
    "prompt": ${JSON.stringify(prompt)},
    "system": ${JSON.stringify(system)},
    "temperature": ${temp},
    "top_p": ${top_p},
    "num_predict": ${num_predict}
}

req = urllib.request.Request(
    "http://${host}/api/stream_chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("data: "):
            data = json.loads(line_str[6:])
            if data.get("type") == "token":
                print(data["text"], end="", flush=True)
print()`;
            }
        }

        function copyExportCode() {
            const text = document.getElementById('modal-code-block').textContent;
            navigator.clipboard.writeText(text);
            const btn = document.getElementById('btn-copy-code');
            btn.textContent = "Copied!";
            setTimeout(() => { btn.textContent = "Copy to Clipboard"; }, 1800);
        }

        // Global Shortcut: Cmd/Ctrl + Enter
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                if (currentMode === 'single') toggleRunSingle();
                else toggleRunArena();
            }
        });
    </script>
</body>
</html>
"""


# ==============================================================================
# HTTP Request Handler
# ==============================================================================
class RemoteLLMHandler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def log_message(self, format, *args):
        log_event(f"{self.client_address[0]} - {format % args}")

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(CHAT_HTML_TEMPLATE.encode("utf-8"))

        elif self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML_TEMPLATE.encode("utf-8"))

        elif self.path == "/playground":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PLAYGROUND_HTML_TEMPLATE.encode("utf-8"))

        elif self.path == "/api/metrics":
            snapshot = METRICS.get_snapshot()
            hw = get_hardware_telemetry()
            payload = json.dumps({"metrics": snapshot, "hardware": hw})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        elif self.path == "/api/tags":
            try:
                req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
                with urllib.request.urlopen(req) as resp:
                    data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/stream_chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req_data = json.loads(body.decode("utf-8"))

            client_ip = self.client_address[0]
            model = req_data.get("model", "bonsai-2:latest")
            prompt = req_data.get("prompt", "")
            media_b64 = req_data.get("image")
            media_type = req_data.get("media_type")

            # Auto-detect media type from binary headers if not passed explicitly
            if media_b64 and not media_type:
                try:
                    head = base64.b64decode(media_b64[:80])
                    if (head[:4] == b"RIFF" and head[8:12] == b"WAVE") or \
                       (len(head) > 8 and head[4:8] == b"ftyp") or \
                       head[:4] == b"caff" or head[:3] == b"ID3" or head[:2] == b"\xff\xfb":
                        media_type = "audio"
                    else:
                        media_type = "image"
                except Exception:
                    media_type = "image"

            # Normalize audio (m4a, mp3, caf, non-16kHz WAV) into 16kHz mono WAV for Ollama
            if media_b64 and media_type == "audio":
                media_b64 = ensure_compatible_audio(media_b64)

            has_media = bool(media_b64)
            actual_type = media_type if has_media else "text"

            # Record in metrics tracker
            METRICS.record_request(client_ip, model, prompt, actual_type)

            media_note = f" [{media_type.upper()} ATTACHED]" if has_media else ""
            log_event(f"[Request] Client: {client_ip} | Model: {model} | Prompt: {prompt[:60]}...{media_note}")

            # Options with configurable defaults for playground & chat
            temp = float(req_data.get("temperature", 0.7))
            top_p = float(req_data.get("top_p", 0.9))
            num_predict = int(req_data.get("num_predict", 1024))
            num_ctx = int(req_data.get("num_ctx", 16384))
            stop = req_data.get("stop")
            system_prompt = req_data.get("system", "")

            opts = {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temp,
                "top_p": top_p,
            }
            if stop:
                if isinstance(stop, str):
                    opts["stop"] = [s.strip() for s in stop.split(",") if s.strip()]
                elif isinstance(stop, list):
                    opts["stop"] = stop

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": opts
            }
            if system_prompt:
                payload["system"] = system_prompt
            if has_media:
                payload["images"] = [media_b64]

            # Start SSE response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                ollama_req = urllib.request.Request(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(ollama_req) as resp:
                    for line in resp:
                        if not line:
                            continue
                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue

                        chunk = json.loads(line_str)

                        # Stream thinking chunk
                        if chunk.get("thinking"):
                            sse_event = json.dumps({"type": "thinking", "text": chunk["thinking"]})
                            try:
                                self.wfile.write(f"data: {sse_event}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                return

                        # Stream answer chunk
                        if chunk.get("response"):
                            sse_event = json.dumps({"type": "token", "text": chunk["response"]})
                            try:
                                self.wfile.write(f"data: {sse_event}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                return

                        # Final done packet
                        if chunk.get("done", False):
                            p_cnt = chunk.get("prompt_eval_count", 0)
                            p_dur = chunk.get("prompt_eval_duration", 1) / 1e6
                            e_cnt = chunk.get("eval_count", 0)
                            e_dur = chunk.get("eval_duration", 1) / 1e6
                            d_tps = (e_cnt / (e_dur / 1000.0)) if e_dur > 0 else 0
                            tpot = (e_dur / e_cnt) if e_cnt > 0 else 0

                            # Record completion metrics
                            METRICS.record_completion(client_ip, model, prompt, actual_type,
                                                     p_cnt, p_dur, e_cnt, d_tps, tpot)

                            meta = {
                                "prefill_tokens": p_cnt,
                                "prefill_ms": p_dur,
                                "decode_tokens": e_cnt,
                                "decode_tps": d_tps,
                                "tpot_ms": tpot
                            }
                            done_reason = chunk.get("done_reason", "stop")
                            log_event(f"[Finished] Client: {client_ip} | Speed: {d_tps:.1f} tok/s | Prefill: {p_dur:.0f} ms | Output: {e_cnt} tokens | Reason: {done_reason}")
                            sse_done = json.dumps({"type": "done", "meta": meta, "done_reason": done_reason})
                            try:
                                self.wfile.write(f"data: {sse_done}\n\n".encode("utf-8"))
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                return
                            break

            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                log_event(f"[Error Streaming] HTTP Error {e.code}: {e.reason} - {err_body}")
                err_event = json.dumps({"type": "error", "text": f"Ollama Error ({e.code}): {err_body}"})
                try:
                    self.wfile.write(f"data: {err_event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            except Exception as e:
                log_event(f"[Error Streaming] {e}")
                err_event = json.dumps({"type": "error", "text": str(e)})
                try:
                    self.wfile.write(f"data: {err_event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses noisy client disconnect tracebacks (e.g. mobile Safari socket resets)."""
    def handle_error(self, request, client_address):
        cls, err, _ = sys.exc_info()
        if cls in (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            return
        super().handle_error(request, client_address)


def run_server():
    local_ip = get_local_ip()
    server_address = ("0.0.0.0", PORT)
    httpd = QuietThreadingHTTPServer(server_address, RemoteLLMHandler)
    httpd.daemon_threads = True

    print("=" * 76)
    print("  APPLE M5 INFERENCE SERVER & TELEMETRY DASHBOARD")
    print("=" * 76)
    print(f"Host listening on all network interfaces (0.0.0.0:{PORT})\n")
    print("Remotely accessible URLs:")
    print(f"  • Chat Interface:    http://{local_ip}:{PORT}")
    print(f"  • LLM Playground:    http://{local_ip}:{PORT}/playground")
    print(f"  • Live Dashboard:    http://{local_ip}:{PORT}/dashboard")
    print(f"  • JSON Metrics API:  http://{local_ip}:{PORT}/api/metrics")
    print(f"  • Via Tailscale:     http://<your-tailscale-ip>:{PORT}/dashboard")
    print("-" * 76)
    log_event(f"Server started on 0.0.0.0:{PORT} (Local IP: {local_ip}, Dashboard: /dashboard)")
    print("Press Ctrl+C to stop the server.")
    print("=" * 76)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
