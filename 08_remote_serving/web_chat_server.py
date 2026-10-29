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

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_ctx": 16384,
                    "num_predict": 1024,
                    "temperature": 0.7,
                }
            }
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
