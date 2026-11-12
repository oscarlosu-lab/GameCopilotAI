# GameCopilot AI

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Project-FF5E5B?style=flat-square&logo=ko-fi&logoColor=white)](https://ko-fi.com/oscarlosu)

**Local AI Gaming Companion** powered by Ollama, RAG (per-game auto-scraped wikis/guides), Whisper voice input, real-time screen vision, a click-through HUD overlay, and an optional Autopilot controller.

Read this in [Español](README_ES.md).

---

## Architecture

Run the AI brain on a dedicated secondary PC (or locally) and keep your gaming rig at 100% FPS:

```text
[ Gaming PC (Windows Client) ]            [ AI Server (Linux / Windows + GPU) ]
  ├─ Hotkeys (F8 Talk, F9 Vision)  ──LAN─►  ├─ FastAPI Backend (:8080)
  ├─ Fast Screen Capture (mss)              ├─ Auto-Guide Web Scraper + SQLite RAG
  ├─ Click-through HUD Overlay              ├─ Whisper STT (CPU int8)
  ├─ Local TTS Voice Feedback               └─ Ollama (qwen2.5vl:3b + nomic-embed-text)
  └─ Optional Autopilot (SendInput)
```

## Features

- **Auto-Learning Guides:** Ask `"let's play <game>"` or click **Search & Index Guide** in the Web UI (`http://<server-ip>:8080`) to automatically fetch wikis/walkthroughs and index them locally. You can also drop `.txt`, `.md`, or `.pdf` files into `server/guides/<game>/`.
- **Voice + Vision On Demand (`F8`):** Hold `F8` to speak while capturing your active screen; Whisper transcribes locally and the VLM answers using both your screen and the game guide.
- **Instant Screen Advice (`F9`):** Captures your screen and gives brief tactical advice via HUD overlay + voice without minimizing fullscreen/borderless games.
- **Autopilot Mode (`F10` / `F12` panic):** Experimental DirectX-compatible `SendInput` loop (`F12` or top-left mouse corner `(0,0)` aborts immediately). *Use Autopilot only in single-player/offline games.*

## Quick Start

### 1. AI Server (with Ollama)

```bash
ollama pull qwen2.5vl:3b
ollama pull nomic-embed-text

cd server
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8080
```

### 2. Windows Gaming Client

```powershell
cd client
.\setup_client.ps1
# Edit client/config.json and set "server_url": "http://<YOUR_SERVER_IP>:8080"
.\start_copilot.bat
```

## Tests

```bash
python -m pytest tests/
```

