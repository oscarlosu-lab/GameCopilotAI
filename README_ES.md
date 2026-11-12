# GameCopilot AI

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Project-FF5E5B?style=flat-square&logo=ko-fi&logoColor=white)](https://ko-fi.com/oscarlosu)

**Asistente de Juegos con IA Local** impulsado por Ollama, RAG (descarga e indexación automática de wikis/guías por juego), entrada de voz con Whisper, visión de pantalla en tiempo real, overlay HUD transparente y modo Piloto Automático opcional.

Read this in [English](README.md).

---

## Arquitectura

Ejecuta el cerebro de IA en un PC secundario dedicado (o en local) para no perder ni 1 FPS en tu PC principal:

```text
[ PC Gaming (Cliente Windows) ]           [ Servidor IA (Linux / Windows + GPU) ]
  ├─ Atajos (F8 Voz, F9 Visión)    ──LAN─►  ├─ Backend FastAPI (:8080)
  ├─ Captura rápida (mss)                   ├─ Scraper de Guías + RAG en SQLite
  ├─ Overlay HUD sin robar foco             ├─ Whisper STT (CPU int8)
  ├─ Respuesta por voz (TTS offline)        └─ Ollama (qwen2.5vl:3b + nomic-embed-text)
  └─ Piloto Automático (SendInput)
```

## Funciones

- **Aprendizaje Automático de Guías:** Di *"vamos a jugar a [Juego]"* o usa la interfaz web (`http://<ip-servidor>:8080`) para buscar e indexar wikis y guías automáticamente. También puedes añadir archivos `.txt`, `.md` o `.pdf` en `server/guides/<juego>/`.
- **Voz + Visión bajo demanda (`F8`):** Mantén pulsado `F8` para hablar mientras captura tu pantalla; Whisper transcribe en local y el modelo multimodal responde combinando tu pantalla y la guía.
- **Consejo Visual Instantáneo (`F9`):** Captura tu pantalla y te guía por overlay y voz sin minimizar ni robar el foco al juego.
- **Piloto Automático (`F10` / `F12` parada de emergencia):** Bucle experimental con `SendInput` compatible con DirectX (`F12` o mover el ratón a `(0,0)` aborta al instante). *Úsalo solo en juegos single-player/offline.*

## Instalación Rápida

### 1. Servidor IA (con Ollama)

```bash
ollama pull qwen2.5vl:3b
ollama pull nomic-embed-text

cd server
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8080
```

### 2. Cliente Windows

```powershell
cd client
.\setup_client.ps1
# Edita client/config.json con "server_url": "http://<IP_DE_TU_SERVIDOR>:8080"
.\start_copilot.bat
```

## Tests

```bash
python -m pytest tests/
```

