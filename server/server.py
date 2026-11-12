"""FastAPI GameCopilot AI server: Auto-RAG guides, Whisper STT, Vision Q&A, and Autopilot."""
import base64
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_engine import GameRAG
from stt_engine import transcribe_wav_bytes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("COPILOT_DATA_DIR", BASE_DIR / "data"))
GUIDES_DIR = Path(os.environ.get("COPILOT_GUIDES_DIR", BASE_DIR / "guides"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

rag = GameRAG(DATA_DIR, GUIDES_DIR)

AUTOPILOT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["key_tap", "key_hold", "mouse_click", "wait"]},
        "key": {"type": "string"},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "duration": {"type": "number"},
    },
    "required": ["thought", "action", "key", "x", "y", "duration"],
}


def _warmup_models() -> None:
    """Keep vision and embedding models pinned in GPU VRAM after boot."""
    try:
        httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": ["warmup"]},
            timeout=60.0,
        )
        httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": VISION_MODEL, "prompt": "ok", "stream": False, "options": {"num_predict": 1}},
            timeout=90.0,
        )
    except Exception as exc:
        print(f"[WARN] Model warmup skipped: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warmup_models, daemon=True).start()
    yield


app = FastAPI(title="GameCopilot AI Server", lifespan=lifespan)


class LearnRequest(BaseModel):
    game: str
    max_pages: int = 10


class AskRequest(BaseModel):
    game: str = "General"
    question: str = ""
    audio_b64: str | None = None
    image_b64: str | None = None
    language: str = "es"


class AutopilotRequest(BaseModel):
    game: str = "General"
    goal: str = "Move toward the objective, interact, or combat safely."
    image_b64: str


def _gpu_stats() -> dict[str, str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            timeout=5,
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "temp_c": parts[0],
            "power": parts[1],
            "vram_used": parts[2],
            "vram_total": parts[3],
            "gpu_util": parts[4],
        }
    except Exception as exc:
        return {"error": str(exc)}


def _call_ollama_chat(prompt: str, system: str, image_b64: str | None = None, json_schema: dict | None = None) -> str:
    msg: dict[str, Any] = {"role": "user", "content": prompt}
    if image_b64:
        msg["images"] = [image_b64]
    payload: dict[str, Any] = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": system},
            msg,
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 220},
    }
    if json_schema is not None:
        payload["format"] = json_schema
    resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GameCopilot AI</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;background:#0e1117;color:#e6edf3;max-width:860px;margin:24px auto;padding:0 16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:16px}
input,button,select,textarea{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:14px}
button{background:#238636;border-color:#2ea043;cursor:pointer;font-weight:600}
button:hover{background:#2ea043}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
pre{white-space:pre-wrap;background:#0d1117;padding:12px;border-radius:6px;border:1px solid #30363d}
.muted{color:#8b949e;font-size:13px}
</style></head><body>
<h2>🎮 GameCopilot AI — Local RAG &amp; Vision Server</h2>
<div class="card" id="health">Loading GPU status...</div>
<div class="card">
  <h3>1. Auto-Learn Game Guides (Wikis + Web)</h3>
  <div class="row">
    <input id="gameName" placeholder="e.g. Aniimo, Elden Ring..." value="Aniimo" style="flex:1">
    <button onclick="learnGame()">🔍 Search &amp; Index Guide</button>
  </div>
  <p class="muted" id="learnStatus"></p>
  <div id="gamesList" class="muted"></div>
</div>
<div class="card">
  <h3>2. Quick Guide Query</h3>
  <div class="row" style="margin-bottom:8px">
    <select id="gameSelect"></select>
    <input id="qInput" placeholder="Ask about the game..." style="flex:1">
    <button onclick="ask()">Ask</button>
  </div>
  <pre id="answerBox">Waiting for question...</pre>
</div>
<script>
async function refresh(){
  const r=await fetch('/api/health'); const d=await r.json();
  const g=d.gpu||{};
  document.getElementById('health').innerHTML=
    `<b>GPU:</b> ${g.temp_c||'?'}°C | <b>Power:</b> ${g.power||'?'} | <b>VRAM:</b> ${g.vram_used||'?'} / ${g.vram_total||'?'} | <b>Util:</b> ${g.gpu_util||'?'}`;
  const list=d.games||[];
  const sel=document.getElementById('gameSelect');
  sel.innerHTML=list.map(x=>`<option value="${x.name}">${x.name} (${x.chunks} chunks)</option>`).join('')||'<option>Aniimo</option>';
  document.getElementById('gamesList').innerHTML='<b>Indexed games:</b> '+
    (list.map(x=>`${x.name} [${x.sources} sources, ${x.chunks} chunks]`).join(' · ')||'none yet');
}
async function learnGame(){
  const g=document.getElementById('gameName').value.trim(); if(!g)return;
  document.getElementById('learnStatus').innerText='Searching web guides and indexing...';
  const r=await fetch('/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({game:g,max_pages:10})});
  const d=await r.json();
  document.getElementById('learnStatus').innerText=`Done: ${d.pages_scraped} pages scraped, ${d.chunks_indexed} chunks indexed.`;
  refresh();
}
async function ask(){
  const game=document.getElementById('gameSelect').value||'Aniimo';
  const question=document.getElementById('qInput').value.trim(); if(!question)return;
  document.getElementById('answerBox').innerText='Thinking...';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({game,question})});
  const d=await r.json();
  document.getElementById('answerBox').innerText=d.answer||JSON.stringify(d);
}
refresh(); setInterval(refresh, 10000);
</script></body></html>"""


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "gpu": _gpu_stats(), "games": rag.list_games()}


@app.get("/api/games")
def list_games() -> dict[str, Any]:
    return {"games": rag.list_games()}


@app.post("/api/learn")
def learn_game(req: LearnRequest) -> dict[str, Any]:
    if not req.game.strip():
        raise HTTPException(status_code=400, detail="Game name required")
    return rag.auto_learn_game(req.game.strip(), max_pages=max(1, min(req.max_pages, 20)))


@app.post("/api/upload")
async def upload_guide(game: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    slug = rag.ensure_game(game)
    safe_name = Path(file.filename or "guide.txt").name
    dest = GUIDES_DIR / slug / safe_name
    dest.write_bytes(await file.read())
    stats = rag.ingest_local_folder(game)
    return {"game": game, "file": safe_name, **stats}


def detect_switch_game(text: str) -> str | None:
    """Detect voice/text commands requesting to switch or learn a game."""
    patterns = [
        r"(?:cambia(?:r)? (?:al juego|de juego)|aprende(?:te)?(?: la gu[ií]a de)?|carga(?:r)?(?: la gu[ií]a de)?|vamos a jugar a)\s+([a-z0-9áéíóúñ :\-]{2,40})",
        r"(?:switch to|learn guide for|let'?s play)\s+([a-z0-9 :\-]{2,40})",
    ]
    low = text.lower().strip()
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return m.group(1).strip(" .,!?")
    return None


@app.post("/api/ask")
def ask_copilot(req: AskRequest) -> dict[str, Any]:
    question = req.question.strip()
    if req.audio_b64:
        wav_bytes = base64.b64decode(req.audio_b64)
        spoken = transcribe_wav_bytes(wav_bytes, language=req.language)
        if spoken:
            question = f"{spoken} {question}".strip()

    if not question and req.image_b64:
        question = "Analiza lo que ves en pantalla y dime brevemente qué me recomiendas hacer ahora según la guía."
    if not question:
        raise HTTPException(status_code=400, detail="Question, audio, or screenshot is required.")

    switched = detect_switch_game(question)
    active_game = switched.title() if switched else req.game
    if switched:
        stats = rag.auto_learn_game(active_game, max_pages=8)
        return {
            "game": active_game,
            "switched_game": active_game,
            "question": question,
            "answer": f"Listo, he descargado y aprendido {stats['pages_scraped']} guías de {active_game} ({stats['chunks_indexed']} fragmentos). Ya puedes preguntarme o pulsar F9 para ver tu pantalla.",
            "sources": [],
        }

    hits = rag.search(active_game, question, top_k=4)
    if not hits and active_game.lower() != "general":
        rag.auto_learn_game(active_game, max_pages=6)
        hits = rag.search(active_game, question, top_k=4)

    context_block = "\n\n".join(f"[{h['title']}]\n{h['content']}" for h in hits)
    system = (
        f"Eres un copiloto experto del videojuego '{active_game}'. "
        "Responde de forma directa, práctica y breve (máximo 3-4 frases cortas) en el idioma de la pregunta "
        "para que el jugador pueda leerlo u oírlo sin distraerse. "
        "Si hay imagen de pantalla, fíjate primero en la UI, vida, mapa, objetos o enemigos visibles "
        "y combínalo con la guía proporcionada. No inventes datos que contradigan la guía."
    )
    prompt = f"GUÍA DEL JUEGO:\n{context_block or 'Sin guía indexada aún.'}\n\nPREGUNTA DEL JUGADOR:\n{question}"
    answer = _call_ollama_chat(prompt, system, image_b64=req.image_b64)
    return {
        "game": active_game,
        "switched_game": None,
        "question": question,
        "answer": answer,
        "sources": [h["title"] for h in hits],
    }


@app.post("/api/autopilot")
def autopilot_step(req: AutopilotRequest) -> dict[str, Any]:
    hits = rag.search(req.game, req.goal, top_k=2)
    context_block = "\n".join(h["content"][:400] for h in hits)
    system = (
        f"Eres el módulo piloto automático de '{req.game}'. Analiza la captura de pantalla y devuelve "
        "UNA sola acción segura en JSON estricto. Coordenadas x,y normalizadas entre 0.0 y 1.0 "
        "(0.5,0.5 es el centro). Teclas válidas: w,a,s,d,e,f,space,enter,escape,tab,1,2,3,4,shift. "
        "Si no estás seguro o hay menú ambiguo, usa action='wait'."
    )
    prompt = f"Objetivo actual: {req.goal}\nPistas de guía:\n{context_block or 'N/A'}"
    raw = _call_ollama_chat(prompt, system, image_b64=req.image_b64, json_schema=AUTOPILOT_SCHEMA)
    try:
        action_obj = json.loads(raw)
    except Exception:
        action_obj = {"thought": raw[:120], "action": "wait", "key": "", "x": 0.5, "y": 0.5, "duration": 0.5}
    return {"game": req.game, "decision": action_obj}

