"""Windows GameCopilot AI client: Push-to-talk, screen capture, HUD overlay, TTS, and Autopilot."""
import base64
import io
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path

import httpx
import keyboard
import mss
import numpy as np
import pyttsx3
import sounddevice as sd
from PIL import Image

from overlay import GameOverlay
from win_input import cursor_in_failsafe_corner, execute_action

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"


def load_config() -> dict:
    """Load client config from config.json or environment with safe defaults."""
    defaults = {
        "server_url": os.environ.get("GAME_COPILOT_SERVER", "http://127.0.0.1:8080"),
        "active_game": "Aniimo",
        "autopilot_goal": "Explora, interactúa o avanza en el objetivo actual de forma segura.",
        "tts_enabled": True,
        "tts_rate": 185,
        "language": "es",
        "hotkeys": {
            "talk": "f8",
            "vision": "f9",
            "autopilot": "f10",
            "stop": "f12",
        },
    }
    if CONFIG_PATH.exists():
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        defaults.update(loaded)
    return defaults


class VoiceSpeaker:
    """Thread-safe offline Windows TTS speaker."""

    def __init__(self, enabled: bool = True, rate: int = 185) -> None:
        self.enabled = enabled
        self.rate = rate
        self._lock = threading.Lock()
        self._engine = None

    def speak(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str) -> None:
        with self._lock:
            try:
                if self._engine is None:
                    self._engine = pyttsx3.init()
                    self._engine.setProperty("rate", self.rate)
                    for voice in self._engine.getProperty("voices"):
                        if "es" in str(voice.languages).lower() or "spanish" in voice.name.lower() or "helena" in voice.name.lower():
                            self._engine.setProperty("voice", voice.id)
                            break
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"[WARN] TTS error: {exc}", file=sys.stderr)

    def stop(self) -> None:
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:
            pass


def capture_screen_b64(max_width: int = 1024) -> str:
    """Capture primary monitor and encode resized JPEG as base64."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    if img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def samples_to_wav_b64(chunks: list[np.ndarray], samplerate: int = 16000) -> str:
    """Convert recorded int16 numpy chunks into base64 WAV."""
    if not chunks:
        return ""
    audio = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CopilotClient:
    """Main hotkey controller connecting Windows game screen/mic to the AI server."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.server_url = cfg["server_url"].rstrip("/")
        self.active_game = cfg["active_game"]
        self.overlay = GameOverlay()
        self.speaker = VoiceSpeaker(cfg.get("tts_enabled", True), cfg.get("tts_rate", 185))
        self.client = httpx.Client(timeout=60.0)
        self.autopilot_active = False
        self._busy = threading.Lock()
        self._recording = False
        self._audio_chunks: list[np.ndarray] = []

    def notify(self, title: str, text: str, speak: bool = True) -> None:
        print(f"[{title}] {text}")
        self.overlay.show(title, text)
        if speak:
            self.speaker.speak(text)

    def ask_server(self, image_b64: str | None = None, audio_b64: str | None = None, question: str = "") -> None:
        if not self._busy.acquire(blocking=False):
            self.overlay.show("⏳ Ocupado", "Procesando consulta anterior...")
            return
        try:
            payload = {
                "game": self.active_game,
                "question": question,
                "image_b64": image_b64,
                "audio_b64": audio_b64,
                "language": self.cfg.get("language", "es"),
            }
            resp = self.client.post(f"{self.server_url}/api/ask", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("switched_game"):
                self.active_game = data["switched_game"]
                self.cfg["active_game"] = self.active_game
                CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            q_txt = data.get("question", "")
            ans = data.get("answer", "Sin respuesta.")
            self.notify(f"🎮 {self.active_game}", ans if not q_txt else f"Tú: {q_txt}\n\n{ans}")
        except Exception as exc:
            self.notify("❌ Error de conexión", str(exc), speak=False)
        finally:
            self._busy.release()

    def on_talk_press(self, _event=None) -> None:
        if self._recording:
            return
        self._recording = True
        self._audio_chunks = []
        screen_b64 = capture_screen_b64()
        self.overlay.show(f"🎙️ Escuchando ({self.active_game})...", "Habla ahora y suelta F8 para enviar.")

        def record_loop() -> None:
            def callback(indata, _frames, _time, _status) -> None:
                if self._recording:
                    self._audio_chunks.append(indata.copy())

            with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=callback):
                while self._recording:
                    time.sleep(0.05)

            audio_b64 = samples_to_wav_b64(self._audio_chunks, 16000)
            if audio_b64:
                self.overlay.show("🧠 Pensando...", "Transcribiendo voz y consultando guía...")
                self.ask_server(image_b64=screen_b64, audio_b64=audio_b64)

        threading.Thread(target=record_loop, daemon=True).start()

    def on_talk_release(self, _event=None) -> None:
        self._recording = False

    def on_vision_hotkey(self) -> None:
        self.overlay.show(f"👁️ Analizando ({self.active_game})...", "Mirando pantalla y consultando guía...")
        screen_b64 = capture_screen_b64()
        threading.Thread(target=self.ask_server, kwargs={"image_b64": screen_b64}, daemon=True).start()

    def toggle_autopilot(self) -> None:
        if self.autopilot_active:
            self.autopilot_active = False
            self.notify("🛑 Piloto Automático OFF", "Has recuperado el control manual.")
            return
        self.autopilot_active = True
        self.notify("🤖 Piloto Automático ON", "Control activo. Pulsa F12 o mueve el ratón a (0,0) para abortar.")
        threading.Thread(target=self._autopilot_loop, daemon=True).start()

    def emergency_stop(self) -> None:
        self.autopilot_active = False
        self._recording = False
        self.speaker.stop()
        self.notify("🚨 PARADA DE EMERGENCIA", "Piloto automático detenido.", speak=False)

    def _autopilot_loop(self) -> None:
        while self.autopilot_active:
            if cursor_in_failsafe_corner():
                self.emergency_stop()
                break
            try:
                screen_b64 = capture_screen_b64(max_width=896)
                resp = self.client.post(
                    f"{self.server_url}/api/autopilot",
                    json={
                        "game": self.active_game,
                        "goal": self.cfg.get("autopilot_goal", "Avanza de forma segura."),
                        "image_b64": screen_b64,
                    },
                )
                resp.raise_for_status()
                decision = resp.json().get("decision", {})
                self.overlay.show(f"🤖 Autopilot ({decision.get('action')})", decision.get("thought", ""))
                if not self.autopilot_active or not execute_action(decision):
                    self.emergency_stop()
                    break
            except Exception as exc:
                print(f"[WARN] Autopilot error: {exc}", file=sys.stderr)
            time.sleep(1.2)

    def run(self) -> None:
        hk = self.cfg["hotkeys"]
        keyboard.on_press_key(hk["talk"], self.on_talk_press)
        keyboard.on_release_key(hk["talk"], self.on_talk_release)
        keyboard.add_hotkey(hk["vision"], self.on_vision_hotkey)
        keyboard.add_hotkey(hk["autopilot"], self.toggle_autopilot)
        keyboard.add_hotkey(hk["stop"], self.emergency_stop)

        self.notify(
            f"🎮 GameCopilot AI Listo ({self.active_game})",
            f"{hk['talk'].upper()}: Hablar | {hk['vision'].upper()}: Ver pantalla | "
            f"{hk['autopilot'].upper()}: Piloto Auto | {hk['stop'].upper()}: Parar",
            speak=False,
        )
        print("GameCopilot AI activo. Pulsa Ctrl+C en consola o F12 para parar acciones.")
        keyboard.wait()


if __name__ == "__main__":
    client = CopilotClient(load_config())
    client.run()

