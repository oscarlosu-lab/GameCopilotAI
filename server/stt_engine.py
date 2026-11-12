"""CPU-based local speech-to-text using faster-whisper (keeps GPU VRAM 100% free)."""
import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
_model = None


def get_whisper():
    """Lazy-load Whisper model on CPU int8."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe_wav_bytes(wav_bytes: bytes, language: str = "es") -> str:
    """Transcribe WAV audio bytes into plain text."""
    if not wav_bytes:
        return ""
    model = get_whisper()
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        segments, _ = model.transcribe(
            tmp_path,
            language=language or None,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception as exc:
                print(f"[WARN] Cleanup temp wav failed: {exc}", file=sys.stderr)

