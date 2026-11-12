"""DirectX-compatible Win32 SendInput keyboard and mouse automation with (0,0) failsafe."""
import ctypes
import sys
import time
from ctypes import wintypes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

SCAN_CODES: dict[str, int] = {
    "escape": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "tab": 0x0F, "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14,
    "enter": 0x1C, "ctrl": 0x1D, "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21,
    "g": 0x22, "shift": 0x2A, "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F,
    "space": 0x39,
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


def _send_input(item: INPUT) -> None:
    user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT))


def cursor_in_failsafe_corner() -> bool:
    """Return True if mouse is at top-left (0,0) safety abort corner."""
    pt = wintypes.POINT()
    if user32.GetCursorPos(ctypes.byref(pt)):
        return pt.x <= 4 and pt.y <= 4
    return False


def press_key(key_name: str, duration: float = 0.08) -> None:
    """Press and release a key using hardware scan codes."""
    code = SCAN_CODES.get(key_name.lower().strip())
    if code is None:
        return
    down = INPUT(type=INPUT_KEYBOARD)
    down.union.ki = KEYBDINPUT(0, code, KEYEVENTF_SCANCODE, 0, 0)
    up = INPUT(type=INPUT_KEYBOARD)
    up.union.ki = KEYBDINPUT(0, code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0)
    _send_input(down)
    time.sleep(max(0.03, min(duration, 2.0)))
    _send_input(up)


def click_normalized(nx: float, ny: float) -> None:
    """Move cursor to normalized (0..1, 0..1) screen position and left-click."""
    nx = max(0.02, min(0.98, nx))
    ny = max(0.02, min(0.98, ny))
    abs_x = int(nx * 65535)
    abs_y = int(ny * 65535)
    move = INPUT(type=INPUT_MOUSE)
    move.union.mi = MOUSEINPUT(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, 0)
    down = INPUT(type=INPUT_MOUSE)
    down.union.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)
    up = INPUT(type=INPUT_MOUSE)
    up.union.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)
    _send_input(move)
    time.sleep(0.04)
    _send_input(down)
    time.sleep(0.04)
    _send_input(up)


def execute_action(decision: dict) -> bool:
    """Execute one autopilot JSON decision. Returns False if failsafe triggered."""
    if cursor_in_failsafe_corner():
        return False
    action = decision.get("action", "wait")
    key = str(decision.get("key", ""))
    duration = float(decision.get("duration", 0.15) or 0.15)
    if action == "key_tap":
        press_key(key, 0.08)
    elif action == "key_hold":
        press_key(key, duration)
    elif action == "mouse_click":
        click_normalized(float(decision.get("x", 0.5)), float(decision.get("y", 0.5)))
    elif action == "wait":
        time.sleep(min(duration, 1.5))
    return True

