"""Click-through transparent Windows overlay that never steals focus from games."""
import ctypes
import queue
import sys
import threading
import tkinter as tk

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000


class GameOverlay:
    """Top-right HUD notification overlay running in its own Tk thread."""

    def __init__(self, width: int = 440, hide_seconds: float = 12.0) -> None:
        self.width = width
        self.hide_ms = int(hide_seconds * 1000)
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()

    def show(self, title: str, message: str) -> None:
        """Schedule an overlay message update from any thread."""
        self._queue.put((title, message))

    def _run_tk(self) -> None:
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.86)
        root.configure(bg="#11161d")

        screen_w = root.winfo_screenwidth()
        x = max(16, screen_w - self.width - 24)
        root.geometry(f"{self.width}x150+{x}+24")

        frame = tk.Frame(frame_parent := root, bg="#11161d", highlightbackground="#2ea043", highlightthickness=2, padx=12, pady=10)
        frame.pack(fill="both", expand=True)

        title_lbl = tk.Label(frame, text="🎮 GameCopilot AI", fg="#58a6ff", bg="#11161d", font=("Segoe UI", 10, "bold"), anchor="w")
        title_lbl.pack(fill="x")

        msg_lbl = tk.Label(
            frame,
            text="",
            fg="#e6edf3",
            bg="#11161d",
            font=("Segoe UI", 10),
            wraplength=self.width - 32,
            justify="left",
            anchor="nw",
        )
        msg_lbl.pack(fill="both", expand=True, pady=(4, 0))

        frame_parent.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)
        root.withdraw()

        hide_after_id = None

        def poll() -> None:
            nonlocal hide_after_id
            try:
                while True:
                    title, message = self._queue.get_nowait()
                    title_lbl.config(text=title)
                    msg_lbl.config(text=message)
                    root.update_idletasks()
                    needed_h = max(80, min(320, msg_lbl.winfo_reqheight() + 52))
                    root.geometry(f"{self.width}x{needed_h}+{x}+24")
                    root.deiconify()
                    if hide_after_id is not None:
                        root.after_cancel(hide_after_id)
                    hide_after_id = root.after(self.hide_ms, root.withdraw)
            except queue.Empty:
                pass
            root.after(100, poll)

        root.after(100, poll)
        root.mainloop()

