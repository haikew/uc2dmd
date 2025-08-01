import os
import threading
import time
from types import SimpleNamespace
from typing import List

# ────────────────────────────
# Safe GPIO import / stub
# ────────────────────────────
try:
    import RPi.GPIO as GPIO  # type: ignore
except ImportError:  # Not running on a Pi – create a dummy interface
    class _MockGPIO:  # noqa: D401 – simple mock
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        HIGH = 1
        LOW = 0
        def setmode(self, *_, **__): pass
        def setup(self, *_, **__): pass  # accept "initial" kw-arg
        def output(self, *_, **__): pass
        def cleanup(self, *_, **__): pass
    GPIO = _MockGPIO()  # type: ignore[arg-type]()  # type: ignore[arg-type]

# ────────────────────────────
# Safe smbus import / stub
# ────────────────────────────
try:
    import smbus  # type: ignore
except ImportError:
    class _MockSMBus:  # noqa: D401 – simple mock
        def __init__(self, *_): pass
        def write_byte_data(self, *_): pass
        def read_byte_data(self, *_): return 0
    smbus = SimpleNamespace(SMBus=_MockSMBus)  # type: ignore

from fastapi import FastAPI
import uvicorn
import numpy as np
from PIL import Image

# ────────────────────────────
# DMD driver import / stub
# ────────────────────────────
try:
    from UV_projector.controller import DLPC1438, Mode  # type: ignore
except ImportError:  # development machine without HW libs
    class _NullDMD:  # noqa: D401 – dummy controller
        def configure_external_print(self, **_): pass
        def switch_mode(self, *_): pass
        def expose_pattern(self, *_): pass
        def stop_exposure(self): pass
    DLPC1438 = lambda *_, **__: _NullDMD()  # type: ignore
    Mode = SimpleNamespace(EXTERNALPRINT=1, STANDBY=0)

# ────────────────────────────
# Constants
# ────────────────────────────
HOST_IRQ_PIN = 6
PROJ_ON_PIN = 5
I2C_BUS_ID = 8
DMD_H, DMD_W = 720, 1280
FB_DEVICE = "/dev/fb0"
PATTERN_DIR = os.getenv("PATTERN_DIR", "/home/pi/Patterns")
DEFAULT_PATTERNS = [
    "white_gray.png",
    "black_gray.png",
    "openMLA_logo_1280x720.png",
]

# ────────────────────────────
# DMD Controller
# ────────────────────────────
class DMDController:
    """Wrap the hardware access so that a mocked environment still works."""

    def __init__(self, pattern_paths: List[str]):
        self._lock = threading.Lock()
        self.t_wait = 0.1
        self.patterns = self._load_patterns(pattern_paths)
        self._running = False
        self._thread: threading.Thread | None = None

        # ─── Hardware init – guarded so mocks work ───
        if hasattr(GPIO, "setmode"):
            GPIO.setmode(GPIO.BCM)  # type: ignore[attr-defined]
            GPIO.setup(PROJ_ON_PIN, GPIO.OUT, initial=getattr(GPIO, "LOW", 0))  # type: ignore[arg-type]
            GPIO.setup(HOST_IRQ_PIN, GPIO.IN)  # type: ignore[arg-type]

        self.i2c = smbus.SMBus(I2C_BUS_ID)  # type: ignore[attr-defined]
        self.dmd = DLPC1438(self.i2c, PROJ_ON_PIN, HOST_IRQ_PIN)  # type: ignore[arg-type]
        self.dmd.configure_external_print(LED_PWM=50)
        self.dmd.switch_mode(Mode.EXTERNALPRINT)
        self.dmd.expose_pattern(-1)

        # mmap framebuffer if device exists; otherwise allocate a buffer in RAM
        try:
            self.fb = np.memmap(FB_DEVICE, dtype="uint8", mode="r+", shape=(DMD_H, DMD_W, 4))
        except (FileNotFoundError, PermissionError, ValueError):
            self.fb = np.zeros((DMD_H, DMD_W, 4), dtype="uint8")
        self.fb[:, :, :] = 0  # clear

    # ───────── image helpers ─────────

    @staticmethod
    def _load_patterns(paths: List[str]) -> List[np.ndarray]:
        out: list[np.ndarray] = []
        for p in paths:
            try:
                img = Image.open(p).resize((DMD_W, DMD_H))
                out.append(np.array(img))
            except FileNotFoundError:
                # fallback: solid black frame
                out.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
        return out

    # ───────── pattern display ─────────

    def display_pattern(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self.patterns):
                self.fb[:, :, 2] = self.patterns[idx]  # red channel only
            else:
                raise IndexError("Pattern id out of range")

    # ───────── loop helpers ─────────

    def _loop(self):
        while self._running:
            for i in range(len(self.patterns)):
                if not self._running:
                    return
                self.display_pattern(i)
                time.sleep(self.t_wait)

    def start_continuous(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_continuous(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def set_wait(self, seconds: float):
        self.t_wait = max(0.001, seconds)

    def shutdown(self):
        self.stop_continuous()
        self.dmd.stop_exposure()
        time.sleep(0.3)
        self.dmd.switch_mode(Mode.STANDBY)
        if hasattr(GPIO, "cleanup"):
            GPIO.cleanup()

# ────────────────────────────
# FastAPI
# ────────────────────────────
pattern_files = [os.path.join(PATTERN_DIR, f) for f in DEFAULT_PATTERNS]
controller = DMDController(pattern_files)
app = FastAPI(title="DMD FastAPI Controller")

@app.on_event("shutdown")
def _cleanup():
    controller.shutdown()

@app.get("/display/{pattern_id}")
def display(pattern_id: int):
    controller.display_pattern(pattern_id)
    return {"status": "displayed", "pattern_id": pattern_id}

@app.post("/start")
def start():
    controller.start_continuous()
    return {"status": "running"}

@app.post("/stop")
def stop():
    controller.stop_continuous()
    return {"status": "stopped"}

@app.post("/wait/{seconds}")
def set_wait(seconds: float):
    controller.set_wait(seconds)
    return {"status": "ok", "wait": seconds}

# ────────────────────────────
# Entrypoint
# ────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
