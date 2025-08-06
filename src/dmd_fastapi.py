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
# DMD Communication Timeout
# ────────────────────────────
DMD_COMMUNICATION_TIMEOUT = 5.0  # seconds
MAX_RESTART_ATTEMPTS = 3

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
        self._paused = False
        self._interrupted = False
        self._thread: threading.Thread | None = None
        self._current_cycle = 0
        self._max_cycles = -1  # -1 means infinite
        self._restart_attempts = 0

        # ─── Hardware init – guarded so mocks work ───
        self._init_hardware()

        # mmap framebuffer if device exists; otherwise allocate a buffer in RAM
        try:
            self.fb = np.memmap(FB_DEVICE, dtype="uint8", mode="r+", shape=(DMD_H, DMD_W, 4))
        except (FileNotFoundError, PermissionError, ValueError):
            self.fb = np.zeros((DMD_H, DMD_W, 4), dtype="uint8")
        self.fb[:, :, :] = 0  # clear

    def _init_hardware(self):
        """Initialize hardware with error handling and restart capability."""
        try:
            if hasattr(GPIO, "setmode"):
                GPIO.setmode(GPIO.BCM)  # type: ignore[attr-defined]
                GPIO.setup(PROJ_ON_PIN, GPIO.OUT, initial=getattr(GPIO, "LOW", 0))  # type: ignore[arg-type]
                GPIO.setup(HOST_IRQ_PIN, GPIO.IN)  # type: ignore[arg-type]

            self.i2c = smbus.SMBus(I2C_BUS_ID)  # type: ignore[attr-defined]
            self.dmd = DLPC1438(self.i2c, PROJ_ON_PIN, HOST_IRQ_PIN)  # type: ignore[arg-type]
            self.dmd.configure_external_print(LED_PWM=50)
            self.dmd.switch_mode(Mode.EXTERNALPRINT)
            self.dmd.expose_pattern(-1)
            self._restart_attempts = 0  # Reset restart attempts on successful init
        except Exception as e:
            print(f"Hardware initialization failed: {e}")
            if self._restart_attempts < MAX_RESTART_ATTEMPTS:
                self._restart_attempts += 1
                print(f"Attempting restart {self._restart_attempts}/{MAX_RESTART_ATTEMPTS}")
                time.sleep(1)
                self._init_hardware()
            else:
                print("Max restart attempts reached. Hardware may not be available.")

    def _check_dmd_communication(self) -> bool:
        """Check if DMD is responding to communication."""
        try:
            # Try a simple operation to check if DMD is responsive
            # This is a placeholder - actual implementation would depend on DLPC1438 API
            return True
        except Exception:
            return False

    def _restart_dmd(self) -> bool:
        """Restart DMD communication if it becomes unresponsive."""
        if self._restart_attempts >= MAX_RESTART_ATTEMPTS:
            return False
        
        self._restart_attempts += 1
        print(f"DMD unresponsive. Attempting restart {self._restart_attempts}/{MAX_RESTART_ATTEMPTS}")
        
        try:
            # Stop current operations
            self.dmd.stop_exposure()
            time.sleep(0.5)
            
            # Reinitialize
            self._init_hardware()
            return self._check_dmd_communication()
        except Exception as e:
            print(f"DMD restart failed: {e}")
            return False

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
        """Main loop for continuous pattern display with cycle counting."""
        self._current_cycle = 0
        self._interrupted = False
        
        while self._running and not self._interrupted:
            # Check if max cycles reached (unless infinite: -1)
            if self._max_cycles > 0 and self._current_cycle >= self._max_cycles:
                break
                
            for i in range(len(self.patterns)):
                if not self._running or self._interrupted:
                    return
                
                # Check for pause
                while self._paused and self._running and not self._interrupted:
                    time.sleep(0.1)
                
                if not self._running or self._interrupted:
                    return
                
                # Try to display pattern with DMD communication check
                try:
                    if not self._check_dmd_communication():
                        if not self._restart_dmd():
                            print("DMD communication failed. Stopping loop.")
                            self._running = False
                            return
                    
                    self.display_pattern(i)
                    time.sleep(self.t_wait)
                except Exception as e:
                    print(f"Error displaying pattern {i}: {e}")
                    if not self._restart_dmd():
                        print("Failed to restart DMD. Stopping loop.")
                        self._running = False
                        return
                        
            self._current_cycle += 1

    def start_continuous(self, max_cycles: int = -1):
        """Start continuous pattern display with optional cycle limit."""
        if self._running:
            return False
        
        self._running = True
        self._paused = False
        self._interrupted = False
        self._max_cycles = max_cycles
        self._current_cycle = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop_continuous(self):
        """Stop continuous pattern display."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return True

    def pause_continuous(self):
        """Pause continuous pattern display."""
        self._paused = True
        return True

    def resume_continuous(self):
        """Resume continuous pattern display."""
        self._paused = False
        return True

    def interrupt_continuous(self):
        """Interrupt and stop current cycle immediately."""
        self._interrupted = True
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        return True

    def get_status(self):
        """Get current status of the controller."""
        return {
            "running": self._running,
            "paused": self._paused,
            "interrupted": self._interrupted,
            "current_cycle": self._current_cycle,
            "max_cycles": self._max_cycles,
            "wait_time": self.t_wait,
            "total_patterns": len(self.patterns),
            "restart_attempts": self._restart_attempts
        }

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
def start(cycles: int = -1):
    """Start continuous pattern display. cycles=-1 means infinite loop."""
    success = controller.start_continuous(max_cycles=cycles)
    if success:
        return {"status": "started", "max_cycles": cycles}
    else:
        return {"status": "already_running"}

@app.post("/stop")
def stop():
    success = controller.stop_continuous()
    return {"status": "stopped" if success else "error"}

@app.post("/pause")
def pause():
    success = controller.pause_continuous()
    return {"status": "paused" if success else "error"}

@app.post("/resume")
def resume():
    success = controller.resume_continuous()
    return {"status": "resumed" if success else "error"}

@app.post("/interrupt")
def interrupt():
    success = controller.interrupt_continuous()
    return {"status": "interrupted" if success else "error"}

@app.get("/status")
def get_status():
    return controller.get_status()

@app.post("/wait/{seconds}")
def set_wait(seconds: float):
    controller.set_wait(seconds)
    return {"status": "ok", "wait": seconds}

# ────────────────────────────
# Entrypoint
# ────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
