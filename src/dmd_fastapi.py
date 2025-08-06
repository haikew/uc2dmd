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
PATTERN_DIR = os.getenv("PATTERN_DIR", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATTERNS = [
    "SIM_grating_phase_1_period_6.png",
    "SIM_grating_phase_2_period_6.png", 
    "SIM_grating_phase_3_period_6.png",
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
            # Try to read the current mode to check if DMD is responsive
            current_mode = self.i2c.read_byte_data(self.dmd.addr, 0x06)
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
            
            # Toggle PROJ_ON pin to reset DMD
            if hasattr(GPIO, "output"):
                GPIO.output(PROJ_ON_PIN, GPIO.LOW)
                time.sleep(1)
                GPIO.output(PROJ_ON_PIN, GPIO.HIGH)
                
                # Wait for HOST_IRQ to go high
                timeout = time.time() + DMD_COMMUNICATION_TIMEOUT
                while not GPIO.input(HOST_IRQ_PIN) and time.time() < timeout:
                    time.sleep(0.001)
                
                if time.time() >= timeout:
                    print("Timeout waiting for HOST_IRQ")
                    return False
            
            # Reinitialize DMD
            self.dmd = DLPC1438(self.i2c, PROJ_ON_PIN, HOST_IRQ_PIN)
            self.dmd.configure_external_print(LED_PWM=50)
            self.dmd.switch_mode(Mode.EXTERNALPRINT)
            self.dmd.expose_pattern(-1)
            
            return self._check_dmd_communication()
        except Exception as e:
            print(f"DMD restart failed: {e}")
            return False

    # ───────── image helpers ─────────

    @staticmethod
    def _load_patterns(paths: List[str]) -> List[np.ndarray]:
        """Load pattern images from file paths."""
        out: list[np.ndarray] = []
        for p in paths:
            try:
                # Check if path is absolute, if not, make it relative to PATTERN_DIR
                if not os.path.isabs(p):
                    p = os.path.join(PATTERN_DIR, p)
                
                print(f"Loading pattern: {p}")
                img = Image.open(p)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to DMD dimensions
                img = img.resize((DMD_W, DMD_H), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and extract grayscale
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    # Convert RGB to grayscale
                    grayscale = np.mean(img_array, axis=2).astype(np.uint8)
                else:
                    grayscale = img_array.astype(np.uint8)
                
                out.append(grayscale)
                print(f"Successfully loaded pattern {len(out)}: {os.path.basename(p)} ({grayscale.shape})")
                
            except FileNotFoundError:
                print(f"Pattern file not found: {p}, using black frame")
                # fallback: solid black frame
                out.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
            except Exception as e:
                print(f"Error loading pattern {p}: {e}, using black frame")
                out.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
                
        if not out:
            print("No patterns loaded, creating default black pattern")
            out.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
            
        return out

    # ───────── pattern display ─────────

    def display_pattern(self, idx: int):
        """Display a pattern by index with error handling."""
        with self._lock:
            if 0 <= idx < len(self.patterns):
                try:
                    # Check pattern dimensions and adjust if necessary
                    pattern = self.patterns[idx]
                    if len(pattern.shape) == 3:  # RGB image
                        # Convert to grayscale if needed
                        if pattern.shape[2] == 3:
                            pattern = np.mean(pattern, axis=2).astype(np.uint8)
                        elif pattern.shape[2] == 4:
                            pattern = pattern[:, :, 0]  # Take first channel
                    
                    # Ensure pattern is the right size
                    if pattern.shape != (DMD_H, DMD_W):
                        pattern = np.resize(pattern, (DMD_H, DMD_W))
                    
                    self.fb[:, :, 2] = pattern  # red channel only
                    self.fb.flush()  # Ensure data is written to framebuffer
                except Exception as e:
                    print(f"Error displaying pattern {idx}: {e}")
                    raise
            else:
                raise IndexError(f"Pattern id {idx} out of range (0-{len(self.patterns)-1})")

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
print("Initializing DMD Controller...")
pattern_files = [os.path.join(PATTERN_DIR, f) for f in DEFAULT_PATTERNS]

# Verify pattern files exist
existing_patterns = []
for f in pattern_files:
    if os.path.exists(f):
        existing_patterns.append(f)
        print(f"Found pattern file: {f}")
    else:
        print(f"Warning: Pattern file not found: {f}")

if not existing_patterns:
    print("Warning: No pattern files found, will use default patterns")
    existing_patterns = pattern_files  # Let the controller handle missing files

try:
    controller = DMDController(existing_patterns)
    print("DMD Controller initialized successfully")
except Exception as e:
    print(f"Error initializing DMD Controller: {e}")
    controller = None

app = FastAPI(title="DMD FastAPI Controller", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    if controller is None:
        print("Warning: DMD Controller not initialized properly")
    else:
        print("DMD FastAPI Controller started successfully")

@app.on_event("shutdown")
def _cleanup():
    """Shutdown event handler."""
    if controller:
        controller.shutdown()
        print("DMD Controller shutdown complete")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if controller is None:
        return {"status": "unhealthy", "message": "DMD Controller not initialized"}
    
    try:
        status = controller.get_status()
        dmd_comm = controller._check_dmd_communication()
        return {
            "status": "healthy",
            "dmd_communication": dmd_comm,
            "controller_status": status,
            "patterns_loaded": len(controller.patterns)
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}

@app.get("/display/{pattern_id}")
def display(pattern_id: int):
    """Display a specific pattern by ID."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
    
    try:
        controller.display_pattern(pattern_id)
        return {"status": "displayed", "pattern_id": pattern_id}
    except IndexError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to display pattern: {str(e)}"}

@app.get("/patterns")
def list_patterns():
    """List all available patterns."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        patterns_info = []
        for i, pattern in enumerate(controller.patterns):
            patterns_info.append({
                "id": i,
                "shape": pattern.shape,
                "dtype": str(pattern.dtype)
            })
        return {
            "total_patterns": len(controller.patterns),
            "patterns": patterns_info,
            "pattern_files": [os.path.basename(f) for f in existing_patterns]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/start")
def start(cycles: int = -1):
    """Start continuous pattern display. cycles=-1 means infinite loop."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        success = controller.start_continuous(max_cycles=cycles)
        if success:
            return {"status": "started", "max_cycles": cycles}
        else:
            return {"status": "already_running", "message": "Controller is already running"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/stop")
def stop():
    """Stop continuous pattern display."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        success = controller.stop_continuous()
        return {"status": "stopped" if success else "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/pause")
def pause():
    """Pause continuous pattern display."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        success = controller.pause_continuous()
        return {"status": "paused" if success else "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/resume")
def resume():
    """Resume continuous pattern display."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        success = controller.resume_continuous()
        return {"status": "resumed" if success else "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/interrupt")
def interrupt():
    """Interrupt and stop current cycle immediately."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        success = controller.interrupt_continuous()
        return {"status": "interrupted" if success else "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/status")
def get_status():
    """Get current status of the controller."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        status = controller.get_status()
        # Add hardware communication check
        status["dmd_communication"] = controller._check_dmd_communication()
        return status
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/wait/{seconds}")
def set_wait(seconds: float):
    """Set wait time between patterns."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        if seconds < 0.001:
            return {"status": "error", "message": "Wait time must be at least 0.001 seconds"}
        controller.set_wait(seconds)
        return {"status": "ok", "wait": seconds}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/test_dmd")
def test_dmd_communication():
    """Test DMD communication and attempt restart if needed."""
    if controller is None:
        return {"status": "error", "message": "DMD Controller not initialized"}
        
    try:
        communication_ok = controller._check_dmd_communication()
        if not communication_ok:
            restart_success = controller._restart_dmd()
            return {
                "communication_ok": False,
                "restart_attempted": True,
                "restart_success": restart_success,
                "restart_attempts": controller._restart_attempts
            }
        else:
            return {
                "communication_ok": True,
                "restart_attempted": False,
                "restart_attempts": controller._restart_attempts
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ────────────────────────────
# Entrypoint
# ────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
