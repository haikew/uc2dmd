#!/usr/bin/env python3
"""
DMD FastAPI Controller - Simplified Version
Author: Based on TI DLPC1438 DMD Controller specifications
Date: 2025

Simplified DMD controller for cycling through three pattern images.
Main features:
- Display patterns from dmd_fastapi_image directory
- Configurable cycle count and display time
- Health monitoring and error recovery
- Compatible with TI DLPC1438 DMD controller
"""

import os
import threading
import time
from typing import List
import numpy as np
from PIL import Image

# ────────────────────────────
# Safe imports with mocks for development
# ────────────────────────────
try:
    import RPi.GPIO as GPIO
except ImportError:
    class _MockGPIO:
        BCM, OUT, IN, HIGH, LOW = "BCM", "OUT", "IN", 1, 0
        def setmode(self, *_): pass
        def setup(self, *_, **__): pass
        def output(self, *_): pass
        def cleanup(self): pass
        def input(self, *_): return 1
    GPIO = _MockGPIO()

try:
    import smbus
except ImportError:
    class _MockSMBus:
        def __init__(self, *_): pass
        def write_byte_data(self, *_): pass
        def read_byte_data(self, *_): return 0
    class _MockSMBusModule: SMBus = _MockSMBus
    smbus = _MockSMBusModule()

try:
    from UV_projector.controller import DLPC1438, Mode
except ImportError:
    class _MockDMD:
        def configure_external_print(self, **_): pass
        def switch_mode(self, *_): pass
        def expose_pattern(self, *_): pass
        def stop_exposure(self): pass
    DLPC1438 = lambda *_, **__: _MockDMD()
    class _MockMode: EXTERNALPRINT, STANDBY = 1, 0
    Mode = _MockMode()

from fastapi import FastAPI
import uvicorn

# ────────────────────────────
# Configuration Constants
# ────────────────────────────
# DMD Hardware Configuration (based on TI DLPC1438 specs)
HOST_IRQ_PIN = 6      # GPIO pin for DMD ready signal
PROJ_ON_PIN = 5       # GPIO pin for DMD power control
I2C_BUS_ID = 8        # I2C bus for DMD communication
DMD_H, DMD_W = 720, 1280    # DMD resolution (standard 0.33" DMD)
FB_DEVICE = "/dev/fb0"      # Linux framebuffer device
LED_PWM = 50                # LED PWM setting (0-100)

# Pattern Configuration
PATTERN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dmd_fastapi_image")
PATTERN_FILES = [
    "SIM_grating_phase_1_period_6.png",
    "SIM_grating_phase_2_period_6.png", 
    "SIM_grating_phase_3_period_6.png",
]

# Timing Configuration
DEFAULT_DISPLAY_TIME = 5.0  # Default display time per pattern (seconds)
MIN_DISPLAY_TIME = 0.1      # Minimum display time
DMD_INIT_TIMEOUT = 3.0      # DMD initialization timeout

# ────────────────────────────
# DMD Controller Class
# ────────────────────────────
class SimpleDMDController:
    """Simplified DMD Controller for pattern cycling."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._current_pattern = 0
        self._current_cycle = 0
        self._max_cycles = -1
        self._display_time = DEFAULT_DISPLAY_TIME
        
        # Load patterns
        self.patterns = self._load_patterns()
        print(f"Loaded {len(self.patterns)} patterns")
        
        # Initialize hardware
        self._init_hardware()
    
    def _load_patterns(self) -> List[np.ndarray]:
        """Load pattern images exactly like dmd_test1.py."""
        patterns = []
        
        for filename in PATTERN_FILES:
            filepath = os.path.join(PATTERN_DIR, filename)
            try:
                if os.path.exists(filepath):
                    # Load exactly like dmd_test1.py: Image.open() -> np.array()
                    img = Image.open(filepath)
                    img_array = np.array(img)
                    
                    # Ensure correct dimensions
                    if img_array.shape[:2] != (DMD_H, DMD_W):
                        print(f"Resizing {filename} from {img_array.shape[:2]} to ({DMD_H}, {DMD_W})")
                        img_pil = Image.fromarray(img_array)
                        img_pil = img_pil.resize((DMD_W, DMD_H), Image.Resampling.NEAREST)
                        img_array = np.array(img_pil)
                    
                    patterns.append(img_array)
                    print(f"✓ Loaded: {filename}")
                else:
                    print(f"✗ File not found: {filepath}")
                    # Create black pattern as fallback
                    patterns.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
                    
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                patterns.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
        
        return patterns if patterns else [np.zeros((DMD_H, DMD_W), dtype=np.uint8)]
    
    def _init_hardware(self):
        """Initialize DMD hardware following TI DLPC1438 specs."""
        try:
            # GPIO setup for DMD control
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(PROJ_ON_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(HOST_IRQ_PIN, GPIO.IN)
            
            # Initialize framebuffer
            self.fb = np.memmap(FB_DEVICE, dtype='uint8', mode='w+', shape=(DMD_H, DMD_W, 4))
            
            # Initialize DMD controller
            self.i2c = smbus.SMBus(I2C_BUS_ID)
            self.dmd = DLPC1438(self.i2c, PROJ_ON_PIN, HOST_IRQ_PIN)
            
            # Configure DMD for external print mode (TI DLPC1438 standard procedure)
            self.dmd.configure_external_print(LED_PWM=LED_PWM)
            self.dmd.switch_mode(Mode.EXTERNALPRINT)
            self.dmd.expose_pattern(-1)  # Continuous exposure
            
            # Clear framebuffer initially
            self.fb[:, :, :] = 0
            if hasattr(self.fb, 'flush'):
                self.fb.flush()
                
            print("✓ DMD hardware initialized successfully")
            
        except Exception as e:
            print(f"⚠ Hardware init failed (mock mode): {e}")
    
    def display_pattern(self, pattern_id: int):
        """Display a pattern - identical to dmd_test1.py logic."""
        with self._lock:
            if 0 <= pattern_id < len(self.patterns):
                pattern = self.patterns[pattern_id]
                
                # Clear and set framebuffer exactly like dmd_test1.py
                self.fb[:, :, 0] = 0  # Blue channel
                self.fb[:, :, 1] = 0  # Green channel
                self.fb[:, :, 2] = 0  # Red channel - clear first
                self.fb[:, :, 2] = pattern  # Only red channel matters
                
                if hasattr(self.fb, 'flush'):
                    self.fb.flush()
                
                self._current_pattern = pattern_id
                print(f"Displayed pattern {pattern_id}")
            else:
                raise IndexError(f"Pattern {pattern_id} not found")
    
    def start_cycling(self, cycles: int = -1, display_time: float = DEFAULT_DISPLAY_TIME):
        """Start pattern cycling."""
        with self._lock:
            if self._running:
                return False
            
            self._max_cycles = cycles
            self._display_time = max(MIN_DISPLAY_TIME, display_time)
            self._current_cycle = 0
            self._running = True
            
            self._thread = threading.Thread(target=self._cycle_loop, daemon=True)
            self._thread.start()
            
            print(f"Started cycling: {cycles} cycles, {display_time}s per pattern")
            return True
    
    def stop_cycling(self):
        """Stop pattern cycling."""
        with self._lock:
            if not self._running:
                return False
            
            self._running = False
            
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        print("Stopped cycling")
        return True
    
    def _cycle_loop(self):
        """Main cycling loop - similar to dmd_test1.py loop."""
        try:
            while self._running:
                # Check cycle limit
                if self._max_cycles > 0 and self._current_cycle >= self._max_cycles:
                    break
                
                # Display each pattern in sequence
                for i in range(len(self.patterns)):
                    if not self._running:
                        return
                    
                    try:
                        self.display_pattern(i)
                        time.sleep(self._display_time)
                    except Exception as e:
                        print(f"Error in cycle loop: {e}")
                
                self._current_cycle += 1
                print(f"Completed cycle {self._current_cycle}")
        
        except Exception as e:
            print(f"Cycle loop error: {e}")
        finally:
            self._running = False
    
    def get_status(self):
        """Get current controller status."""
        return {
            "running": self._running,
            "current_pattern": self._current_pattern,
            "current_cycle": self._current_cycle,
            "max_cycles": self._max_cycles,
            "display_time": self._display_time,
            "total_patterns": len(self.patterns)
        }
    
    def shutdown(self):
        """Shutdown controller and cleanup resources."""
        self.stop_cycling()
        
        try:
            if hasattr(self, 'dmd'):
                self.dmd.stop_exposure()
                time.sleep(0.2)
                self.dmd.switch_mode(Mode.STANDBY)
            GPIO.cleanup()
            print("✓ DMD shutdown complete")
        except Exception as e:
            print(f"Shutdown warning: {e}")

# ────────────────────────────
# FastAPI Application
# ────────────────────────────
print("Initializing DMD FastAPI Controller...")
print(f"Pattern directory: {PATTERN_DIR}")

# Initialize controller
try:
    controller = SimpleDMDController()
    print("✓ DMD Controller ready")
except Exception as e:
    print(f"✗ Controller init failed: {e}")
    controller = None

app = FastAPI(
    title="DMD Pattern Controller", 
    version="2.0",
    description="Simplified DMD controller for pattern cycling"
)

# ────────────────────────────
# API Endpoints
# ────────────────────────────
@app.get("/health")
def health():
    """Health check endpoint."""
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    
    status = controller.get_status()
    return {
        "status": "healthy",
        "patterns_loaded": status["total_patterns"],
        "running": status["running"]
    }

@app.get("/display/{pattern_id}")
def display_single(pattern_id: int):
    """Display a single pattern."""
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    
    try:
        controller.display_pattern(pattern_id)
        return {"status": "displayed", "pattern_id": pattern_id}
    except IndexError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Display failed: {e}"}

@app.post("/start")
def start_cycling(cycles: int = -1, display_time: float = DEFAULT_DISPLAY_TIME):
    """Start pattern cycling.
    
    Args:
        cycles: Number of cycles (-1 for infinite)
        display_time: Time per pattern in seconds
    """
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    
    try:
        success = controller.start_cycling(cycles, display_time)
        if success:
            return {
                "status": "started",
                "cycles": cycles,
                "display_time": display_time
            }
        else:
            return {"status": "error", "message": "Already running"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/stop")
def stop_cycling():
    """Stop pattern cycling."""
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    
    try:
        success = controller.stop_cycling()
        return {"status": "stopped" if success else "not_running"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/status")
def get_status():
    """Get controller status."""
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    
    return controller.get_status()

@app.get("/patterns")
def list_patterns():
    """List available patterns."""
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    
    return {
        "total_patterns": len(controller.patterns),
        "pattern_files": PATTERN_FILES,
        "pattern_directory": PATTERN_DIR
    }

# ────────────────────────────
# Application Lifecycle
# ────────────────────────────
@app.on_event("shutdown")
def cleanup():
    """Cleanup on app shutdown."""
    if controller:
        controller.shutdown()

# ────────────────────────────
# Main Entry Point
# ────────────────────────────
if __name__ == "__main__":
    print("Starting DMD FastAPI server on http://0.0.0.0:8000")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if controller:
            controller.shutdown()
