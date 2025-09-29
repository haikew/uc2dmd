#!/usr/bin/env python3
"""
DMD FastAPI Controller - Simplified Version
Author: Based on TI DLPC1438 DMD Controller specifications
Date: 2025

Simplified DMD controller for displaying requested pattern images.
Main features:
- Display patterns from dmd_fastapi_image directory
- Health monitoring and error recovery
- Compatible with TI DLPC1438 DMD controller
"""

import os
import threading
import time
from typing import List, Optional
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
from pydantic import BaseModel
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
    "binary_p6_s2_idx00.png",
    "binary_p6_s2_idx01.png", 
    "binary_p6_s2_idx02.png",
]

# Timing Configuration
DMD_INIT_TIMEOUT = 3.0      # DMD initialization timeout

# Simple in-memory framebuffer simulator when /dev/fb0 is not available
class _ArrayFB:
    def __init__(self, shape):
        self._buf = np.zeros(shape, dtype=np.uint8)
    def __getitem__(self, idx):
        return self._buf.__getitem__(idx)
    def __setitem__(self, key, value):
        return self._buf.__setitem__(key, value)
    def flush(self):
        pass

# ────────────────────────────
# DMD Controller Class
# ────────────────────────────
class SimpleDMDController:
    """Simplified DMD Controller for manual pattern display."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_pattern = 0

        # Dynamic pattern configuration (aligned with dmd3.py style)
        self.pattern_dir = PATTERN_DIR
        self.pattern_files = list(PATTERN_FILES)
        
        # Load patterns
        self.patterns = self._load_patterns()
        print(f"Loaded {len(self.patterns)} patterns")
        
        # Initialize hardware
        self._init_hardware()
    
    def _load_patterns(self) -> List[np.ndarray]:
        """Load pattern images in 8-bit grayscale (L), resized to DMD."""
        patterns = []
        for filename in self.pattern_files:
            filepath = os.path.join(self.pattern_dir, filename)
            try:
                if os.path.exists(filepath):
                    img = Image.open(filepath).convert("L")  # grayscale
                    if img.size != (DMD_W, DMD_H):
                        print(f"Resizing {filename} from {img.size[::-1]} to ({DMD_H}, {DMD_W})")
                        img = img.resize((DMD_W, DMD_H), Image.Resampling.NEAREST)
                    img_array = np.array(img, dtype=np.uint8)  # (H, W)
                    patterns.append(img_array)
                    print(f"✓ Loaded: {filename}")
                else:
                    print(f"✗ File not found: {filepath}")
                    patterns.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                patterns.append(np.zeros((DMD_H, DMD_W), dtype=np.uint8))
        return patterns if patterns else [np.zeros((DMD_H, DMD_W), dtype=np.uint8)]
    
    def _init_hardware(self):
        """Initialize DMD hardware following TI DLPC1438 specs."""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(PROJ_ON_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(HOST_IRQ_PIN, GPIO.IN)
            
            # Open framebuffer device with r+ (safer); fallback to in-memory FB on failure
            try:
                self.fb = np.memmap(FB_DEVICE, dtype='uint8', mode='r+', shape=(DMD_H, DMD_W, 4))
            except Exception as e_fb:
                print(f"⚠ Framebuffer open failed, using in-memory FB: {e_fb}")
                self.fb = _ArrayFB((DMD_H, DMD_W, 4))
            
            self.i2c = smbus.SMBus(I2C_BUS_ID)
            self.dmd = DLPC1438(self.i2c, PROJ_ON_PIN, HOST_IRQ_PIN)
            
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
            # Ensure subsequent display logic remains functional
            self.fb = _ArrayFB((DMD_H, DMD_W, 4))
    
    def display_pattern(self, pattern_id: int):
        """Display a pattern - identical to dmd_test1.py logic."""
        with self._lock:
            if 0 <= pattern_id < len(self.patterns):
                pattern = self.patterns[pattern_id]
                # Ensure 2D grayscale uint8
                if pattern.ndim == 3:
                    pattern = pattern[..., 0]
                pattern = pattern.astype(np.uint8, copy=False)
                
                # Clear and set framebuffer
                self.fb[:, :, 0] = 0  # Blue
                self.fb[:, :, 1] = 0  # Green
                self.fb[:, :, 2] = 0  # Red clear
                self.fb[:, :, 2] = pattern  # Red channel carries pattern
                
                if hasattr(self.fb, 'flush'):
                    self.fb.flush()
                
                self._current_pattern = pattern_id
                print(f"Displayed pattern {pattern_id}")
            else:
                raise IndexError(f"Pattern {pattern_id} not found")

    def reload_patterns(self, pattern_dir: Optional[str] = None, pattern_files: Optional[List[str]] = None):
        """Reload patterns from directory/files dynamically."""
        with self._lock:
            if pattern_dir is not None:
                self.pattern_dir = pattern_dir
            if pattern_files is not None:
                self.pattern_files = list(pattern_files)
            self.patterns = self._load_patterns()
            self._current_pattern = 0
            print(f"✓ Reloaded patterns from {self.pattern_dir} ({len(self.patterns)} loaded)")
    
    def get_status(self):
        """Get current controller status."""
        return {
            "current_pattern": self._current_pattern,
            "total_patterns": len(self.patterns)
        }

    def shutdown(self):
        """Shutdown controller and cleanup resources."""
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
    description="Simplified DMD controller for manual pattern display"
)

# Request model (add reload similar to dmd3 style)
class ReloadRequest(BaseModel):
    pattern_dir: Optional[str] = None
    pattern_files: Optional[List[str]] = None

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
        "patterns_loaded": status["total_patterns"]
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
        "pattern_files": controller.pattern_files,
        "pattern_directory": controller.pattern_dir
    }

@app.post("/reload")
def reload_patterns(req: ReloadRequest):
    """Reload patterns from given directory and file list."""
    if controller is None:
        return {"status": "error", "message": "Controller not initialized"}
    try:
        controller.reload_patterns(req.pattern_dir, req.pattern_files)
        return {
            "status": "reloaded",
            "pattern_directory": controller.pattern_dir,
            "pattern_files": controller.pattern_files,
            "total_patterns": len(controller.patterns),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
