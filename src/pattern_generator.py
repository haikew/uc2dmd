import numpy as np
from pathlib import Path
from PIL import Image
import argparse  # kept in case future extension needed (currently unused)

WIDTH = 1280
HEIGHT = 720
PERIOD = 6            # 3 white + 3 black per period along the horizontal (x) direction
SHIFT_PIXELS = 2      # exact phase shift (pixels) between successive images (must stay 2)
OUTPUT_FILENAMES_COS = ["grating_cos_phase0.png", "grating_cos_phase1.png", "grating_cos_phase2.png"]  # precise cosine set (only output required now)

# Output directory changed to 'new_pattern' inside the src folder
OUTPUT_DIR = Path(__file__).resolve().parent / "new_pattern"


def generate_base_pattern(width: int, height: int, period: int) -> np.ndarray:
    """Generate a binary grating of vertical stripes (intensity varies along x):
    Pattern per period (left->right): 3 white pixels (255) then 3 black pixels (0).
    Returns: uint8 ndarray shape (H, W) containing only 0 and 255.
    """
    # Create one horizontal line containing repeating 3 white + 3 black pattern
    # x % period < period//2 -> first half (0..2) is white when period=6
    row = ((np.arange(width) % period) < (period // 2)).astype(np.uint8) * 255
    pattern = np.repeat(row[np.newaxis, :], height, axis=0)
    return pattern


def shift_pattern(pattern: np.ndarray, shift: int) -> np.ndarray:
    """Shift pattern horizontally (to the right) by 'shift' pixels with wrap-around."""
    return np.roll(pattern, shift=shift, axis=1)


def save_pattern(array: np.ndarray, path: Path):
    img = Image.fromarray(array, mode='L')  # grayscale only
    img.save(path, format='PNG')


def generate_three_phase_sinusoids(width: int, height: int, period: int, gamma: float = 1.0) -> list:
    """Generate 3 phase-shifted sinusoidal gratings (0°, 120°, 240°) as 8-bit grayscale.
    These are closer to ideal for 3-phase demodulation than hard binary stripes (reduced harmonics).
    gamma: apply output gamma correction (>1 darkens mid-tones). Use 1.0 if DMD already linear.
    Returns list of np.uint8 arrays.
    """
    x = np.arange(width)
    phase_base = 2 * np.pi * x / period
    phases = [0.0, 2*np.pi/3, 4*np.pi/3]
    imgs = []
    for ph in phases:
        wave = 0.5 + 0.5 * np.cos(phase_base - ph)  # 0..1
        if gamma != 1.0:
            wave = np.power(wave, 1.0 / gamma)
        arr = (wave * 255.0 + 0.5).astype(np.uint8)
        imgs.append(np.repeat(arr[np.newaxis, :], height, axis=0))
    return imgs


def generate_three_phase_cosine(width: int, height: int, period: int, phase_offset_px: float = 0.0, gamma: float = 1.0) -> list:
    """Generate mathematically standard 3-phase cosine set (0°,120°,240°) with optional spatial phase offset.
    I_k(x) = 0.5 + 0.5 * cos( 2π (x/period - offset) + φ_k ), φ_k in {0, 2π/3, 4π/3}
    phase_offset_px: shift in pixels applied equally to all three phases so you can align cos maximum to e.g. center of first bright lobe.
    gamma: optional output gamma (1.0 -> linear). Returns list of HxW uint8 arrays.
    """
    x = np.arange(width)
    # Convert pixel offset to radians
    offset_rad = 2 * np.pi * (phase_offset_px / period)
    base = 2 * np.pi * x / period - offset_rad
    phase_terms = [0.0, 2*np.pi/3, 4*np.pi/3]
    imgs = []
    for ph in phase_terms:
        wave = 0.5 + 0.5 * np.cos(base + ph)  # 0..1
        if gamma != 1.0:
            wave = np.power(wave, 1.0 / gamma)
        line = (wave * 255.0 + 0.5).astype(np.uint8)
        imgs.append(np.repeat(line[np.newaxis, :], height, axis=0))
    return imgs


## Binary pattern generator retained (unused) for possible future extension
def generate_three_phase_gratings():
    pass  # intentionally disabled per latest requirement


## Sinusoidal (generic) generator removed from active use per requirement
def generate_three_phase_gratings_sinusoidal(gamma: float = 1.0):
    pass


def generate_three_phase_gratings_cosine(gamma: float = 1.0, align_center: bool = True):
    """Generate and save ONLY the required 3 cosine 120° phase-shifted gratings (grayscale, mode 'L').
    This is now the sole active output per latest instruction.
    align_center: keep True to align phase0 peak to pixel 1 for period=6; set False to start at x=0 peak.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_offset_px = 1.0 if align_center else 0.0
    cos_imgs = generate_three_phase_cosine(WIDTH, HEIGHT, PERIOD, phase_offset_px=phase_offset_px, gamma=gamma)
    for arr, name in zip(cos_imgs, OUTPUT_FILENAMES_COS):
        out_path = OUTPUT_DIR / name
        save_pattern(arr, out_path)
        print(f"Saved: {out_path} (cosine 120°, offset_px={phase_offset_px})")


def main():
    # Directly generate only the cosine 3-phase set per the latest requirement
    generate_three_phase_gratings_cosine(gamma=1.0, align_center=True)

    print("Done. Only cosine 120° phase patterns generated to 'new_pattern'.")

if __name__ == "__main__":
    main()
