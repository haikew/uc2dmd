import numpy as np
from PIL import Image

def generate_binary_grating(h, w, c, cycles):
    # Generate horizontal coordinates
    x = np.linspace(0, 2 * np.pi * cycles, w)
    # Create stripes using a sine function and convert to -1 or 1
    grating = np.sign(np.sin(x))
    # Map from -1/1 to 0/1
    binary_line = (grating + 1) / 2
    # Expand the 1D array to 2D (identical rows)
    pattern = np.tile(binary_line, (h, 1))
    # Initialize image array (uint8 type)
    img = np.zeros((h, w, c), dtype=np.uint8)
    # Assign binary pattern to RGB channels (0: black, 1: white)
    img[:, :, :3] = (pattern[..., None] * 255).astype(np.uint8)
    # Set alpha channel to 255 (fully opaque) if present
    if c == 4:
        img[:, :, 3] = 255
    return img

def save_grating_to_bmp(filename, h, w, c, cycles):
    img_array = generate_binary_grating(h, w, c, cycles)
    mode = 'RGBA' if c == 4 else 'RGB'
    image = Image.fromarray(img_array, mode)
    image.save(filename, format='BMP')
    print(f"Image saved as {filename}")

# Parameter settings
h, w, c = 720, 1280, 4
cycles = 100  # Adjustable grating frequency

save_grating_to_bmp("binary_grating.bmp", h, w, c, cycles)