import RPi.GPIO as GPIO
import time
import smbus  # I2C
from PIL import Image
from UV_projector.controller import DLPC1438, Mode
import numpy as np

GPIO.setmode(GPIO.BCM)

HOST_IRQ = 6  # GPIO pin for HOST_IRQ
PROJ_ON = 5   # GPIO pin for PROJ_ON

i2c = smbus.SMBus(8)  # configure i2c

# Configure DMD
DMD = DLPC1438(i2c, PROJ_ON, HOST_IRQ)
DMD.configure_external_print(LED_PWM=10)  # default settings

DMD.switch_mode(Mode.EXTERNALPRINT)
DMD.expose_pattern(-1)

# Function to convert an 8-bit grayscale image to rgb565 format
def grayscale_to_rgb565(img):
    """
    img: 2D numpy array of dtype uint8, values 0-255
    Calculation:
        red   = (img * 31) // 255   (5 bits)
        green = (img * 63) // 255   (6 bits)
        blue  = (img * 31) // 255   (5 bits)
    rgb565 = (red << 11) | (green << 5) | blue
    """
    r = (img.astype(np.uint16) * 31) // 255
    g = (img.astype(np.uint16) * 63) // 255
    b = (img.astype(np.uint16) * 31) // 255
    return (r << 11) | (g << 5) | b

# In rgb565 mode, each pixel is 2 bytes, so we use a uint16 memmap
h, w = 720, 1280
fb = np.memmap('/dev/fb0', dtype='uint16', mode='w+', shape=(h, w))

# Clear the framebuffer (black screen)
fb[:] = 0

# Read and convert images (ensure they are loaded in grayscale mode)
# white_frame = Image.open('src/white_gray.png').convert('L')
# white_array = np.array(white_frame)
# black_frame = Image.open('src/black_gray.png').convert('L')
# black_array = np.array(black_frame)
log_frame = Image.open('openMLA_logo_1280x720.png').convert('L')
log_array = np.array(log_frame)
# grating_frame = Image.open('src/GratingWidth=5_gray.png').convert('L')
# grating_array = np.array(grating_frame)
# grating_array0 = Image.open('SLM_0.25_1.50_33_wl488_ang0_pha0.png').convert('L')
# grating_array0 = np.array(grating_array0)
# grating_array1 = Image.open('SLM_0.25_1.50_33_wl488_ang1_pha0.png').convert('L')
# grating_array1 = np.array(grating_array1)
# grating_array2 = Image.open('SLM_0.25_1.50_33_wl488_ang2_pha0.png').convert('L')
# grating_array2 = np.array(grating_array2)


# Convert to rgb565 format
# white_fb = grayscale_to_rgb565(white_array)
# black_fb = grayscale_to_rgb565(black_array)
log_fb = grayscale_to_rgb565(log_array)
# grating_fb = grayscale_to_rgb565(grating_array)
# grating_fb0 = grayscale_to_rgb565(grating_array0)
# grating_fb1 = grayscale_to_rgb565(grating_array1)
# grating_fb2 = grayscale_to_rgb565(grating_array2)
# # show withe array
# fb[:] = white_fb

# #show black array
# fb[:] = black_fb

# Loop to display images, each for 3 seconds
for i in range(3):
    # fb[:] = grating_fb0
    # time.sleep(3)
    # fb[:] = grating_fb1
    # time.sleep(3)
    # fb[:] = grating_fb2
    # time.sleep(3)
    # fb[:] = white_fb
    # time.sleep(3)
    # fb[:] = black_fb
    # time.sleep(3)
    fb[:] = log_fb
    time.sleep(30)
    # fb[:] = grating_fb
    # time.sleep(5)

DMD.stop_exposure()
time.sleep(0.3)
DMD.switch_mode(Mode.STANDBY)  # System enters standby
