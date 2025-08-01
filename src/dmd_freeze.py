# run_uv_print.py  – user script
import RPi.GPIO as GPIO
import time
import smbus
from PIL import Image
from UV_projector.controller import DLPC1438, Mode
import numpy as np

GPIO.setmode(GPIO.BCM)

HOST_IRQ = 6
PROJ_ON  = 5

i2c = smbus.SMBus(8)

DMD = DLPC1438(i2c, PROJ_ON, HOST_IRQ)

# Disable Pixel-Expand / actuator drive
DMD.disable_pixel_expand()

DMD.configure_external_print(LED_PWM=50)
DMD.switch_mode(Mode.EXTERNALPRINT)
DMD.expose_pattern(-1)

# framebuffer handling
h, w, c = 720, 1280, 4
fb = np.memmap('/dev/fb0', dtype='uint8', mode='w+', shape=(h, w, c))
fb[:, :, 0:3] = 0

log_pixeldata = np.array(Image.open('openMLA_logo_1280x720.png'))
grating_array0 = np.array(Image.open('SLM_0.25_1.50_33_wl488_ang0_pha0.png'))
grating_array1 = np.array(Image.open('SLM_0.25_1.50_33_wl488_ang1_pha0.png'))
grating_array2 = np.array(Image.open('SLM_0.25_1.50_33_wl488_ang2_pha0.png'))

for _ in range(10):
    fb[:, :, 2] = log_pixeldata;  time.sleep(10)
    fb[:, :, 2] = grating_array0; time.sleep(5)
    fb[:, :, 2] = grating_array1; time.sleep(5)
    fb[:, :, 2] = grating_array2; time.sleep(5)

DMD.stop_exposure()
time.sleep(0.3)
DMD.switch_mode(Mode.STANDBY)
# leave system in standby