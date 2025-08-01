import RPi.GPIO as GPIO
import time
import smbus # I2C
from PIL import Image
from UV_projector.controller import DLPC1438, Mode
import numpy as np

GPIO.setmode(GPIO.BCM)

HOST_IRQ = 6 # GPIO pin for HOST_IRQ
PROJ_ON = 5 # GPIO pin for PROJ_ON

i2c = smbus.SMBus(8) #confirgure i2c

# Configure it (NOTE: Mode.STANDBY can clear these settings, so call it again after standby)
DMD = DLPC1438(i2c, PROJ_ON, HOST_IRQ)
DMD.configure_external_print(LED_PWM=50) # not important, set it as default

DMD.switch_mode(Mode.EXTERNALPRINT)
DMD.expose_pattern(-1) 

# write data to framebuffer
h, w, c = 720, 1280, 4 # buffer size
fb = np.memmap('/dev/fb0', dtype='uint8', mode='w+', shape=(h,w,c))

# zero the framebuffer 
fb[:,:, 0] = 0 
fb[:,:, 1] = 0 
fb[:,:, 2] = 0 
#only red channel matters, which is fb[:,:, 2]
# convert to numpy array
white_frame = Image.open('white_gray.png')
white_pixeldata = np.array(white_frame)
black_frame = Image.open('black_gray.png')
black_pixeldata = np.array(black_frame)
log_frame = Image.open('openMLA_logo_1280x720.png')
log_pixeldata = np.array(log_frame)
grating_frame = Image.open('binary_grating_gray.png')
grating_pixeldata = np.array(grating_frame)


# while loop
for i in range(10):
    fb[:,:,2] = white_pixeldata
    time.sleep(5)
    fb[:,:,2] = black_pixeldata
    time.sleep(5)
    fb[:,:,2] = log_pixeldata
    time.sleep(10)
#     # fb[:,:,2] = grating_pixeldata
#     # time.sleep(3)

DMD.stop_exposure()  

time.sleep(0.3)
DMD.switch_mode(Mode.STANDBY)  # leave system in standby

