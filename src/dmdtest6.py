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
# convert to numpy array - using grayscale conversion
log_frame = Image.open('openMLA_logo_1280x720.png').convert('L')
log_pixeldata = np.array(log_frame)

grating_array0 = Image.open('dmd_fastapi_image/binary_p6_s1_idx00.png').convert('L')
grating_array0 = np.array(grating_array0)

grating_array1 = Image.open('dmd_fastapi_image/binary_p6_s1_idx01.png').convert('L')
grating_array1 = np.array(grating_array1)

grating_array2 = Image.open('dmd_fastapi_image/binary_p6_s1_idx02.png').convert('L')
grating_array2 = np.array(grating_array2)

grating_array3 = Image.open('dmd_fastapi_image/binary_p6_s1_idx03.png').convert('L')
grating_array3 = np.array(grating_array3)

grating_array4 = Image.open('dmd_fastapi_image/binary_p6_s1_idx04.png').convert('L')
grating_array4 = np.array(grating_array4)

grating_array5 = Image.open('dmd_fastapi_image/binary_p6_s1_idx05.png').convert('L')
grating_array5 = np.array(grating_array5)

# while loop
for i in range(10):
    fb[:,:,2] = log_pixeldata
    time.sleep(10)
    fb[:,:,2] = grating_array0
    time.sleep(5)
    fb[:,:,2] = grating_array1
    time.sleep(5)
    fb[:,:,2] = grating_array2
    time.sleep(5)
    fb[:,:,2] = grating_array3
    time.sleep(5)
    fb[:,:,2] = grating_array4
    time.sleep(5)
    fb[:,:,2] = grating_array5
    time.sleep(5)

DMD.stop_exposure()  

time.sleep(0.3)
DMD.switch_mode(Mode.STANDBY)  # leave system in standby

