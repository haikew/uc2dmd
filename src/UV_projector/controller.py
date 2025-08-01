import warnings
import RPi.GPIO as GPIO
import time
import enum
import math
import numpy as np

class Mode(enum.IntEnum):
    STANDBY       = 0xFF,
    EXTERNALPRINT = 0x06,
    TESTPATTERN   = 0x01
    

class DLPC1438:
    """
    DLPC1438 controller driver.
    """

    addr = 0x1B  # I²C address

    def __init__(self, i2c_bus, PROJ_ON_PIN, HOST_IRQ_pin):
        print("Intialising the DLPC1438...")

        self.PROJ_ON  = PROJ_ON_PIN
        self.HOST_IRQ = HOST_IRQ_pin

        GPIO.setup(self.PROJ_ON, GPIO.OUT)
        GPIO.setup(self.HOST_IRQ, GPIO.IN)

        self.i2c = i2c_bus

        GPIO.output(self.PROJ_ON, GPIO.LOW)
        time.sleep(1)

        GPIO.output(self.PROJ_ON, GPIO.HIGH)
        t0 = time.time()
        while not GPIO.input(self.HOST_IRQ):
            time.sleep(0.001)
        print(f"DLPC1438 ready {time.time()-t0:.2f}s after PROJ_ON ↑")

        while True:
            time.sleep(0.05)
            try:
                self.i2c.read_byte(self.addr)
                print("DLPC1438 I²C online")
                time.sleep(0.3)
                break
            except OSError:
                pass

    # ------------------------------------------------------------------
    # low-level helpers
    def __i2c_read(self, reg, n):
        return self.i2c.read_i2c_block_data(self.addr, reg, n)

    def __i2c_write(self, reg, data):
        return self.i2c.write_i2c_block_data(self.addr, reg, data)

    # ------------------------------------------------------------------
    # *** new: disable Pixel-Expand / actuator drive ***
    def disable_pixel_expand(self):
        """
        Completely disable Pixel-Expand / 4-phase actuator shifting.

        0xAA – Actuator Output Select: bit0=0 → actuator OFF :contentReference[oaicite:0]{index=0}
        0xA6 – Number of Segments: 0x02 (minimum)              :contentReference[oaicite:1]{index=1}
        0xC8 – Pixel-Sub-Frame Order: [1,1,1,1,1]               :contentReference[oaicite:2]{index=2}
        0x72 – Actuator Gain: 0x00                              :contentReference[oaicite:3]{index=3}
        """
        self.__i2c_write(0xAA, [0x00])
        self.__i2c_write(0xA6, [0x02])
        self.__i2c_write(0xC8, [0x01]*5)
        self.__i2c_write(0x72, [0x00])
    # ------------------------------------------------------------------

    def switch_mode(self, new_mode):
        if isinstance(new_mode, Mode):
            print(f"> Switch DLPC1438 to {new_mode.name}")
            self.__i2c_write(0x05, [new_mode])
            time.sleep(0.4)

            if new_mode != Mode.STANDBY:
                assert self.__i2c_read(0x06, 1)[0] == new_mode.value, "Mode switch failed"
            else:
                queried = self.__i2c_read(0x06, 1)[0]
                print(f"Standby status: {queried}")
                assert queried == new_mode.value, "Mode switch failed"

            if new_mode == Mode.EXTERNALPRINT:
                time.sleep(1)
        else:
            raise Exception("Invalid DLPC1438 mode Enum")

    def configure_external_print(self, LED_PWM):
        print(f"\nConfiguring external print, LED PWM={LED_PWM/1023*100:.1f}%")
        assert isinstance(LED_PWM, int) and 0 <= LED_PWM < 1024, "LED_PWM 10-bit int"

        self.__i2c_write(0xA8, [0x00, 0x04])
        pwm_bytes = [0x00, 0x00, 0x00, 0x00] + list(LED_PWM.to_bytes(2, 'little'))
        self.__i2c_write(0x54, pwm_bytes)
        self.__i2c_write(0x14, [0x00])

    def expose_pattern(self, exposed_frames, dark_frames=5):
        assert isinstance(dark_frames, int) and 0 <= dark_frames < 65536, "dark_frames"
        if exposed_frames > 0:
            assert isinstance(exposed_frames, int) and 0 <= exposed_frames < 65536
            print(f"> UV exposure {exposed_frames} frames ({exposed_frames/60:.2f}s)")
            data = [0x00] + list(dark_frames.to_bytes(2, 'little')) + list(exposed_frames.to_bytes(2, 'little'))
            self.__i2c_write(0xC1, data)
        elif exposed_frames == -1:
            print("> UV exposure ∞")
            self.__i2c_write(0xC1, [0x00] + list(dark_frames.to_bytes(2, 'little')) + [0xFF, 0xFF])
        else:
            raise Exception("Invalid exposure time")
        time.sleep(0.02)

    def stop_exposure(self):
        print("Stopping UV exposure")
        self.__i2c_write(0xC1, [0x01, 0x00, 0x00, 0x00, 0x00])
        time.sleep(0.1)
