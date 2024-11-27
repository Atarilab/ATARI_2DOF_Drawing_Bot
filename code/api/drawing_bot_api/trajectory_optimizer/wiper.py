import serial
import platform
import time

PORT = 'ARDUINO PORT' # STILL NEED TO ADD THIS PROPERLY
BAUD = 9600
WRITE_TIMEOUT = 5

class Wiper:
    def __init__(self):
        self.port = PORT
        self.baud = BAUD
        self.write_timeout = WRITE_TIMEOUT

    def connect_to_serial_port(self):
        serial_port = None

        while True:
            try:
                print(f'Connecting to serial_port...')

                if platform.system() == 'Darwin':
                    print('AHA')
                    serial_port = serial.Serial('/dev/cu.usbserial-0001', self.baud, write_timeout=self.write_timeout)

                elif platform.system() == 'Linux':
                    serial_port = serial.Serial('/dev/ttyUSB0', self.baud, write_timeout=self.write_timeout)

                print(f'Serial port connected.')
                break
            
            except:
                print('Cannot connect to serial port')
                time.sleep(0.5)

        return serial_port
    
    def wipe(self):
        serial_port = self.connect_to_serial_port()
        time.sleep(3)
        serial_port.close()
        
    