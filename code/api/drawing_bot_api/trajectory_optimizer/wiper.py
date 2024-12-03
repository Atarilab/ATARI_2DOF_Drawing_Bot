import serial
import platform
import time

PORT = '/dev/cu.usbmodem144101'
BAUD = 9600
WRITE_TIMEOUT = 5

class Wiper:
    def __init__(self):
        self.port = PORT
        self.baud = BAUD
        self.write_timeout = WRITE_TIMEOUT

    def _connect_to_serial_port(self):
        serial_port = None

        while True:
            try:
                print(f'Connecting to serial_port...')

                if platform.system() == 'Darwin':
                    serial_port = serial.Serial(PORT, self.baud, write_timeout=self.write_timeout)

                elif platform.system() == 'Linux':
                    serial_port = serial.Serial('/dev/ttyUSB0', self.baud, write_timeout=self.write_timeout)

                print(f'Serial port connected.')
                break
            
            except:
                print('Cannot connect to serial port')
                time.sleep(0.5)

        return serial_port
    
    def __call__(self):
        serial_port = self._connect_to_serial_port()
        time.sleep(3)
        serial_port.close()
        
if __name__ == '__main__':
    wiper = Wiper()
    wiper()