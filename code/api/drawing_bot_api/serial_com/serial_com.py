import serial
from config import *

serial_port = serial.Serial('/dev/cu.usbserial-0001', BAUD)
