#!/usr/bin/env python3

import serial
from config import *
import time
import setproctitle
import socket

setproctitle.setproctitle('drawing_bot_serial_com')

class Serial_communicator():
    def __init__(self):
        self.serial = self.connect_to_serial_port()
        while not self.is_ready():
            time.sleep(0.1)

    def check_connection(self):
        try:
            self.serial.write('_\n'.encode('utf-8'))
            return True
        except:
            print('Serial connection lost.')
            return False

    def handle_serial_commands(self, message):
        if not self.serial.is_open:
            self.reconnect()
        self.serial.write(message)
        print(f'Wrote {message} to serial_port')

    def is_ready(self):
        if not self.serial.is_open:
            self.serial.open()

        buffer = []
        while self.serial.in_waiting:
            buffer.append(self.serial.read(1).decode('utf-8'))
        joined_list = ''.join(buffer)

        return 'RDY' in joined_list

    def restart(self):
        message = f'R'
        self.serial.write(message.encode('utf-8'))
        self.serial.close()

    def connect_to_serial_port(self):
        serial_port = None
        while True:
            try:
                print(f'Connecting to serial_port /dev/cu.usbserial-0001')
                serial_port = serial.Serial('/dev/cu.usbserial-0001', BAUD, write_timeout=WRITE_TIMEOUT)
                print(f'Serial port connected.')
                break
            except:
                print('Cannot connect to serial port')
                time.sleep(0.5)
        return serial_port
    
    def reconnect(self):
        print('Reconnecting serial port')
        self.serial.close()
        self.serial = self.connect_to_serial_port()
        while not self.is_ready():
            time.sleep(0.1)

def main():
    serial_com = Serial_communicator()

    while True:
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        #client_socket.setblocking(False)
        print(serial_com.check_connection())
        if not serial_com.check_connection():
            exit()

        try:
            client_socket.connect(('localhost', 65432))  # Connect to the producer
            print('Socket connection established')

            try:
                while True:

                    data = client_socket.recv(1024)  # Receive data
                    print(f'Received Data: {data}')
                    if not data:
                        print('Closing socket connection.')
                        client_socket.close()
                        break  # Exit the loop if no data is received

                    # Send message; reconnects if serial connections fails until it works
                    serial_com.handle_serial_commands(data)
            except Exception as e:
                print(f'The following exception was raised: {e}')
                raise
            finally:
                client_socket.close()  # Close the connection when done
        except:
            time.sleep(0.5)

if __name__ == "__main__":
    main()
