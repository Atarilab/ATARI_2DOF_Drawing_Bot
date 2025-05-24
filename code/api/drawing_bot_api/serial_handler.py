import socket
import time
import os
import psutil
import subprocess
from drawing_bot_api.config import *
import platform
from drawing_bot_api.logger import Log

class Serial_handler:

    def __init__(self, verbose=1):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.settimeout(20)
        #self.server_socket.setblocking(False)
        self.server_socket.bind(('localhost', 65432))  # Bind to localhost on port 65432
        self.server_socket.listen()
        self.conn = None
        self.addr = None
        self.buffer = []
        self.log = Log(verbose)

    def __init_connection(self):
        while True:
            if self.check_serial_script_running():
                break
            else:
                self.start_serial_script()
                time.sleep(3)
                self.log('Starting serial communication script')
        
        self.connect_to_serial_script()

    def __disconnect(self):
        if self.conn != None:
            self.conn.close()

    def connect_to_serial_script(self):
        self.log(f'Connecting to serial script...')
        self.conn, self.addr = self.server_socket.accept()
        self.log('Connected to serial script.')

    def check_socket_connected(self, conn):
        try:
            data = conn.recv(1024)
            #print(f'Data from socket client: {data}')
            if not data:
                print("Client has disconnected.")
                return 0
        except socket.error:
            print("Error with the client connection.")
            return 0
        return 1

    def check_serial_script_running(self, kill=False):

        proc_name = 'drawing_bot_serial_com'

        # Iterate over all running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):

            try:
                # Check if the process is a Python process and contains the script name
                if platform.system() == 'Darwin':
                    if 'Python' in proc.info['name'] and proc_name in proc.info['cmdline']:
                        #print(f"Script {proc_name} is running with PID {proc.info['pid']}")
                        if kill:
                            proc.terminate()
                        return True
                    
                elif platform.system() == 'Linux':
                    if proc_name in proc.info['name']:
                        #print(f"Script {proc_name} is running with PID {proc.info['pid']}")
                        if kill:
                            proc.terminate()
                        return True
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        return False

    def start_serial_script(self):
        command = 'nohup python3 ./drawing_bot_api/serial_com/serial_com.py'
        #print(os.system(f'echo | {command}'))
        print(subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        #return process

    def kill_serial_script(self):
        while self.check_serial_script_running(kill=True):
            time.sleep(0.1)

    def send_buffer(self, promting):
        self.__init_connection()
        __time = self.millis()

        self.conn.sendall(str(self.buffer[0]).encode('utf-8'))
        self.conn.sendall(str(self.buffer[1]).encode('utf-8'))
        time.sleep(0.5)

        if promting:
            answer = input('Do you want to continue with this drawing? (y/n)\n')
            if answer != 'y':
                self.buffer.clear()
                return 1

        for message in self.buffer:
            try:
                self.conn.sendall(str(message).encode('utf-8'))
            except:
                pass
                # RAISE EXCEPTION ABOUT THIS

            __delay = SERIAL_DELAY - ((self.millis() - __time)/1000)
            if __delay < 0:
                __delay = 0
            time.sleep(__delay)
            __time = self.millis()

        self.buffer.clear()

        self.__disconnect()

    def millis(self):
        return time.time()*1000

    def __call__(self, message):
        self.buffer.append(message)

if __name__ == '__main__':
    serial_handler = Serial_handler()
    #while(serial_handler.check_serial_script_running()):
        #serial_handler.kill_serial_script()
    serial_handler.start_serial_script()
    #while(1):
        #serial_handler.kill_serial_script()
        #serial_handler('test')