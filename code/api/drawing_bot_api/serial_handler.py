import socket
import time
import os
import psutil
import subprocess

class Serial_handler:

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.settimeout(10)
        #self.server_socket.setblocking(False)
        self.server_socket.bind(('localhost', 65432))  # Bind to localhost on port 65432
        self.server_socket.listen()
        self.conn = None
        self.addr = None

    def connect_to_serial_script(self):
        self.conn, self.addr = self.server_socket.accept()
        print('Connected to serial script.')

    def check_socket_still_connected(self, conn):
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
                if 'Python' in proc.info['name'] and proc_name in proc.info['cmdline']:
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
        self.check_serial_script_running(kill=True)

    def send_message_to_script(self, message):
        is_sent = False

        while not is_sent:
            try:
                if self.check_socket_still_connected(self.conn):
                    #print("Sending message")
                    self.conn.sendall(str(message).encode('utf-8'))
                    is_sent = True
                else:
                    print('No connection')
                    print("Connecting to serial script...")
                    self.connect_to_serial_script()
            except:
                print("Connecting to serial script...")
                self.connect_to_serial_script()

    def __call__(self, message):
        if self.check_serial_script_running():
            self.send_message_to_script(message)
            return 0
        else:
            self.start_serial_script()
            time.sleep(3)
            print('Starting serial communication script')
            return 1

if __name__ == '__main__':
    serial_handler = Serial_handler()
    #while(serial_handler.check_serial_script_running()):
        #serial_handler.kill_serial_script()
    #serial_handler.start_serial_script()
    while(1):
        #serial_handler.kill_serial_script()
        serial_handler('yes yes')