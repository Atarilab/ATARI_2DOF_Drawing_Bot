import multiprocessing
import time
import os
import psutil
import subprocess

class Serial_handler:

    def __init__(self):
        self.queue = multiprocessing.Queue()

    def check_serial_script_running(self, kill=False):

        proc_name = 'drawing_bot_serial_com'

        # Iterate over all running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):

            try:
                # Check if the process is a Python process and contains the script name
                if 'Python' in proc.info['name'] and proc_name in proc.info['cmdline']:
                    print(f"Script {proc_name} is running with PID {proc.info['pid']}")
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

    def send_message_to_script_helper(self, queue, message):
        queue.put(message)

    def send_message_to_script(self, message):
        producer_process = multiprocessing.Process(target=self.send_message_to_script_helper, args=(self.queue, message,))
        producer_process.start()
        producer_process.join()
        producer_process.close()

    def __call__(self, message):
        if self.check_serial_script_running():
            print('Its running')
            self.send_message_to_script(message)
            return 0
        else:
            self.start_serial_script()
            time.sleep(3)
            print('Starting script')
            return 1

if __name__ == '__main__':
    serial_handler = Serial_handler()
    #serial_handler.start_serial_script()
    while(1):
        #serial_handler.kill_serial_script()
        serial_handler('yes yes')