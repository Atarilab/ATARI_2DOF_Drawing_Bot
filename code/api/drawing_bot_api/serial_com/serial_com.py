import serial
from config import *
import multiprocessing
import time
import setproctitle

setproctitle.setproctitle('drawing_bot_serial_com')

def handle_serial_commands(queue):
    serial_port = serial.Serial('/dev/cu.usbserial-0001', BAUD)
    while True:
        # Check if there is something in the queue
        if not queue.empty():
            message = queue.get()  # Retrieve data from the queue
            serial_port.write(message.encode('utf-8'))
            print(f"Consumer: Got {message} from queue")
        else:
            # If the queue is empty, break the loop (simulation stop condition)
            break

def print_input(queue):
    while True:
        if not queue.empty():
            print(queue.get())
            queue.put('AH YES')

def main():

    # Create the same queue that producer is using
    queue = multiprocessing.Queue()

    # Start the consumer process
    #consumer_process = multiprocessing.Process(target=handle_serial_commands, args=(queue,))
    consumer_process = multiprocessing.Process(target=print_input, args=(queue,))
    consumer_process.start()

    # Wait for the consumer to finish
    consumer_process.join()

if __name__ == "__main__":
    main()
