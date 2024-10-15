
import multiprocessing
import time
import random

def producer(queue):
    while True:
        # Simulate data production
        data = random.randint(1, 100)  # Generate some random data
        queue.put(data)  # Put data into the queue
        print(f"Produced: {data}")
        time.sleep(1)  # Simulate a delay in production

def consumer(queue):
    while True:
        data = queue.get()  # Get data from the queue
        print(f"Consumed: {data}")
        # Process the data (replace with actual processing)
        time.sleep(2)  # Simulate processing time

if __name__ == '__main__':
    # Create a queue for communication
    queue = multiprocessing.Queue()

    # Create producer and consumer processes
    producer_process = multiprocessing.Process(target=producer, args=(queue,))
    consumer_process = multiprocessing.Process(target=consumer, args=(queue,))

    # Start the processes
    producer_process.start()
    consumer_process.start()

    try:
        # Keep the main program running
        while True:
            time.sleep(1)  # Simulate doing other things
    except KeyboardInterrupt:
        print("Shutting down...")
        producer_process.terminate()  # Terminate the producer process
        consumer_process.terminate()  # Terminate the consumer process
        producer_process.join()  # Wait for the producer to finish
        consumer_process.join()  # Wait for the consumer to finish