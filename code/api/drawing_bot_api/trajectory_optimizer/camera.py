import cv2
import time
import os

class Camera:
    image_counter = 0

    def __call__(self):
        cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
        time.sleep(3)
        _,image = cap.read() # return a single frame in variable `frame`
        self.image_counter += 1
        cap.release()

        return image

if __name__ == '__main__':
    cam = Camera()
    cam()