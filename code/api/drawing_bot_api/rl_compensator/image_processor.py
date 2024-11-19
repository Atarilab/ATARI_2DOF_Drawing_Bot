from camera import Camera
import cv2
import os
import numpy as np

class Image_Processor:
    def __init__(self):
        self._camera = Camera()
        self._image_counter = 0

    def save_image(self, image, directory):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(_script_dir, f'images/{directory}/image_{str(self._image_counter)}.png')
        cv2.imwrite(_path, image)
        print(f'Saved image to {_path}')
    
    def simplify(self, image):
        # Blurring for noise reduction
        _blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Enhance Contrast
        _gray = cv2.cvtColor(_blurred, cv2.COLOR_BGR2GRAY)
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _enhanced_contrast = _clahe.apply(_gray)

        # Convert to Black and White
        _, _black_and_white = cv2.threshold(_enhanced_contrast, 80, 255, cv2.THRESH_BINARY)

        # Invert
        _inverted = cv2.bitwise_not(_black_and_white)

        return _inverted
        

    def __call__(self, template):
        _template = template
        _drawing = self._camera()
        self._image_counter += 1

        _simpl_drawing = self.simplify(_drawing)
        self.save_image(_simpl_drawing, 'edited')

        contours, _ = cv2.findContours(_simpl_drawing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_trajectories = []
        for contour in contours:
            # Extract points of each trajectory
            trajectory_points = contour.reshape(-1, 2)
            all_trajectories.append(trajectory_points)

            # Optional: Draw each trajectory for visualization
            cv2.drawContours(_drawing, [contour], -1, (0, 255, 0), 1)

        #trajectory_image = np.zeros_like(_drawing)

        cv2.imshow('All Trajectories', _drawing)
        cv2.waitKey(0)

img_proc = Image_Processor()
img_proc(None)

    