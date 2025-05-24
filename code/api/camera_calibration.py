import drawing_bot_api.trajectory_optimizer
import drawing_bot_api.trajectory_optimizer.fourier_compensator
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.commands import DrawingBot
from drawing_bot_api.trajectory_optimizer.shape_generator import ShapeGenerator
import drawing_bot_api
import cv2
import numpy as np
import sys
import select

DRAWING_CROP = [0, 55, 187, 203] # top # bottom # left # right
TEMPLATE_CROP = [135, 110, 170, 135]

shape_generator = ShapeGenerator()
drawing_bot = DrawingBot()
image_processor = ImageProcessor()

for shape in shape_generator():
    drawing_bot.add_shape(shape)

trajectory = drawing_bot._get_all_points()
template = drawing_bot.plot(training_mode=True, points=trajectory) 

drawing_bot.execute(promting=False, points=trajectory)

reward, drawing, template = image_processor(template, save_images=False, return_image=True, crop_drawing=DRAWING_CROP, crop_template=TEMPLATE_CROP)
template_array = np.array(template)
drawing_array = np.array(drawing)
difference_image = np.zeros(np.shape(template_array))
difference_image = np.expand_dims(difference_image, axis=-1)
difference_image = np.repeat(difference_image, 3, axis=-1)
print(np.shape(difference_image))
difference_image[:, :, 0] = template_array
difference_image[:, :, 2] = drawing_array

cv2.imshow('difference', difference_image)
#cv2.imshow('template', template)
#cv2.imshow('drawing', drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()

input 
