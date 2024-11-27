from drawing_bot_api.trajectory_optimizer.shape_generator import ShapeGenerator
from drawing_bot_api.trajectory_optimizer.shape_generator import RESTING_POINT
from drawing_bot_api import DrawingBot
import cv2
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor

shape_generator = ShapeGenerator()
drawing_bot = DrawingBot()
image_processor = ImageProcessor()

drawing_bot.move_to_point(RESTING_POINT)

for shape in shape_generator():
    drawing_bot.add_shape(shape)

template = drawing_bot.plot(training_mode=True)

drawing_bot.execute(promting=True)

score = image_processor(template)
print(score)

#cv2.imshow("generated shape", template)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

