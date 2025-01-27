from drawing_bot_api.trajectory_optimizer.shape_generator import ShapeGenerator
from drawing_bot_api.trajectory_optimizer.shape_generator import RESTING_POINT
from drawing_bot_api import DrawingBot
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.trajectory_optimizer.wiper import Wiper
from drawing_bot_api.trajectory_optimizer.simulator import Simulator
from drawing_bot_api.trajectory_optimizer.training_v1 import Trainer

shape_generator = ShapeGenerator()
drawing_bot = DrawingBot()
image_processor = ImageProcessor()
wiper = Wiper()
error_simulator = Simulator(strength=15, pattern_length=20)
model = Trainer('ignore')

#drawing_bot.move_to_point(RESTING_POINT)

for shape in shape_generator():
    drawing_bot.add_shape(shape)

points = drawing_bot._get_all_points()
#points = error_simulator(points)

template = drawing_bot.plot(training_mode=True)

#--------------------------------------------------------------------

#cv2.imshow("generated shape", template)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

drawing_bot.execute(promting=False, points=points)

for shape in shape_generator():
    drawing_bot.add_shape(shape)

template2 = drawing_bot.plot(training_mode=True)

score = image_processor(template)#, drawing=template)
print(score)
wiper()

#cv2.imshow("generated shape", template)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

drawing_bot.move_to_point(RESTING_POINT)

