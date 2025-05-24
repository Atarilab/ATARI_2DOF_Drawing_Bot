'''import numpy as np
import matplotlib.pyplot as plt


data_points = np.random.normal(0.4, 0.05, 50)
scaler = np.arange(0, 1, 1/len(data_points)) * 0.1
data_points_ref = np.random.normal(data_points-scaler, 0.05)

plt.xlabel('generation')
plt.ylabel('reward')
plt.plot(data_points, label='reward')
plt.plot(data_points_ref, label='template reward')

plt.legend(bbox_to_anchor=(1, 1.15), ncol=3)  
plt.show() '''

from drawing_bot_api.commands import DrawingBot
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt
from drawing_bot_api.trajectory_optimizer.shape_generator import ShapeGenerator
from drawing_bot_api.trajectory_optimizer.simulator import Simulator
from drawing_bot_api.trajectory_optimizer.utils import normalize_linear_incl_neg, normalize_linear_only_pos

drawing_bot = DrawingBot()
image_processor = ImageProcessor()
shape_generator = ShapeGenerator()
simulator = Simulator()

def get_images_of_individual_points(trajectory, resolution):
    _images = [] 
    _points = np.array(trajectory)

    for _index in range(0, int(len(_points)), resolution):
        _images.append(drawing_bot.plot(training_mode=True, points=_points[_index:_index+resolution]))
    plt.close('all')
    return _images

def save_visualization(trajectory, rewards, cycle_index, name):
    reward_assigned_drawing = drawing_bot.plot(training_mode=True, points=trajectory, color_assignment=rewards)
    image_processor.save_image(reward_assigned_drawing, name, f'{name}_visualization', cycle_index)
    del reward_assigned_drawing

GRANULAR_REWARD = True
def get_rewards(template, drawing, trajectory):
        if GRANULAR_REWARD:
            resolution = 5
            _images_of_template_points = get_images_of_individual_points(trajectory, resolution)
            rewards = np.array(image_processor.calc_rewards_for_individual_points(_images_of_template_points, drawing))
            rewards = np.repeat(rewards, resolution, axis=0)
            rewards = np.nan_to_num(rewards, nan=1)
            
            return rewards
        
        else:
            reward = image_processor(template, drawing=drawing)
            if reward is None:
                return None
            return reward
for i in range(100):  
    for shape in shape_generator():
        drawing_bot.add_shape(shape)
    # get template for drawing
    trajectory = np.array(drawing_bot._get_all_points())
    template = np.array(drawing_bot.plot(training_mode=True, points=trajectory))
    drawing_bot.shapes.clear()

    adjusted_trajectory = trajectory + np.random.normal(0, 0.15, trajectory.shape)
    simulated_trajectory = simulator(adjusted_trajectory)
    simulated_drawing = drawing_bot.plot(training_mode=True, points=simulated_trajectory)

    rewards = get_rewards(template, simulated_drawing, trajectory)
    rewards = 1-normalize_linear_only_pos(None, rewards)
    print(rewards)
    save_visualization(trajectory, None, 2*i, 'reward')
    save_visualization(simulated_trajectory, rewards, 2*i+1, 'reward')