from drawing_bot_api.trajectory_optimizer.simulator import Simulator, ExponentialDecaySimulator
from drawing_bot_api.trajectory_optimizer.fourier_compensator import FourierCompensator
from drawing_bot_api.trajectory_optimizer.shape_generator import ShapeGenerator
from drawing_bot_api.commands import DrawingBot
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.trajectory_optimizer.utils import Scheduler
from drawing_bot_api.trajectory_optimizer.config_fourier import *
import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import select

#######################################################################################
## SETTINGS AND UTILS #################################################################
#######################################################################################

DEBUG_MODE = False
np.set_printoptions(threshold=np.inf)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_console():
    os.system('cls' if os.name=='nt' else 'clear')

def timeout_handler(signum, frame):
    raise TimeoutError("timeout occured")

signal.signal(signal.SIGALRM, timeout_handler)

########################################################################################
## DEFINES #############################################################################
########################################################################################

drawing_bot = DrawingBot(verbose=0)
simulator = Simulator()
compensator = FourierCompensator()
shape_generator = ShapeGenerator()
image_processor = ImageProcessor()

#########################################################################################
## PLOTTING FUNCTIONS ###################################################################
#########################################################################################

def plot_graph(data, labels, scale='linear', axis_labels=['x', 'y']):
    plt.yscale(scale)
    for _i in range(len(data)):
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.plot(data[_i], label=labels[_i])
    if labels[0] is not None:
        plt.legend(bbox_to_anchor=(1, 1.15), ncol=3)  
    plt.show() 

def plot_bar_graph(values, catagories, axis_labels):
    x = np.arange(len(values[0]))  # X positions
    width = 0.4  # Bar width

    # Plot bars
    plt.bar(x - width/2, values[0], width, label=catagories[0])
    plt.bar(x + width/2, values[1], width, label=catagories[1])

    # Labels & legend
    plt.xticks(x, x)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=3)

    plt.show()

#########################################################################################
## UTIL FUNCTIONS #######################################################################
#########################################################################################

def ask_if_hold_is_necessary():
    ready, _, _ = select.select([sys.stdin], [], [], WIPE_TIME)
    if ready:
        sys.stdin.readline().strip()
        input('Press enter to continue...')

def get_trajectory():
    for shape in shape_generator(seed=SHAPE_SEED):
        drawing_bot.add_shape(shape)
    trajectory = np.array(drawing_bot._get_all_points())
    template = drawing_bot.plot(training_mode=True, points=trajectory)
    drawing_bot.shapes.clear()
    return trajectory, template

def get_set_of_trajectories(nr=10):
    _trajectories = []
    _templates = []
    for _ in range(nr):
        while True:
            signal.alarm(3)
            
            try:
                print(f'Generating shape {_}...')
                clear_console()
                _trajectory, _template = get_trajectory()
                _trajectories.append(_trajectory)
                _templates.append(_template)
                break
            except Exception as e:
                print(f'Timeout occured')
            finally:
                signal.alarm(0)
    
    return _trajectories, _templates

def get_template_rewards(simulator, trajectories, templates):
    _rewards = []
    for _index in range(len(templates)):
        if SIMULATION_MODE:
            simulated_trajectory = simulator(trajectories[_index])
            drawing = drawing_bot.plot(training_mode=True, points=simulated_trajectory)
            _rewards.append(image_processor(templates[_index], drawing, save_images=False))
            image_processor.save_image(drawing, 'original', 'simulated', nr=0)
        else:
            print(f'Drawing reference drawing {_index}...')
            drawing_bot.execute(promting=False, points=trajectories[_index])
            _reward, _drawing, _ = image_processor(templates[_index], save_images=True, save_folder='templates', return_image=True, crop_drawing=DRAWING_CROP, crop_template=TEMPLATE_CROP)
            _rewards.append(_reward)

            print(f'{bcolors.OKCYAN}WIPE NOW!{bcolors.ENDC}')
            time.sleep(WIPE_TIME)
    return _rewards

def test_parameters(parameters, trajectory, template):
    compensated_trajectory, _, _ = compensator(trajectory, type='fourier', parameters=parameters) #, tanh_scaling=[4])
    if SIMULATION_MODE:
        simulated_trajectory = simulator(compensated_trajectory)
        drawing = drawing_bot.plot(training_mode=True, points=simulated_trajectory)
        reward = image_processor(template, drawing, save_images=False)
    else:
        drawing_bot.execute(promting=False, points=compensated_trajectory)
        reward, drawing, _ = image_processor(template, save_images=False, save_folder='evolutionary', return_image=True, crop_drawing=DRAWING_CROP, crop_template=TEMPLATE_CROP)
        print(f'{bcolors.OKCYAN}WIPE NOW!{bcolors.ENDC} Press enter to hold execution!')
        ask_if_hold_is_necessary()
    return reward, drawing

###############################################################################################
## MAIN FUNCTION FOR TRAINING #################################################################
###############################################################################################

def find_parameters_via_evo():
    best_set = None
    trajectory = None
    template = None
    best_set_history = []

    parameters = np.random.uniform(-RANDOM_PARAM_LIMIT, RANDOM_PARAM_LIMIT, (GENERATION_SIZE, NUM_OF_PARAMETERS))
    if INIT_SET is not None:
        parameters = np.repeat([INIT_SET], GENERATION_SIZE, axis=0)
        for i in range(1, GENERATION_SIZE-NEW_BRANCHES_PER_GEN):
            parameters[i] += np.random.normal(0, SIGMA_BASE, NUM_OF_PARAMETERS)

    sigma_schedule = Scheduler(SIGMA_BASE, SIGMA_DECAY)
    reward_history = []
    template_reward_history = []
    averaged_reward_history = []
    trajectories, templates = get_set_of_trajectories(nr=NUMBER_OF_SAMPLES)
    template_rewards = get_template_rewards(simulator, trajectories, templates)
    image_processor.save_image(templates[0], 'original', 'template', nr=0)

    for generation_index in range(GENERATION_START_NR, GENERATIONS+GENERATION_START_NR):
        signal.alarm(2000)

        try:
            sigma = sigma_schedule(x=generation_index)
            trajectory_index = generation_index % NUMBER_OF_SAMPLES #np.random.randint(0, NUMBER_OF_TRAJECTORIES-1)
            trajectory = trajectories[trajectory_index]
            template = templates[trajectory_index]
            template_reward = template_rewards[trajectory_index]

            rewards = []
            drawings = []
            for _set_index in range(len(parameters)):
                print(f'Testing parameters: {np.round(parameters[_set_index], 3)}')
                reward, drawing = test_parameters(parameters[_set_index], trajectory, template)
                rewards.append(reward)
                drawings.append(drawing)
                print(f'Generation: {generation_index} \t| Set: {_set_index}/{GENERATION_SIZE-1} \t| Reward: {np.round(reward, 3)}')
                #image_processor.save_image(drawing, 'evolutionary', f'generation_{_set_index}_{reward}', nr=generation_index)

            reward_history.append(max(rewards))
            template_reward_history.append(template_reward)
            reward_delta = max(rewards) - template_reward
            averaged_reward = np.mean(reward_history[np.max([len(reward_history)-30, 0]):])
            averaged_reward_history.append(averaged_reward)
            
            # Get indices of the three best sets
            rewards = np.nan_to_num(rewards, nan=-np.inf)
            best_indices = np.argsort(rewards)[-KEEP_SETS:]
            best_sets = [parameters[i] for i in best_indices]
            best_set_history.append(parameters[np.argmax(rewards)])
            #if best_set is not None:
            #    best_set = best_set + LEARNING_RATE * (parameters[np.argmax(rewards)] - best_set) * np.abs(reward_delta)
            #else:
            best_set = parameters[np.argmax(rewards)]

            print(f'{bcolors.OKCYAN}generation: {generation_index}{bcolors.ENDC}\tbest_set: {np.round(best_set, 2)}\t{bcolors.OKGREEN}reward: {np.round(max(rewards), 3)} / {np.round(template_reward, 3)}{bcolors.ENDC}\tsigma: {np.round(sigma, 3)}')
            
            parameters = np.random.uniform(-RANDOM_PARAM_LIMIT, RANDOM_PARAM_LIMIT, (GENERATION_SIZE, NUM_OF_PARAMETERS))
            parameters[0] = best_set
            for i in range(1, GENERATION_SIZE-NEW_BRANCHES_PER_GEN):
                parameters[i] = best_sets[i % KEEP_SETS]
                parameters[i] += np.random.normal(0, sigma, NUM_OF_PARAMETERS)

            if generation_index % 10 == 9:
                pass
            image_processor.save_image(drawings[np.argmax(rewards)], 'evolutionary', 'generation', nr=generation_index)

            plt.close('all')
        except Exception as e:
            if DEBUG_MODE:
                raise
            else:
                print(f"Timeout occurred: {e}")

        finally:
            signal.alarm(0)  # Cancel the alarm

    #best_set = best_set_history[np.argmax(averaged_reward_history)]
    print(f'BEST SET: {best_set.tolist()}')
    reward_delta = np.array(reward_history) - np.array(template_reward_history)
    plot_graph([reward_history, template_reward_history], ['reward', 'reward without compensation'], scale='linear', axis_labels=['generation', 'reward'])
    plot_graph([averaged_reward_history], ['averaged_reward'], scale='linear', axis_labels=['generation', 'reward'])

    compensated_trajectory, fourier_series, point_offsets = compensator(trajectory, type='fourier', parameters=best_set)
    if SIMULATION_MODE:
        simulated_trajectory = simulator(compensated_trajectory)
        simulated_template = simulator(trajectory)
        drawing = drawing_bot.plot(training_mode=True, points=simulated_trajectory, plot_in_training=True)
        uncompensated_drawing = drawing_bot.plot(training_mode=True, points=simulated_template, plot_in_training=True)
    plot_graph([fourier_series], ['compensation_function'], scale='linear', axis_labels=['x', 'f(x)'])
    plot_graph([point_offsets], ['point offset'], scale='linear', axis_labels=['point_index', 'offset'])

    return best_set

#################################################################################################
## MAIN FUNCTION FOR EVALUATINO #################################################################
#################################################################################################

def evaluate_parameters(parameters):
    trajectories, templates = get_set_of_trajectories(NUMBER_OF_EVALUATION_SAMPLES)
    template_rewards = get_template_rewards(simulator, trajectories, templates)
    
    print(f'BEST PARAMETERS: {np.round(parameters, 3)}')

    rewards = []

    for _index in range(len(trajectories)):
        reward, drawing = test_parameters(parameters, trajectories[_index], templates[_index])
        rewards.append(reward)
        image_processor.save_image(drawing, 'evolutionary', 'evaluation', nr=_index)

    # plot rewards
    plot_bar_graph([rewards, template_rewards], ['rewards', 'rewards without compensation'], axis_labels=['sample', 'reward'])
'''
    for i in range(len(trajectories)):
        # plot fourier series
        compensated_trajectory, fourier_series, point_offsets = compensator(trajectories[i], type='fourier', parameters=parameters)
        #simulated_trajectory = simulator(compensated_trajectory)
        #simulated_template = simulator(trajectories[i])
        #drawing = drawing_bot.plot(training_mode=True, points=simulated_trajectory, plot_in_training=True)
        #uncompensated_drawing = drawing_bot.plot(training_mode=True, points=simulated_template, plot_in_training=True)
        plot_graph([fourier_series], ['fourier_function'], scale='linear', axis_labels=['x', 'f(x)'])
        plot_graph([point_offsets], ['point offset'], scale='linear', axis_labels=['x', 'f(x)'])
'''

###############################################################################################
## MAIN #######################################################################################
###############################################################################################

if __name__ == '__main__':
    best_set = find_parameters_via_evo()
    print(f'BEST SET: {best_set}')
    #evaluate_parameters(best_set)