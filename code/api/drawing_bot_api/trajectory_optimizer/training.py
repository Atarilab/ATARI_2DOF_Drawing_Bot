from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import keras
import numpy as np
import tensorflow as tf
from drawing_bot_api.trajectory_optimizer.config import *
from drawing_bot_api.config import PLOT_XLIM, PLOT_YLIM
import os
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.logger import Log
from math import sqrt, pow, atan2

class LossHistory(keras.callbacks.Callback):
    losses = [] 
    def on_train_begin(self, logs={}):
        #self.losses = []
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    
class Step:
    def __init__(self, state, action):
        self.state = state
        self.action = action

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def __call__(self, step):
        self.buffer.append(step)

    def clear(self):
        self.buffer.clear()

class Trainer:
    actor = None
    critic = None
    
    replay_buffers = ReplayBuffer()
    states_history = []
    offsets_history = []
    reward_history = []
    image_processor = ImageProcessor()
    log = Log(verbose=1)
    loss_history = LossHistory()

    def __init__(self, model=None, **kwargs):
        if model == None:
            self.new_model(**kwargs)
        elif model == 'ignore':
            self.actor = None
        else:
            self.load_model(model)

    #####################################################################
    # INFERENCE METHODS
    #####################################################################

    def _normalize_trajectory(self, trajectory):
        for point in trajectory:
            point[0] = point[0] / PLOT_XLIM[1]
            point[1] = point[1] / PLOT_YLIM[1]

    def adjust_trajectory(self, trajectory, exploration_factor=0):
        _phases = self._points_to_phases(trajectory)                        # turn two dimensional points of trajectory into phases (one dimensional)
        _states = np.array(self._get_states(_phases))                       # batch phases together into sequences of phases with dimension input_size
        self.states_history.append(_states)
        _offsets = self._get_offsets(_states, exploration_factor)           # actor inference, returns two dimensional offset
        self.offsets_history.append(_offsets)
        _adjusted_trajectory = self._apply_offsets(_offsets, trajectory)    # add offset to originial trajectory
        return _adjusted_trajectory
        
    def _apply_offsets(self, offsets, trajectory):
        # check if dimensions are right
        if len(offsets) != len(trajectory):
            print('Dimension of inferenced offsets and original trajectory dont match')
            return 0
        
        # add offset to trajectory
        _new_trajectory = []
        for _point_index in len(trajectory):
            _new_point_x = trajectory[_point_index][0]+offsets[_point_index][0]
            _new_point_y = trajectory[_point_index][1]+offsets[_point_index][1]
            _new_trajectory.append([_new_point_x, _new_point_y])

        return _new_trajectory

    def _get_offsets(self, _states, exploration_factor):
        _offsets = self.actor.predict(_states, batch_size=len(_states))

        # exchange some offsets with a random offset with probability exploration factor
        for _offset in _offsets:
            if np.random.random(1) < exploration_factor:
                _offset = (np.random.random(2) * 2 ) - 1
        return _offsets

    def _get_states(self, _phases):
        _states = []
        for _current_point_index in range(len(_phases)):
            _state = []
            for _offset_from_current_point in range(-(INPUT_DIM-NUM_LEADING_POINTS), NUM_LEADING_POINTS):
                try:
                    _state.append(_phases[_current_point_index+_offset_from_current_point])
                except:
                    _state.append(0)
            _states.append(_state)
        return _states
                
    def _get_phase(self, point, prev_point):
        _pointing_vector = [point[0]-prev_point[0], point[1]-prev_point[1]]
        _phase = atan2(_pointing_vector[1], _pointing_vector[0])
        return _phase
    
    def _points_to_phases(self, trajectory):
        _phases = []
        for _i in range(1, len(trajectory)):
            _point = trajectory[_i]
            _prev_point = trajectory[_i -1]
            _phases.append(self._get_phase(_point, _prev_point))
        return _phases

    #####################################################################
    # TRAINING METHODS
    #####################################################################

    def train(self, reward):
        self._update_actor_and_critic(reward)

    def _update_actor_and_critic(self, reward):
        gamma = 0.99
        _states = self.states_history[-1]
        _offsets = self.offsets_history[-1]

        if len(_states) != len(_offsets):
            print('Dimensional mismatch between states and trajectory pulled from history')
            return 1
        
        for _t in range(len(_states)):
            if _t == (len(_states) + 1):
                v_target = [reward]
            else:
                v_target = [reward + gamma * self.critic.predict(_states[_t+1])]

            self.critic.fit(_states[_t], v_target)
            
            # Compute advantage for actor update
            advantage = v_target - self.critic.predict(_states[_t])
            advantage_vector = np.full_like(_offsets[_t], advantage)

            # Train actor using advantage
            self.actor.fit(_states[_t], advantage_vector)

    #####################################################################
    # MODEL CREATION, SAVING and LOADING
    #####################################################################

    def new_model(self, input_size=INPUT_DIM, output_size=ACTION_DIM, hidden_layer_size=HIDDEN_LAYER_DIM):
        # create critic
        _inputs_critic = keras.layers.Input(shape=(input_size,))
        _hidden_critic = keras.layers.Dense(hidden_layer_size, activation="relu")
        _output_critic = keras.layers.Dense(1)
        self.critic = Sequential([_inputs_critic, _hidden_critic, _output_critic])

        # create actor
        _inputs_actor = keras.layers.Input(shape=(input_size,))
        _hidden_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _output_actor = keras.layers.Dense(output_size, activation='tanh')
        self.actor = Sequential([_inputs_actor, _hidden_actor, _output_actor])

        # compile
        _optimizer = keras.optimizers.SGD()
        _loss = keras.losses.MSE()

        self.actor.compile(optimizer=_optimizer, loss=_loss, metrics=['accuracy'])
        self.critic.compile(optimizer=_optimizer, loss=_loss, metrics=['accuracy'])

    def load_model(self, model_id):
        # Looks how many models are in directory
        # If model > number of models, return Error
        # else load model from directory
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _num_of_models = len(os.listdir(os.path.join(_script_dir, f'models')))
        if model_id > _num_of_models:
            print('Model does not exist')
            exit()
        _path = os.path.join(_script_dir, f'models/model_{str(model_id)}.h5')
        model = load_model(_path)

    def save_model(self, model_id=None):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _num_of_models = len(os.listdir(os.path.join(_script_dir, f'models')))
        _model_id = 0

        if model_id:
            if model_id > _num_of_models:
                print(f'Chosen model ID is not valid. Will set model id to {_num_of_models} .')
                _model_id = _num_of_models
            else:
                _model_id = model_id
        else:
            _model_id = _num_of_models

        _path = os.path.join(_script_dir, f'models/model_{str(_model_id)}.h5')
        self.actor.save(_path)
