from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Lambda, Conv1D, MaxPool1D, Flatten
from keras.api.models import load_model
import keras.api.backend as K
import keras
from keras import ops
import numpy as np
import tensorflow as tf
from drawing_bot_api.trajectory_optimizer.config import *
from drawing_bot_api.config import PLOT_XLIM, PLOT_YLIM
import os
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.logger import Log
from math import sqrt, pow, atan2, pi, tanh
import sys
from drawing_bot_api.trajectory_optimizer.custom_losses import *
from drawing_bot_api.trajectory_optimizer.utils import *

############################################################
# TRAINER ##################################################
############################################################

class Trainer:
    model = None
    
    # training histories
    trajectory_history = []
    states_history = []
    action_history = []
    reward_history = []
    output_history = []

    # utils and logs
    image_processor = ImageProcessor()
    log = Log(verbose=VERBOSE)
    loss_history = LossHistory()

    def __init__(self, model=None):
        if model == None:
            self.new_model()
        elif model == 'ignore':
            self.actor = None
        else:
            self.load_model(model)

    #####################################################################
    # INFERENCE METHODS
    #####################################################################

    def adjust_trajectory(self, trajectory, template_rewards, exploration_factor=0):
        np.set_printoptions(threshold=np.inf)
        self.trajectory_history.append(trajectory)
        _phases = self._points_to_phases(trajectory)                        # turn two dimensional points of trajectory into phases (one dimensional)
        _states = np.array(self._get_states(_phases))                       # batch phases together into sequences of phases with dimension input_size
        self.states_history.append(_states)
        _offsets = self._get_offsets(_states, exploration_factor, template_rewards)           # actor inference, returns two dimensional offset
        self.action_history.append(_offsets)
        _adjusted_trajectory = self._apply_offsets(_offsets, trajectory)    # add offset to originial trajectory
        self.adjusted_trajectory_history.append(_adjusted_trajectory)
        return _adjusted_trajectory
        
    def _apply_offsets(self, offsets, trajectory):
        # calculate difference between amount of trajectory points and amount of offsets
        _index_offset = len(trajectory) - len(offsets[0])
        _offset_x, _offset_y = offsets

        # add unaccounted for values to trajectory
        _new_trajectory = [trajectory[0]]
        if USE_PHASE_DIFFERENCE:
            _new_trajectory.append(trajectory[1])

        # apply offset
        for _point_index in range(_index_offset, len(trajectory)):
            _new_point_x = trajectory[_point_index][0] + OUTPUT_SCALING * _offset_x[_point_index-_index_offset]
            _new_point_y = trajectory[_point_index][1] + OUTPUT_SCALING * _offset_y[_point_index-_index_offset]
            _new_trajectory.append([_new_point_x, _new_point_y])

        return _new_trajectory

    def _get_offsets(self, _states):
        _output = np.array(self.model.predict(_states, batch_size=len(_states), verbose=0))
        self.output_history.append(_output)

        _mu1 = _output[0]
        _mu2 = _output[1]
        _sigma1 = _output[2]
        _sigma2 = _output[3]

        _offset_x = np.random.normal(loc=_mu1, scale=np.clip(_sigma1, 0, 100))
        _offset_y = np.random.normal(loc=_mu2, scale=np.clip(_sigma2, 0, 100))

        return [_offset_x, _offset_y]
    
    def _get_states(self, phases):
        _phases = phases
        _normalize_factor = 2 * pi
        if USE_PHASE_DIFFERENCE:
            _phases = np.array(self._get_phase_difference(phases))
            _normalize_factor = pi

        _phases = (np.array(_phases) / _normalize_factor) + 0.5

        _phases = np.append(np.zeros(INPUT_DIM - NUM_LEADING_POINTS), _phases)
        window_size = INPUT_DIM
        _states = np.array([_phases[i:i + window_size] for i in range(len(_phases) - window_size + 1)])
        
        _states = shift_to_range(_states, 0.5)

        if ADD_PROGRESS_INDICATOR:
            _num_of_states = np.shape(_states)[0]
            _progress_indicators = np.arange(_num_of_states)
            _progress_indicators = _progress_indicators / _num_of_states
            _progress_indicators = _progress_indicators.reshape(-1 ,1)
            _states = np.hstack((_states, _progress_indicators))

        return _states
    
    def _get_phase_difference(self, phases):
        _phase_differences = []

        for _index in range(1, len(phases)):
            
            _phase = phases[_index]
            _prev_phase = phases[_index - 1]
            _phase_difference = _phase-_prev_phase

            if abs(_phase_difference) > pi:
                _phase_difference = -np.sign(_phase_difference) * ((2 * pi) - abs(_phase_difference))
            
            _phase_differences.append(_phase_difference)
        
        return _phase_differences
    
    def _points_to_phases(self, trajectory):
        _points = np.array(trajectory)[1:]
        _prev_points = np.array(trajectory)[:-1]
        _direction_vectors = _points - _prev_points
        _direction_vectors = _direction_vectors.T
        _phases = np.arctan2(_direction_vectors[1], _direction_vectors[0])
        return _phases

    #####################################################################
    # TRAINING METHODS
    #####################################################################

    def train(self, reward):
        return self._update_model(reward)

    def _update_model(self, reward):
        _states = self.states_history[-1]
        _actions = np.array(self.action_history[-1])
        _model_output = self.output_history[-1]

        # CRITIC ########

        _critic_predictions = _model_output[4:]

        _reward = []
        if SPARSE_REWARDS and GRANULAR_REWARD:
            _prev_reward = reward[0]
            for _r in reward:
                if _r != _prev_reward:
                    _reward.append(_prev_reward)
                else:
                    _reward.append(0)
                _prev_reward = _r
        else:
            _reward = reward

        _reward = np.array(_reward).reshape(-1, 1)

        _v_targets = REWARD_DISCOUNT * _critic_predictions
        _v_targets = _v_targets[1:]

        if not CUMULATIVE_VALUE_TRAINING: # This means the final reward is given at every point
            _repeated_reward = np.repeat(_reward, len(_v_targets), axis=0)
            _v_targets = _repeated_reward[:len(_v_targets)] + _v_targets

        if GRANULAR_REWARD: # Alternative reward calculation method that allows to calculate a reward in intervals
            _v_targets = _reward[:len(_v_targets)] + _v_targets

        _v_targets = np.append(_v_targets, np.mean(_reward))
        _v_targets = _v_targets.reshape(-1, 1)

        self.critic.fit(_states, _v_targets, batch_size=len(_states), callbacks=[self.loss_critic_log])
            
        # ACTOR #########

        _actions = np.array(_actions)
        _actions = _actions.squeeze().T

        # calc advantage
        _advantage = _v_targets - _critic_predictions
        _advantage = _advantage[:len(_actions)]
        _advantage = self._normalize_advantage_subtract_mean(_advantage)

        _actor_ytrue = tf.concat([_actions, _advantage], axis=1)

        self.model.fit(_states, [_actor_ytrue, _v_targets], batch_size=32, callbacks=[self.loss_history])

        self.action_history.clear()
        self.states_history.clear()
        self.trajectory_history.clear()

        return 0

    #####################################################################
    # MODEL CREATION, SAVING and LOADING
    #####################################################################

    def new_model(self):

        _input_size = INPUT_DIM + 1

        _input = keras.layers.Input(shape=(_input_size,))

        _hidden_1 = Dense(HIDDEN_LAYER_DIM_ACTOR, activation="relu")(_input)
        _hidden_2 = Dense(HIDDEN_LAYER_DIM_ACTOR, activation="relu")(_hidden_1)
        _hidden_3 = Dense(HIDDEN_LAYER_DIM_ACTOR, activation="relu")(_hidden_2)
        _hidden_4 = Dense(HIDDEN_LAYER_DIM_ACTOR, activation="relu")(_hidden_3)
        _hidden_5 = Dense(HIDDEN_LAYER_DIM_ACTOR, activation="relu")(_hidden_4)
        _hidden_6 = Dense(HIDDEN_LAYER_DIM_ACTOR, activation="relu")(_hidden_5)

        _output_mu1 = Dense(1, activation='tanh', name='mu1')(_hidden_6)
        _output_mu2 = Dense(1, activation='tanh', name='mu2')(_hidden_6)
        _merged_mus = Lambda(lambda x: tf.concat(x, axis=-1), name="merged_mus")([_output_mu1, _output_mu2])

        _sigma_initializer = keras.initializers.RandomUniform(-SIGMA_INIT_WEIGHT_LIMIT, SIGMA_INIT_WEIGHT_LIMIT)
        _output_sigma1 = Dense(1, activation='softplus', name='sigma1', kernel_initializer=_sigma_initializer)(_hidden_6)
        _output_sigma1 = Lambda(lambda x: SIGMA_OUTPUT_SCALING * x)(_output_sigma1)
        _output_sigma2 = Dense(1, activation='softplus', name='sigma2', kernel_initializer=_sigma_initializer)(_hidden_6)
        _output_sigma2 = Lambda(lambda x: SIGMA_OUTPUT_SCALING * x)(_output_sigma2)
        _merged_sigmas = Lambda(lambda x: tf.concat(x, axis=-1), name="merged_sigmas")([_output_sigma1, _output_sigma2])

        _output_critic = Dense(1, activation='linear', name='output critic')(_hidden_6)
        _output_actor = Lambda(lambda x: tf.concat(x, axis=-1), name="output actor")([_merged_mus, _merged_sigmas])

        self.model = Model(inputs=_input, outputs=[_output_actor, _output_critic])

        # compile #######################################################
        _optimizer = keras.optimizers.Adam(learning_rate=LR)
        _loss = custom_loss
        _critic_loss = keras.losses.MeanSquaredError()

        self.model.compile(optimizer=_optimizer, loss=[_loss, _critic_loss])

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
