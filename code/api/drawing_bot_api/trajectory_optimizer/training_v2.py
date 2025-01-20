from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.models import load_model
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

print(tf.config.list_physical_devices('GPU'))

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = [] 

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


def pass_through_loss(y_true, y_pred):
    return ops.mean(y_true * y_pred, axis=-1)

class Trainer:
    actor = None
    critic = None
    
    replay_buffers = ReplayBuffer()
    states_history = []
    action_history = []
    reward_history = []
    adjusted_trajectory_history = []
    image_processor = ImageProcessor()
    log = Log(verbose=VERBOSE)
    loss_history_actor = LossHistory()
    loss_history_critic = LossHistory()
    absolute_average_action_history = []

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


    def adjust_trajectory(self, trajectory, exploration_factor=0):
        _phases = self._points_to_phases(trajectory)                        # turn two dimensional points of trajectory into phases (one dimensional)
        self.log(f'Phases: {_phases}')
        _states = np.array(self._get_states(_phases))                       # batch phases together into sequences of phases with dimension input_size
        self.states_history.append(_states)
        _offsets = self._get_offsets(_states, exploration_factor)           # actor inference, returns two dimensional offset
        self.action_history.append(_offsets)
        self.absolute_average_action_history.append(np.mean(np.abs(_offsets)))
        _adjusted_trajectory = self._apply_offsets(_offsets, trajectory)    # add offset to originial trajectory
        self.adjusted_trajectory_history.append(_adjusted_trajectory)
        return _adjusted_trajectory
        
    def _apply_offsets(self, offsets, trajectory):
        # calculate difference between amount of trajectory points and amount of offsets
        _index_offset = len(trajectory) - len(offsets)
        
        # add unaccounted for values to trajectory
        _new_trajectory = [trajectory[0]]
        if USE_PHASE_DIFFERENCE:
            _new_trajectory.append(trajectory[1])

        # apply offset
        for _point_index in range(_index_offset, len(trajectory)):
            _new_point_x = trajectory[_point_index][0]+offsets[_point_index-_index_offset][0]
            _new_point_y = trajectory[_point_index][1]+offsets[_point_index-_index_offset][1]
            _new_trajectory.append([_new_point_x, _new_point_y])

        return _new_trajectory

    def _get_offsets(self, _states, exploration_factor):
        _offsets = self.actor.predict(_states, batch_size=len(_states), verbose=0)

        # exchange some offsets with a random offset with probability exploration factor
        for _offset_index in range(len(_offsets)):
            if np.random.random(1) < exploration_factor:
                _offsets[_offset_index] = (np.random.random(2) * 2 ) - 1
        return _offsets
    
    def _get_state(self, phases, index):
            _state = []
            for _offset_from_current_point in range(-(INPUT_DIM-NUM_LEADING_POINTS), NUM_LEADING_POINTS):
                try:
                    _state.append(phases[index+_offset_from_current_point])
                except:
                    _state.append(0)
            return _state

    def _get_states(self, phases):
        _phases = phases

        if USE_PHASE_DIFFERENCE:
            _phases = self._get_phase_difference(phases)
        
        self.log(f'Phase (-difference): {_phases}')
        
        _states = []

        for _current_point_index in range(len(_phases)):
            _states.append(self._get_state(_phases, _current_point_index))
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
                
    def _get_phase(self, point, prev_point):
        _pointing_vector = [point[0]-prev_point[0], point[1]-prev_point[1]]
        _phase = atan2(_pointing_vector[1], _pointing_vector[0])
        return _phase
    
    def _points_to_phases(self, trajectory):
        _points = np.array(trajectory)[1:]
        _prev_points = np.array(trajectory)[:-1]
        _direction_vectors = _points - _prev_points
        _direction_vectors = _direction_vectors.T
        _phases = np.arctan2(_direction_vectors[1], _direction_vectors[0])
        return _phases
    
    def _points_to_phases_old(self, trajectory):
        _phases = []
        for _i in range(1, len(trajectory)):
            _point = trajectory[_i]
            _prev_point = trajectory[_i -1]
            _phases.append(self._get_phase(_point, _prev_point))
        return _phases
    
    def _get_adjusted_states(self, states, adjusted_phases):
        _adjusted_phases = adjusted_phases

        if USE_PHASE_DIFFERENCE:
            _adjusted_phases = self._get_phase_difference(_adjusted_phases)
        
        if len(states) != len(_adjusted_phases):
            self.log(f'def _get_adjusted_states(): "Length of states ({len(states)}) and adjusted phases ({len(_adjusted_phases)}) does not match"')

        _adjusted_states = [] 
        for _index in range(len(states)):
            _new_state = states[_index]
            _new_state[INPUT_DIM-NUM_LEADING_POINTS] = _adjusted_phases[_index]
            _adjusted_states.append(_new_state)
        
        return _adjusted_states

    #####################################################################
    # TRAINING METHODS
    #####################################################################

    def train(self, reward):
        self._update_actor_and_critic(reward)

    def _update_actor_and_critic(self, reward):
        gamma = 0.99
        _states = self.states_history[-1]
        _actions = self.action_history[-1]

        # generate states where the offset is applied to (and only to) the center point, therefore representing clear state transistions
        _adjusted_trajectory = self.adjusted_trajectory_history[-1]
        _adjusted_phases = self._points_to_phases(_adjusted_trajectory)
        _adjusted_states = self._get_adjusted_states(_states, _adjusted_phases)

        _critic_predictions_without_actions = self.critic.predict(np.array(_states), verbose=0)
        _critic_predictions_with_actions = self.critic.predict(np.array(_adjusted_states), verbose=0)

        print(f'Critic value predictions: {np.mean(_critic_predictions_with_actions)}')

        # calc value targets
        _v_targets = reward + gamma * _critic_predictions_with_actions
        _v_targets = _v_targets[:-1]
        _v_targets = np.append(_v_targets, reward)
        _v_targets = _v_targets.reshape(-1, 1)

        # calc advantage
        #print(f'{_critic_predictions_with_actions[0]} - {_critic_predictions_without_actions[0]} = {_critic_predictions_with_actions[0] - _critic_predictions_without_actions[0]}')
        #_advantage = (_critic_predictions_with_actions - _critic_predictions_without_actions) * 100000000
        _advantage = _v_targets - _critic_predictions_without_actions
        _actor_loss = -_advantage
        print(f'Advantage: {np.mean(_advantage)}\tActor loss: {np.mean(_actor_loss)}')
        _actor_loss = np.repeat(_actor_loss, 2, axis=1)

        #print(f'_v_targets: {_v_targets}, length states: {len(_states)}, length _v_targets: {len(_v_targets)}')
        #print(f'_advantages: {_advantages}, length of advantages: {len(_advantages)}')
        if False:
            with tf.GradientTape() as tape:
                y_pred = self.actor(_states)
                loss = pass_through_loss(_advantage, y_pred)
                grads = tape.gradient(loss, self.actor.trainable_variables)
                print(grads)

        self.critic.fit(np.array(_states), _v_targets, batch_size=len(_states), callbacks=[self.loss_history_critic])
        self.actor.fit(_states, _actor_loss, batch_size=len(_states), callbacks=[self.loss_history_actor])

        self.action_history.clear()
        self.states_history.clear()
        self.adjusted_trajectory_history.clear()


    def _update_actor_and_critic_old(self, reward):
        gamma = 0.99
        _states = self.states_history[-1]
        _actions = self.action_history[-1]
        _states_reassigned = []
        _target_values = []
        _advantage_vectors = []

        if len(_states) != len(_actions):
            print('Dimensional mismatch between states and trajectory pulled from history')
            return 1
        
        for _t in range(len(_states)):
            # target value calculation
            if _t == (len(_states) - 1):
                v_target = [reward]
                #v_target = self._reshape_vector(v_target)
            else:
                _next_state = self._reshape_vector(_states[_t+1])
                v_target = reward + gamma * self.critic.predict(_next_state)[0]

            _target_values.append(v_target)

            # append state
            _state = _states[_t]
            _states_reassigned.append(_state)
            _state = self._reshape_vector(_state)

            # Compute advantage for actor update
            _advantage = self.critic.predict(_state)[0] - v_target[0] # advantage acts as loss function here
            print(f'Advantage: {_advantage}\tprediction: {self.critic.predict(_state)[0]}\tv_target: {v_target[0]}')
            _advantage_vector = [_advantage[0], _advantage[0]]
            print(f'Advantage vector: {_advantage_vector}')
            #advantage_vector = tf.fill(tf.shape(_offsets[_t]), advantage[0])
            _advantage_vectors.append(_advantage_vector)

            # Train actor using advantage

        _states_reassigned = np.array(_states_reassigned)
        _target_values = np.array(_target_values)
        _advantage_vectors = np.array(_advantage_vectors)
        
        print(f'States: {_states_reassigned}')
        print(f'target value: {_target_values}')
        print(f'advantage vectors: {_advantage_vectors}')
        self.critic.fit(_states_reassigned, _target_values, batch_size=10, callbacks=[self.loss_history_critic])
        self.actor.fit(_states_reassigned, _advantage_vectors, batch_size=10, callbacks=[self.loss_history_actor])

    def _reshape_vector(self, state):
        _state = np.array(state)
        _state = _state.reshape(1, -1)
        return _state

    #####################################################################
    # MODEL CREATION, SAVING and LOADING
    #####################################################################

    def new_model(self, input_size=INPUT_DIM, output_size=ACTION_DIM, hidden_layer_size=HIDDEN_LAYER_DIM_ACTOR):
        # create critic
        _initializer_critic = 'random_normal'
        _inputs_critic = keras.layers.Input(shape=(input_size,))
        _hidden_1_critic = keras.layers.Dense(hidden_layer_size, activation="sigmoid", kernel_initializer=_initializer_critic)
        _hidden_2_critic = keras.layers.Dense(hidden_layer_size, activation="relu", kernel_initializer=_initializer_critic)
        _hidden_3_critic = keras.layers.Dense(hidden_layer_size, activation="relu", kernel_initializer=_initializer_critic)
        _output_critic = keras.layers.Dense(1, activation='relu', kernel_initializer=_initializer_critic)
        self.critic = Sequential([_inputs_critic, _hidden_1_critic, _hidden_2_critic, _hidden_3_critic, _output_critic])

        # create actor
        _inputs_actor = keras.layers.Input(shape=(input_size,))
        _hidden_1_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _hidden_2_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _hidden_3_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _hidden_4_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _output_actor = keras.layers.Dense(output_size, activation='tanh')
        self.actor = Sequential([_inputs_actor, _hidden_1_actor, _hidden_2_actor, _hidden_3_actor, _hidden_4_actor, _output_actor])

        # compile
        _optimizer_critic = keras.optimizers.Adam(learning_rate=0.001)
        _optimizer_actor = keras.optimizers.Adam(learning_rate=0.00001)
        _loss_critic = keras.losses.MeanSquaredError()
        _loss_actor = pass_through_loss

        self.actor.compile(optimizer=_optimizer_actor, loss=_loss_actor, metrics=['accuracy'])
        self.critic.compile(optimizer=_optimizer_critic, loss=_loss_critic, metrics=['accuracy'])

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
