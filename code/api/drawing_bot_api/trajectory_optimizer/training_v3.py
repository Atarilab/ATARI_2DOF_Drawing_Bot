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

class LossHistory(keras.callbacks.Callback):
    def __init__(self, type='critic'):
        self.losses = []
        self.buffer = []
        self.call_counter = 0
        self.type = type

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
"""
        if self.type == 'critic':
            self._save_loss_critic(logs=logs)
        elif self.type == 'actor':
            self._save_loss_critic(logs=logs)
        else:
            print(f'type not found.')
    
    def _save_loss_critic(self, logs={}):
        self.call_counter += 1
        self.buffer.append(logs.get('loss'))

        if self.call_counter % 28 == 0:
            self.losses.append(np.mean(self.buffer))
            self.buffer.clear()

    def _save_loss_actor(self, logs={}):
        self.call_counter += 1
        self.losses.append(logs.get('loss'))
"""
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
    #tf.print("y_true:", y_true)
    return ops.mean(ops.abs(y_true * y_pred), axis=-1)

def weighted_MSE(y_true, y_pred):
    return ops.mean(1 * ops.square(y_true - y_pred))

class Trainer:
    actor = None
    critic = None
    
    replay_buffers = ReplayBuffer()
    states_history = []
    action_history = []
    reward_history = []
    critic_mean_history = []
    critic_var_history = []
    critic_min_history = []
    critic_max_history = []
    adjusted_trajectory_history = []
    image_processor = ImageProcessor()
    log = Log(verbose=VERBOSE)
    loss_history_actor = LossHistory(type='actor')
    loss_history_critic = LossHistory(type='critic')
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
        np.set_printoptions(threshold=np.inf)
        _phases = self._points_to_phases(trajectory)#                        # turn two dimensional points of trajectory into phases (one dimensional)
        #print(f'Phases: {_phases}')
        _states = np.array(self._get_states(_phases))                       # batch phases together into sequences of phases with dimension input_size
        #print(f'States: {_states}')
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
            _new_point_x = trajectory[_point_index][0] + OUTPUT_SCALING * offsets[_point_index-_index_offset][0]
            _new_point_y = trajectory[_point_index][1] + OUTPUT_SCALING * offsets[_point_index-_index_offset][1]
            _new_trajectory.append([_new_point_x, _new_point_y])

        return _new_trajectory

    def _get_offsets(self, _states, exploration_factor):
        _offsets = self.actor.predict(_states, batch_size=len(_states), verbose=0)

        if exploration_factor:
            #_offsets = np.zeros(np.shape(_offsets))
            
            # exchange some offsets with a random offset with probability exploration factor
            for _offset_index in range(len(_offsets)):
                if np.random.random() < exploration_factor:
                    _offsets[_offset_index] = (np.random.random(2) * 2 ) - 1
        
        return _offsets
    
    def _get_state(self, phases, index):
            _state = []
            for _offset_from_current_point in range(-(INPUT_DIM - NUM_LEADING_POINTS), NUM_LEADING_POINTS):
                try:
                    _state.append(phases[index+_offset_from_current_point])
                except:
                    _state.append(0)
            return _state

    def _get_states_old(self, phases):
        _phases = phases

        if USE_PHASE_DIFFERENCE:
            _phases = self._get_phase_difference(phases)
        
        self.log(f'Phase (-difference): {_phases}')
        
        _states = []

        for _current_point_index in range(len(_phases)):
            _states.append(self._get_state(_phases, _current_point_index))
        return _states
    
    def _get_states(self, phases):
        _phases = phases
        _normalize_factor = 2 * pi
        if USE_PHASE_DIFFERENCE:
            _phases = np.array(self._get_phase_difference(phases))
            _normalize_factor = pi
        if NORMALIZE_STATES:
            _phases = np.array(_phases) / _normalize_factor

        _phases = np.append(np.zeros(INPUT_DIM - NUM_LEADING_POINTS), _phases)
        window_size = INPUT_DIM
        _states = np.array([_phases[i:i + window_size] for i in range(len(_phases) - window_size + 1)])
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
        _adjusted_phases = np.zeros([INPUT_DIM-NUM_LEADING_POINTS])
        _adjusted_phases = np.append(_adjusted_phases, adjusted_phases)
        _new_states = np.copy(states)

        _normalize_factor = 2 * pi
        if USE_PHASE_DIFFERENCE:
            _adjusted_phases = self._get_phase_difference(_adjusted_phases)
            _normalize_factor = pi
        if NORMALIZE_STATES:
            _adjusted_phases = np.array(_adjusted_phases) / _normalize_factor
        
        if len(states) != len(_adjusted_phases):
            self.log(f'def _get_adjusted_states(): "Length of states ({len(states)}) and adjusted phases ({len(_adjusted_phases)}) does not match"')

        for _index in range(len(states)):
            _new_states[_index][:INPUT_DIM-NUM_LEADING_POINTS] = _adjusted_phases[_index:_index+INPUT_DIM-NUM_LEADING_POINTS]
        
        return _new_states
    
    def _normalize_to_range_incl_neg(self, data):
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    
    def _normalize_to_range_pos(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    #####################################################################
    # TRAINING METHODS
    #####################################################################

    def train(self, reward, train_actor=True):
        return self._update_actor_and_critic(reward, train_actor)

    def _update_actor_and_critic(self, reward, train_actor):
        gamma = 0.7
        _states = self.states_history[-1]
        _actions = self.action_history[-1]

        # generate states where the offset is applied to (and only to) the center point, therefore representing clear state transistions
        _adjusted_trajectory = self.adjusted_trajectory_history[-1]
        _adjusted_phases = self._points_to_phases(_adjusted_trajectory)
        _adjusted_states = self._get_adjusted_states(_states, _adjusted_phases)

        if COMBINE_STATES_FOR_CRITIC:
            _adjusted_states = np.hstack((_adjusted_states, _states))
            _num_of_states = np.shape(_adjusted_states)[0]
            _progress_indicators = np.arange(_num_of_states)
            _progress_indicators = _progress_indicators / _num_of_states

            if ADD_PROGRESS_INDICATOR:
                _progress_indicators = _progress_indicators.reshape(-1 ,1)
                _adjusted_states = np.hstack((_adjusted_states, _progress_indicators))

            #print(f'Stacked states: {_adjusted_states}')

        #_critic_predictions_without_actions = self.critic.predict(np.array(_states), verbose=0)
        _critic_predictions_with_actions = self.critic.predict(np.array(_adjusted_states), verbose=0)

        # CRITIC ########

        # calc value targets
        #_v_targets = np.full_like(_critic_predictions_with_actions, reward)
        _reward = np.array(reward).reshape(-1, 1)

        _v_targets = gamma * _critic_predictions_with_actions
        _v_targets = _v_targets[1:]
        #print(f'v targets: {_v_targets}')
        #print(f'rewards: {_reward}')
        _v_targets = _reward[:len(_v_targets)] + _v_targets
        _v_targets = np.append(_v_targets, np.mean(_reward))
        _v_targets = _v_targets.reshape(-1, 1)

        _critic_mean = np.mean(_critic_predictions_with_actions)
        self.critic_mean_history.append(_critic_mean)
        _critic_var = np.var(_critic_predictions_with_actions)
        self.critic_var_history.append(_critic_var)
        _critic_min = np.min(_critic_predictions_with_actions)
        self.critic_min_history.append(_critic_min)
        _critic_max = np.max(_critic_predictions_with_actions)
        self.critic_max_history.append(_critic_max)

        print(f'Critic | mean: {_critic_mean}\tvar: {_critic_var}\tmin: {_critic_min}\tmax: {_critic_max}')

        #print(f'_v_targets: {_v_targets}, length states: {len(_states)}, length _v_targets: {len(_v_targets)}')
        #print(f'_advantages: {_advantages}, length of advantages: {len(_advantages)}')
        if False:
            with tf.GradientTape() as tape:
                y_pred = self.actor(_states)
                loss = pass_through_loss(_advantage, y_pred)
                grads = tape.gradient(loss, self.actor.trainable_variables)
                print(grads)

        self.critic.fit(np.array(_adjusted_states), _v_targets, batch_size=16, callbacks=[self.loss_history_critic])
        
        # ACTOR #########

        # calc advantage
        #_advantage = _v_targets - _critic_predictions_with_actions
        _advantage = _critic_predictions_with_actions
        _advantage = self._normalize_to_range_pos(_advantage)
        _actor_loss = 1-_advantage #(1-np.abs(_advantage)) * np.sign(_advantage)
        #_actor_loss = self._normalize_to_range(_actor_loss)
        _actor_loss = np.repeat(_actor_loss, 2, axis=1)

        if train_actor:
            self.actor.fit(_states, _actor_loss, batch_size=16, callbacks=[self.loss_history_actor])

        self.action_history.clear()
        self.states_history.clear()
        self.adjusted_trajectory_history.clear()

        #return np.abs(self._normalize_to_range_incl_neg(_critic_predictions_with_actions).T)[0]
        return np.abs(_actor_loss.T)[0]


    def _update_actor_and_critic_old(self, reward):
        gamma = 0.9
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

    def new_model(self, input_size=INPUT_DIM, output_size=ACTION_DIM, hidden_layer_size=HIDDEN_LAYER_DIM):
        # create critic

        _critic_input_size = input_size
        if COMBINE_STATES_FOR_CRITIC:
            _critic_input_size = _critic_input_size * 2
        if ADD_PROGRESS_INDICATOR:
            _critic_input_size += 1

        _inputs_critic = keras.layers.Input(shape=(_critic_input_size,))
        #_hidden_1_critic = keras.layers.LSTM(128, return_sequences=True)
        _hidden_2_critic = keras.layers.Dense(hidden_layer_size, activation="relu")
        _hidden_3_critic = keras.layers.Dense(128, activation="relu")
        _hidden_4_critic = keras.layers.Dense(hidden_layer_size, activation="relu")
        _output_critic = keras.layers.Dense(1)#, activation='sigmoid')
        self.critic = Sequential([_inputs_critic, _hidden_2_critic, _hidden_3_critic, _hidden_4_critic, _output_critic])

        # create actor
        _inputs_actor = keras.layers.Input(shape=(input_size,))
        _hidden_1_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _hidden_2_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _hidden_3_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        #_hidden_4_actor = keras.layers.Dense(hidden_layer_size, activation="relu")
        _output_actor = keras.layers.Dense(output_size, activation='tanh', kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1))
        self.actor = Sequential([_inputs_actor, _hidden_1_actor, _hidden_2_actor, _hidden_3_actor, _output_actor])

        # compile
        _optimizer_critic = keras.optimizers.Adam(learning_rate=0.0001)
        _optimizer_actor = keras.optimizers.Adam(learning_rate=0.000001)
        _loss_critic = keras.losses.MeanSquaredError() #weighted_MSE
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
