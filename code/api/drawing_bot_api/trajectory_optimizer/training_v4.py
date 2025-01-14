from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Lambda
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

class Scheduler:
    def __init__(self, base_value, gamma):
        self.base_value = base_value
        self.gamma = gamma
        self.call_counter = 0

    def __call__(self, count_up=True):
        _value = self.base_value * pow(self.gamma, self.call_counter)
        if count_up:
            self.call_counter += 1
        return _value

############################################################
# CUSTOM LOSSES ############################################
############################################################

def pass_through_loss(y_true, y_pred):
    #tf.print("y_true:", y_true)
    return ops.mean(ops.abs(y_true * y_pred), axis=-1)

def entropy_loss(y_true, y_pred):
    entropy = -tf.reduce_sum(y_pred * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)), axis=-1)
    advantage_loss = ops.mean(ops.abs(y_true * y_pred), axis=-1)
    return advantage_loss - 0.01 * entropy  # Add entropy regularization

def actor_loss(y_true, y_pred):
   # y_true contains [actions, advantages]
    actions = y_true[:, :2]  # Extract actions
    advantages = y_true[:, 2:]  # Extract advantages

    # y_pred is a list of outputs: [means, sigmas]
    means = y_pred[:, :2]
    sigmas = y_pred[:, 2:]

    sigmas = (sigmas * SIGMA_SCALING) + SIGMA_LIMIT_MIN

    # Compute Gaussian log-probabilities
    log_probs = -0.5 * ops.sum(((actions - means) / (sigmas + 1e-8))**2 + 2 * ops.log(sigmas + 1e-8) + ops.log(2 * np.pi), axis=1)

    # Scale log-probabilities by advantages
    loss = -log_probs * advantages
    return ops.mean(loss)

def weighted_MSE(y_true, y_pred):
    _weight = 1
    return ops.mean(_weight * ops.square(y_true - y_pred))

############################################################
# TRAINER ##################################################
############################################################

class Trainer:
    actor = None
    critic = None

    sigma_scheduler = Scheduler(SIGMA, SIGMA_DECAY)
    
    # training histories
    states_history = []
    action_history = []
    reward_history = []
    adjusted_trajectory_history = []

    # utils and logs
    image_processor = ImageProcessor()
    log = Log(verbose=VERBOSE)
    critic_mean_log = []
    critic_var_log = []
    critic_min_log = []
    critic_max_log = []
    action_mean_log = []
    action_max_log = []
    action_min_log = []
    actor_output = []
    loss_actor_log = LossHistory(type='actor')
    loss_critic_log = LossHistory(type='critic')

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
        self.action_mean_log.append(np.mean(_offsets))
        self.action_max_log.append(np.max(_offsets))
        self.action_min_log.append(np.min(_offsets))
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

    def _get_offsets(self, _states, random_action_probability):
        _actor_output = np.array(self.actor.predict(_states, batch_size=len(_states), verbose=0)).T
        self.actor_output = _actor_output
        _mu1, _sigma1, _mu2, _sigma2 = _actor_output
        _sigma1 = (_sigma1 * SIGMA_SCALING) + SIGMA_LIMIT_MIN
        _sigma2 = (_sigma2 * SIGMA_SCALING) + SIGMA_LIMIT_MIN

        _offset_x = np.random.normal(loc=_mu1, scale=np.abs(_sigma1))
        _offset_y = np.random.normal(loc=_mu2, scale=np.abs(_sigma2))
        '''
        if random_action_probability:
            #_offsets = np.zeros(np.shape(_offsets))
            
            # exchange some offsets with a random offset with probability exploration factor
            for _offset_index in range(len(_offset_x)):
                if np.random.random() < random_action_probability:
                    _offset_x[_offset_index] = np.random.normal(loc=0, scale=RANDOM_ACTION_SCALE)
        '''
        return [_offset_x, _offset_y]
    
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
        gamma = 0.9
        _states = self.states_history[-1]
        _actions = self.action_history[-1]

        # generate states where the offset is applied to (and only to) the center point, therefore representing clear state transistions
        _adjusted_trajectory = self.adjusted_trajectory_history[-1]
        _adjusted_phases = self._points_to_phases(_adjusted_trajectory)
        _adjusted_states = self._get_adjusted_states(_states, _adjusted_phases)

        if COMBINE_STATES_FOR_CRITIC:
            _adjusted_states = np.hstack((_adjusted_states, _states))

            if ADD_PROGRESS_INDICATOR:
                _num_of_states = np.shape(_adjusted_states)[0]
                _progress_indicators = np.arange(_num_of_states)
                _progress_indicators = _progress_indicators / _num_of_states
                _progress_indicators = _progress_indicators.reshape(-1 ,1)
                _adjusted_states = np.hstack((_adjusted_states, _progress_indicators))

            #print(f'Stacked states: {_adjusted_states}')

        #_critic_predictions_without_actions = self.critic.predict(np.array(_states), verbose=0)
        _critic_predictions_with_actions = self.critic.predict(np.array(_adjusted_states), verbose=0)

        # CRITIC ########

        #_v_targets = np.full_like(_critic_predictions_with_actions, reward)
        _reward = []
        if SPARSE_REWARDS:
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

        _v_targets = gamma * _critic_predictions_with_actions
        _v_targets = _v_targets[1:]
        #print(f'rewards: {_reward}')
        _v_targets = _reward[:len(_v_targets)] + _v_targets
        _v_targets = np.append(_v_targets, np.mean(_reward))
        _v_targets = _v_targets.reshape(-1, 1)
        #print(f'v targets: {_v_targets}')

        _critic_mean = np.mean(_critic_predictions_with_actions)
        self.critic_mean_log.append(_critic_mean)
        _critic_var = np.var(_critic_predictions_with_actions)
        self.critic_var_log.append(_critic_var)
        _critic_min = np.min(_critic_predictions_with_actions)
        self.critic_min_log.append(_critic_min)
        _critic_max = np.max(_critic_predictions_with_actions)
        self.critic_max_log.append(_critic_max)

        print(f'Critic | mean: {_critic_mean}\tvar: {_critic_var}\tmin: {_critic_min}\tmax: {_critic_max}')

        #print(f'_v_targets: {_v_targets}, length states: {len(_states)}, length _v_targets: {len(_v_targets)}')
        #print(f'_advantages: {_advantages}, length of advantages: {len(_advantages)}')

        #_adjusted_states = (np.array(_adjusted_states) + 1) / 2
        self.critic.fit(_adjusted_states, _v_targets, batch_size=64, callbacks=[self.loss_critic_log])
        
        # ACTOR #########

        _actions = np.array(_actions)
        _actions = _actions.squeeze().T

        # calc advantage
        _advantage = _v_targets - _critic_predictions_with_actions
        #_advantage = reward.reshape(-1, 1)
        _advantage = _advantage[:len(_actions)]
        _advantage = self._normalize_advantage(_advantage)
        #_advantage = self._normalize_to_range_pos(_advantage)
        #_advantage = np.repeat(_advantage, 2, axis=1)

        # actor ytrue vector
        _actor_ytrue = tf.concat([_actions, _advantage], axis=1)

        """ _noise = np.random.normal(loc=0.0, scale=0.1, size=_states.shape)
        _states = _states + _noise
        _states = _states * 0.1 # scale input states to make network more sensitive to small changes """

        if train_actor:
            self.actor.fit(_states, _actor_ytrue, batch_size=64, callbacks=[self.loss_actor_log])

        self.action_history.clear()
        self.states_history.clear()
        self.adjusted_trajectory_history.clear()

        #return np.abs(self._normalize_to_range_incl_neg(_critic_predictions_with_actions).T)[0]
        return np.array(_advantage).T[0], self.actor_output

    def _reshape_vector(self, state):
        _state = np.array(state)
        _state = _state.reshape(1, -1)
        return _state
    
    def _normalize_advantage(self, advantage):
        return (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

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
        _hidden_1_critic = keras.layers.Dense(hidden_layer_size, activation="relu")(_inputs_critic)
        _hidden_2_critic = keras.layers.Dense(hidden_layer_size, activation="relu")(_hidden_1_critic)
        _output_critic = keras.layers.Dense(1, activation='linear')(_hidden_2_critic)
        self.critic = Model(inputs=_inputs_critic, outputs=_output_critic)

        _inputs_actor = keras.layers.Input(shape=(input_size,))
        _hidden_1_actor = keras.layers.Dense(hidden_layer_size, activation="relu")(_inputs_actor)
        _hidden_2_actor = keras.layers.Dense(hidden_layer_size, activation="relu")(_hidden_1_actor)
        _output_mu1 = keras.layers.Dense(1, activation='tanh', name='mu1')(_hidden_2_actor)
        _output_sigma1 = keras.layers.Dense(1, activation='softplus', name='sigma1')(_hidden_2_actor)
        _output_mu2 = keras.layers.Dense(1, activation='tanh', name='mu2')(_hidden_2_actor)
        _output_sigma2 = keras.layers.Dense(1, activation='softplus', name='sigma2')(_hidden_2_actor)
        merged_output = Lambda(lambda x: tf.concat(x, axis=-1), name="merged_output")([_output_mu1, _output_sigma1, _output_mu2, _output_sigma2])
        self.actor = Model(inputs=_inputs_actor, outputs=merged_output)

        # compile
        _optimizer_critic = keras.optimizers.Adam(learning_rate=LR_CRITIC)
        _optimizer_actor = keras.optimizers.Adam(learning_rate=LR_ACTOR)
        _loss_critic = keras.losses.MeanSquaredError() #weighted_MSE
        _loss_actor = actor_loss

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
