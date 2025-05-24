from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import keras
import numpy as np
from drawing_bot_api.trajectory_optimizer.config_rl import *
from drawing_bot_api.config import PLOT_XLIM, PLOT_YLIM
import os
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.logger import Log
from math import sqrt, pow, atan2

# INPUT SPACE: 5 left angles + 5 right angles = 10
# ACTION SPACE: 1 left angle + 1 right angle = 2

#/Users/leon/.plaidml

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
    model = None
    replay_buffers = ReplayBuffer()
    input_buffer = None
    action_buffer = None
    image_processor = ImageProcessor()
    log = Log(verbose=1)
    loss_history = LossHistory()

    def __init__(self, model=None, **kwargs):
        if model == None:
            self.model = self.new_model(**kwargs)
        elif model == 'ignore':
            self.model = None
        else:
            self.model = self.load_model(model)

    def _normalize_trajectory(self, trajectory):
        for point in trajectory:
            point[0] = point[0] / PLOT_XLIM[1]
            point[1] = point[1] / PLOT_YLIM[1]

    def adjust_trajectory(self, trajectory, exploration_factor=0):
        # trajectory has the form: [[x1, y1], [x2, y2], ... , [xn, yn]]
        _trajectory = trajectory.copy()

        self._normalize_trajectory(_trajectory)
        _batched_inputs = []

        # iterating over all points
        for _index in range(len(_trajectory)):

            _input = []
            
            # iterating over the last 10 points
            for _pointer in range(_index-int(INPUT_DIM)+1, _index+1):
                if _pointer < 0:
                    _input.append(0)
                else:
                    _prev_point = _trajectory[_pointer-1]
                    _point = _trajectory[_pointer]
                    _input.append(self._get_phase(_point, _prev_point))
    
            _batched_inputs.append(_input)

        _batched_inputs = np.array(_batched_inputs)
        self.input_buffer = _batched_inputs

        self.model.compile()
        _offsets = self.model.predict(_batched_inputs, batch_size=len(_batched_inputs))

        for i in range(len(_offsets)):
            if np.random.randint(0, 100) > exploration_factor*100:
                _offsets[i][0] = np.random.random(1)
                _offsets[i][1] = np.random.random(1)

        print(f'Offsets: {_offsets}')

        self.action_buffer = _offsets

        # the offset is scaled to match with dimensions of drawings
        return OUTPUT_SCALING * _offsets
    
    def _get_phase(self, point, prev_point):
        _pointing_vector = [point[0]-prev_point[0], point[1]-prev_point[1]]
        _phase = atan2(_pointing_vector[1], _pointing_vector[0])
        return _phase
    
    def _get_target_vector_via_subtraction_method(self, reward):
        _actions = self.action_buffer
        _actions = _actions - np.sign(_actions) * (1 - reward)
        return _actions

    def train(self, reward):
        _target_vector = self.action_buffer * reward
        _target_vector = self._get_target_vector_via_subtraction_method(reward)
        #print(f'Action Buffer: {self.action_buffer}')
        #print(f'Target vector: {_target_vector}')
        self.model.fit(self.input_buffer, _target_vector, batch_size=len(self.input_buffer), callbacks=[self.loss_history])
        self.input_buffer = None
        self.action_buffer = None

    def new_model(self, input_size=INPUT_DIM, output_size=ACTION_DIM, hidden_layer_size=HIDDEN_LAYER_DIM_ACTOR):
                
        _kernel_initializer = 'zeros'#keras.initializers.RandomUniform(minval=-0.005, maxval=0.005, seed=None) # inititalizing weights

        _hidden_layer_1 = Dense(
            hidden_layer_size, 
            activation='linear', 
            input_shape=(input_size,), 
            kernel_initializer=_kernel_initializer,
            bias_initializer='zeros'
        )

        _hidden_layer_2 = Dense(
            hidden_layer_size, 
            activation='linear', 
            kernel_initializer=_kernel_initializer,
            bias_initializer='zeros'
        )

        _output_layer = Dense(
            output_size, 
            activation='linear', 
            kernel_initializer=_kernel_initializer,
            bias_initializer='zeros'
        ) 

        _lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.96)
        _optimizer = keras.optimizers.SGD(learning_rate=_lr_schedule)

        #_loss = keras.losses.MeanSquaredError()
        _loss = keras.losses.KLDivergence(reduction='sum_over_batch_size')


        model = Sequential([_hidden_layer_1, _hidden_layer_2, _output_layer])
        model.compile(optimizer=_optimizer, loss=_loss, metrics=['accuracy'])

        #print(f'Model summary: {model.summary()}')

        return model

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
        self.model.save(_path)
