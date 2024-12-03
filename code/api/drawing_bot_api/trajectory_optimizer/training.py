from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
from drawing_bot_api.trajectory_optimizer.config import *
from drawing_bot_api.config import PLOT_XLIM, PLOT_YLIM
import os
from drawing_bot_api.trajectory_optimizer.image_processor import ImageProcessor
from drawing_bot_api.logger import Log

# INPUT SPACE: 5 left angles + 5 right angles = 10
# ACTION SPACE: 1 left angle + 1 right angle = 2

#/Users/leon/.plaidml
    
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
    image_processor = ImageProcessor()
    log = Log(verbose=1)

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

    def adjust_trajectory(self, trajectory):
        # trajectory has the form: [[x1, y1], [x2, y2], ... , [xn, yn]]

        self._normalize_trajectory(trajectory)
        _batched_inputs = []

        # iterating over all points
        for _index in range(len(trajectory)):

            _input = []
            
            # iterating over the last 10 points
            for _pointer in range(_index-int(INPUT_DIM/2)+1, _index+1):
                if _pointer < 0:
                    _input.extend([0, 0])
                else:
                    _input.extend(trajectory[_pointer])
    
            _batched_inputs.append(_input)

        _batched_inputs = np.array(_batched_inputs)
        _offsets = self.model.predict(_batched_inputs, batch_size=1)
        
        return _offsets
    
    '''
    def _adjust_trajectory(self, trajectory):
        self._normalize_trajectory
        trajectory = np.array(trajectory)
        trajectory = trajectory.reshape(-1)
        print(trajectory.shape)

        _input = np.append(trajectory, np.zeros((2000-len(trajectory), 1)))
        print(_input.shape)
        _input = _input.reshape(1, -1)
        _new_trajectory = self.model.predict(_input)
        _new_trajectory.reshape(-1, 2)
        return _new_trajectory
    '''

    def train(self, replay_buffer, reward):
        # Calculates reward and ajusts model based on data in replay buffer
        # Clears cache in the end
        # Update Actor
        _states = []
        _rewards = []

        for step in replay_buffer.buffer:
            _states.append(step.state)
            _rewards.append(step.action * reward)
        
        self.model.fit(_states, _rewards, batchsize=len(replay_buffer.buffer))

    def new_model(self, input_size=INPUT_DIM, output_size=ACTION_DIM, hidden_layer_size=HIDDEN_LAYER_DIM):
        hidden_layer = Dense(hidden_layer_size, activation='relu', input_shape=(input_size,))
        output_layer = Dense(ACTION_DIM, activation='tanh') 

        model = Sequential([hidden_layer, output_layer])
        model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
