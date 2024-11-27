from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf
import numpy as np
from config import *
import os
from image_processor import ImageProcessor
from math import tanh, log

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[idx] for idx in indices]

class Trainer:
    model = None
    replay_buffers = []
    image_processor = ImageProcessor()


    def __init__(self, model=None, **kwargs):
        if model == None:
            self.model = self.new_model(**kwargs)
        else:
            self.model = self.load_model(model)

    def get_joint_offset(self, input):
        # this is the inference function
        # Input: Trajectory
        # Output: Adjusted trajectory

        pass

    def adjust_trajectory(self, trajectory):
        # Takes the trajectory and loops the predection cycle until all points have an offset
        # Writes to buffer
        replay_buffer = ReplayBuffer(len(trajectory))
        pass

    def teach(self, template_img):
        # Calculates reward and ajusts model based on data in replay buffer
        # Clears cache in the end
        # Update Actor
        similarity = self.image_processor(template_img)
        reward = log(similarity)

        with tf.GradientTape() as tape:
            actor_grads = tape.gradient(reward, self.model.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    def new_model(self, input_size=INPUT_DIM, hidden_layer_size=HIDDEN_LAYER_DIM):
        hidden_layer = Dense(hidden_layer_size, activation='relu', input_shape=(input_size,))
        output_layer = Dense(1, activation='tanh') 

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
