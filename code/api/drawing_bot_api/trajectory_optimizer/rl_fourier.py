from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Lambda
from keras.api.models import load_model
import keras.api.backend as K
import keras
from keras import ops
import numpy as np
import tensorflow as tf
from drawing_bot_api.trajectory_optimizer.config_fourier import *
from drawing_bot_api.config import PLOT_XLIM, PLOT_YLIM
from drawing_bot_api.trajectory_optimizer.custom_losses import *
from drawing_bot_api.trajectory_optimizer.utils import *

class RlModel:
    model = None
    optimizer = None
    loss_history = LossHistory()

    def __init__(self):
        self._get_model()

    def _normalize_trajectory(self, trajectory):
        return (trajectory - PLOT_YLIM[0]) / (PLOT_YLIM[1] - PLOT_YLIM[0])
    
    def _reshape_trajectory(self, trajectory):
        _trajectory = np.zeros((MODEL_INPUT_SIZE, 2))
        _trajectory[:len(trajectory)] = trajectory
        _trajectory = self._normalize_trajectory(_trajectory)
        _trajectory = _trajectory.reshape(-1, 1).T
        return _trajectory

    def get_parameters(self, trajectory):
        _trajectory = self._reshape_trajectory(trajectory)
        _params = self.model.predict(_trajectory, verbose=0)
        return np.squeeze(_params)

    def train(self, trajectory, params, reward, template_reward, sigma):
        _params = np.array(params)
        #_advantage = np.array(reward - template_reward)
        #_advantage = np.where(_advantage < 0, _advantage / template_reward, _advantage / (1-template_reward)) * ADVANTAGE_SCALING
        _advantage = (1-reward) * ADVANTAGE_SCALING
        _sigma = np.array(sigma)

        _ytrue = np.hstack((_params, _advantage, _sigma))
        _ytrue = _ytrue.reshape(-1, 1).T

        _trajectory = self._reshape_trajectory(trajectory)

        self.model.fit(_trajectory, _ytrue, verbose=0, callbacks=[self.loss_history])


    def _get_model(self):
        kernel_initializer = keras.initializers.RandomUniform(-KERNEL_INITIALIZER_LIMITS, KERNEL_INITIALIZER_LIMITS)
        _input = keras.layers.Input(shape=(2*MODEL_INPUT_SIZE,))
        _hidden1 = keras.layers.Dense(MODEL_HIDDEN1_SIZE, activation='relu')(_input)
        _hidden2 = keras.layers.Dense(MODEL_HIDDEN2_SIZE, activation='relu')(_hidden1)
        _output = keras.layers.Dense(NUM_OF_PARAMETERS, activation='tanh')(_hidden2)
        _output = Lambda(lambda x: MODEL_OUTPUT_SCALING * x)(_output)
        self.model = Model(inputs=_input, outputs=_output)

        _optimizer = keras.optimizers.Adam(learning_rate=LR)
        _loss = reward_loss
        
        self.model.compile(_optimizer, _loss)
        
