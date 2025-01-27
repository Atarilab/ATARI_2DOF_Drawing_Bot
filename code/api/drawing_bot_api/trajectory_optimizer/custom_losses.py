from keras import ops
import tensorflow as tf
from drawing_bot_api.trajectory_optimizer.config import *
import numpy as np

def entropy_loss(y_true, y_pred):
    entropy = -tf.reduce_sum(y_pred * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)), axis=-1)
    advantage_loss = ops.mean(ops.abs(y_true * y_pred), axis=-1)
    return advantage_loss - 0.01 * entropy  # Add entropy regularization

def custom_loss(y_true, y_pred):
    """Calculates the log probabilities of taken actions and multiplies result with the advantage"""
   # y_true contains [actions, advantages]
    actions = y_true[:, :2]  # Extract actions
    advantages = y_true[:, 2:]  # Extract advantages
    advantages = ops.mean(advantages, axis=-1)

    # y_pred is a list of outputs: [means, sigmas]
    means = y_pred[:, :2]
    sigmas = y_pred[:, 2:]
    sigmas = tf.clip_by_value(sigmas, SIGMA_MIN, SIGMA_MAX)
    sigmas_mean = ops.mean(sigmas, axis=-1)

    # Compute Gaussian log-probabilities
    log_probs = -0.5 * ops.sum(((actions - means) / (sigmas + 1e-8))**2 + 2 * ops.log(sigmas + 1e-8) + ops.log(2 * np.pi), axis=1)

    # calculate entropies
    action_penalty = ops.average(ops.square(means))
    sigma_entropy = ops.sum(ops.log(sigmas + 1e-8))
    sigma_penalty = ops.average(ops.square(sigmas))
    advantage_penalty = ops.max(ops.stack((-advantages, advantages*sigmas_mean), axis=1), axis=1)

    # Scale log-probabilities by advantages
    means_loss = ops.mean(-log_probs * advantages) + ACTION_PENALTY_FACTOR * action_penalty
    sigmas_loss = advantage_penalty * ADVANTAGE_FACTOR - SIGMA_ENTROPY_FACTOR * sigma_entropy #+ SIGMA_PENALTY_FACTOR * sigma_penalty
    loss = means_loss + sigmas_loss
    loss = ops.clip(loss, -GRADIENT_CLIPPING_LIMIT, GRADIENT_CLIPPING_LIMIT)
    return loss

def weighted_MSE(y_true, y_pred):
    _weight = 1
    return ops.mean(_weight * ops.square(y_true - y_pred))