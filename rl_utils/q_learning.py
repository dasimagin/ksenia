import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def q_learning(state_values, action_batch, reward_batch, next_state_values, gamma=0.97, alpha=0.1):
    expected_state_action_values = (np.amax(next_state_values) * gamma) + reward_batch
    true_state_values = state_values.copy()
    true_state_values[action_batch] = (1 - alpha) * state_values + alpha * expected_state_action_values
    return true_state_values

def mse_and_penalty(input, target):
    pass
