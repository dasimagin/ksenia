from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'reward'))

class ReplayMemory(object):
    def __init__(self, config):
        self.capacity = config.memory_net.capacity
        self.gamma = config.memory_net.gamma
        self.memory = []
        self.episode = []
        self.position = 0

    def reset(self):
        self.memory = []
        self.episode = []
        self.position = 0

    def episode_save(self, state):
        self.episode.append(state)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def episode_push(self, reward):
        discouted_reward = reward
        for item in self.episode[::-1]:
            self.push(item, discouted_reward)
            discouted_reward *= self.gamma
        self.episode = []

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train_memory_critic(critic, memory, critic_loss, critic_optimizer):
    pass
