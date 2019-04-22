from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def reverse_sample(self):
        return self.memory[::-1]

    def __len__(self):
        return len(self.memory)


def save_episode(
         memory,
         curricua,
         env,
         model,
         device
    ):
    task_len = curricua.sample()
    temp_env = env.reset(task_len)
    model.init_sequence(1, device)
    while not temp_env.finished:
        readed = torch.eye(temp_env.len_alphabet)[temp_env.read()]
        action_probas = model.step(readed)
        reward = temp_env.step(action_probas.argmax())
        new_readed = torch.eye(temp_env.len_alphabet)[temp_env.read()] if not temp_env.finished else None
        # print(action_probas.argmax().view(1, -1).shape)
        memory.push(readed, action_probas.argmax().view(1, -1), new_readed, reward)
    return temp_env.episode_total_reward
