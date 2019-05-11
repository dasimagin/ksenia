import numpy as np
from gym.utils import colorize

def arr_to_str(arr):
    return ' '.join(arr)

class Curriculum(object):
    """Curriculum as in """
    def __init__(self, config):
        super(Curriculum, self).__init__()
        self.rand = np.random.RandomState(config.seed)
        self.max_rep = config.task.max_len
        self.temp_size = config.task.min_len
        self.update_limit = config.task.update_limit


    def update(self, config, val_acc):
        config.iteration += 1
        if self.temp_size <= self.max_rep and val_acc >= self.update_limit and config.iteration > config.task.min_update_iters:
            self.temp_size += 1
            config.q_learning.eps_start = config.q_learning.eps_restart
            config.iteration = 0
        return config

    def sample(self):
        type = self.rand.choice(3, p = [0.1, 0.25, 0.65])
        if type == 0:
            return self.rand.choice(self.max_rep - 1) + 1
        e = np.array(np.around(self.rand.geometric(p=0.5, size=1)), dtype='int')
        if type == 1:
            return self.rand.choice(max(1, self.temp_size - e), self.temp_size + e)
        if type == 2:
            return self.rand.choice(e) + 1


class CopyEnv(object):
    """docstring for CopyEnv."""
    def __init__(self, n_copies=1, len_input_seq=None, len_alphabet=2, seed=None):
        self.len_input_seq = len_input_seq
        self.n_copies = n_copies
        self.len_alphabet = len_alphabet
        if seed is None:
            self.random = np.random
        else:
            self.random = np.random.RandomState(seed)
        self.reset()

    def reset(self, len_input_seq=None):
        self.input_panel = self._create_input(len_input_seq if len_input_seq is not None else self.len_input_seq)
        self.input_place = 0
        self.output_place = 0
        self.time = 0
        self.time_limit = 3 * len(self.input_panel)
        self.true_output = np.array(list(self.input_panel.flatten()) * self.n_copies)
        self.output_panel = np.full_like(self.true_output, -1)
        self.finished = False
        self.action_space = np.arange(self.len_alphabet + 2)
        self.episode_total_reward = 0
        return self

    def _create_input(self, len_input_seq, **params):
        self.len_input_seq = self.random.choice(20) if len_input_seq is None else len_input_seq
        return self.random.randint(self.len_alphabet, size=self.len_input_seq)

    def read(self):
        if not self.finished:
            return self.input_panel[self.input_place]
        else:
            return -1

    def step(self, action):
        action = int(action)
        if action not in self.action_space:
            raise ValueError('Vy mem')
        self.time += 1
        if self.time > self.time_limit:
            self.finished = True
            self.episode_total_reward -= 1
            return -1
        if action > 1:
            # write zero to output
            writed_symbol = action - 2
            self.output_panel[self.output_place] = writed_symbol
            value = 1 if self.output_panel[self.output_place] == self.true_output[self.output_place] else -0.5
            self.output_place += 1
            if self.output_place >= len(self.output_panel):
                self.finished = True
            self.episode_total_reward += value
            return value
        elif action == 0:
            # go left
            self.input_place -= 1
            self.input_place %= len(self.input_panel)
            return 0
        elif action == 1:
            # go right
            self.input_place += 1
            self.input_place %= len(self.input_panel)
            return 0
        else:
            raise ValueError('Unknown action')

    def render(self, **print_params):
        print(f'Input:   {arr_to_str(self.input_panel[:self.input_place])} {colorize(self.input_panel[self.input_place], "green", highlight=True)} {arr_to_str(self.input_panel[self.input_place + 1:])}',  **print_params)
        print(f'Output:   {arr_to_str(self.output_panel[:self.output_place])} {colorize(self.output_panel[self.output_place], "green", highlight=True)} {arr_to_str(self.output_panel[self.output_place + 1:])}', **print_params)
        print(f'Total reward: {self.episode_total_reward}')

class SortEnv(object):
    """docstring for SortEnv."""
    def __init__(self, len_input_seq=None, len_alphabet=2, seed=None):
        self.len_input_seq = len_input_seq
        self.len_alphabet = len_alphabet
        if seed is None:
            self.random = np.random
        else:
            self.random = np.random.RandomState(seed)
        self.reset()

    def reset(self, len_input_seq=None):
        self.input_panel = self._create_input(len_input_seq if len_input_seq is not None else self.len_input_seq)
        self.input_place = 0
        self.output_place = 0
        self.time = 0
        self.time_limit = 3 * len(self.input_panel)
        self.true_output = self.input_panel[self.input_panel.argsort()]
        self.output_panel = np.full_like(self.true_output, -1)
        self.finished = False
        self.action_space = np.arange(self.len_alphabet + 2)
        self.episode_total_reward = 0
        return self

    def _create_input(self, len_input_seq, **params):
        self.len_input_seq = self.random.choice(20) if len_input_seq is None else len_input_seq
        return self.random.randint(self.len_alphabet, size=self.len_input_seq)

    def read(self):
        if not self.finished:
            return self.input_panel[self.input_place]
        else:
            return -1

    def step(self, action):
        action = int(action)
        if action not in self.action_space:
            raise ValueError('Vy mem')
        self.time += 1
        if self.time > self.time_limit:
            self.finished = True
            self.episode_total_reward -= 1
            return -1
        if action > 1:
            # write zero to output
            writed_symbol = action - 2
            self.output_panel[self.output_place] = writed_symbol
            value = 1 if self.output_panel[self.output_place] == self.true_output[self.output_place] else -0.5
            self.output_place += 1
            if self.output_place >= len(self.output_panel):
                self.finished = True
            self.episode_total_reward += value
            return value
        elif action == 0:
            # go left
            self.input_place -= 1
            self.input_place %= len(self.input_panel)
            return 0
        elif action == 1:
            # go right
            self.input_place += 1
            self.input_place %= len(self.input_panel)
            return 0
        else:
            raise ValueError('Unknown action')

    def render(self, **print_params):
        print(f'Input:   {arr_to_str(self.input_panel[:self.input_place])} {colorize(self.input_panel[self.input_place], "green", highlight=True)} {arr_to_str(self.input_panel[self.input_place + 1:])}',  **print_params)
        print(f'Output:   {arr_to_str(self.output_panel[:self.output_place])} {colorize(self.output_panel[self.output_place], "green", highlight=True)} {arr_to_str(self.output_panel[self.output_place + 1:])}', **print_params)
        print(f'Total reward: {self.episode_total_reward}')
