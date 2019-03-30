"""Datasets for different tasks with bit vectors.
Contains torch datasets for copy, repeat-copy, associative-recall
"""

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F


class BitBatchSampler():
    """Generates indexes equal to randomly chosen seq_len for each batch.
    """
    def __init__(self, batch_size, min_len=1, max_len=20, min_rep=1, max_rep=1, length=None, seed=5):
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.min_rep = min_rep
        self.max_rep = max_rep
        self.length = length
        self.rand = np.random.RandomState(seed)

    def __iter__(self):
        while True:
            seq_len = self.rand.randint(self.min_len, self.max_len + 1)
            num_rep = self.rand.randint(self.min_rep, self.max_rep + 1)

            if self.max_rep == 1:                           # copy task
                yield [seq_len] * self.batch_size
            else:                                           # repeat copy task
                yield [(seq_len, num_rep)] * self.batch_size

    def __len__(self):
        if self.length:
            return self.length
        return 0x7FFFFFFF


class CopyTask(torch.utils.data.Dataset):
    """Dataset for generating copy and repeat copy examples.
    Uses hack with batch sampler to get same length samples in mini-batch.
    """
    def __init__(self, bit_width=8, seed=1):
        self.bit_width = bit_width
        self.rand = np.random.RandomState(seed)

    def __len__(self):
        return 0x7FFFFFFF

    def __getitem__(self, seq_len):
        seq = self.rand.binomial(1, 0.5, size=(seq_len, self.bit_width)).astype(np.float32)

        # extra channel for delimeter
        inp = np.zeros((2 * seq_len + 1, self.bit_width + 1))
        inp[:seq_len, :self.bit_width] = seq
        inp[seq_len, self.bit_width] = 1.0

        out = np.zeros((2 * seq_len + 1, self.bit_width))
        out[seq_len + 1:, :self.bit_width] = seq

        return inp.astype(np.float32), out.astype(np.float32)

    def loss(self, prediction, target):
        return F.binary_cross_entropy(prediction, target, reduction='mean')


class RepeatCopyTask(torch.utils.data.Dataset):
    def __init__(self, bit_width=8, seed=1):
        self.bit_width = bit_width
        self.rand = np.random.RandomState(seed)

    def __len__(self):
        return 0x7FFFFFFF

    def __getitem__(self, key):
        seq_len, num_rep = key
        seq = self.rand.binomial(1, 0.5, size=(seq_len, self.bit_width)).astype(np.float32)
        actual_len = seq_len * (num_rep + 1) + 2    # two extra vectors (num_rep and ending)

        inp = np.zeros((actual_len, self.bit_width + 1))
        inp[:seq_len, :self.bit_width] = seq
        inp[seq_len, self.bit_width] = float(num_rep)

        out = np.zeros((actual_len, self.bit_width + 1))
        out[-1, -1] = 1.0   # ending marker
        out[seq_len + 1:-1, :self.bit_width] = np.tile(seq, (num_rep, 1))

        return inp, out

    def loss(self, prediction, target):
        return F.binary_cross_entropy(prediction, target, reduction='mean')


class AssociativeRecall(torch.utils.data.Dataset):
    # TODO
    pass


if __name__ == "__main__":
    copy_sampler = BitBatchSampler(batch_size=10, max_len=2, min_rep=1, max_rep=1, seed=1)
    repeat_sampler = BitBatchSampler(batch_size=10, max_len=2, min_rep=2, max_rep=2, seed=1)

    copy_loader = torch.utils.data.DataLoader(CopyTask(seed=1), batch_sampler=copy_sampler)
    repeat_loader = torch.utils.data.DataLoader(RepeatCopyTask(seed=1), batch_sampler=repeat_sampler)

    print('Copy')
    inp, out = next(iter(copy_loader))
    print('Input shape:', inp.shape)
    print('Output shape:', out.shape)
    print('Input:')
    print(inp)
    print('Output:')
    print(out)

    print('Repeat copy')
    inp, out = next(iter(repeat_loader))
    print('Input shape:', inp.shape)
    print('Output shape:', out.shape)
    print('Input:')
    print(inp)
    print('Output:')
    print(out)
