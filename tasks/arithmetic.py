#!/usr/bin/env python3

"""Dataloaders for arithmetic tasks.
"""

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder


class Arithmetic:
    """Data generator for arithmetic tasks.
    """
    def __init__(self, batch_size, min_len, max_len, task, seed=1):
        """task is string from {'+', '*', '+*', '*+'}"""
        self.batch_size = batch_size
        assert min_len >= 3, 'min_len can\'t be less than 3'
        self.min_len = min_len
        self.max_len = max_len
        self.task = task
        if task == '+':
            self.symbols_amount = 4
            self.ohe = OneHotEncoder(categories=[['0', '1', '+', '#']], sparse=False)
        elif task == '*':
            self.symbols_amount = 4
            self.ohe = OneHotEncoder(categories=[['0', '1', '*', '#']], sparse=False)
        self.rand = np.random.RandomState(seed)
        np.random.seed(seed)

    def gen_example(self, seq_len, max_len_batch):
        if self.task == '+':
            num1_len = np.random.randint(1, seq_len - 1)
            num2_len = seq_len - num1_len - 1
            num1 = 0
            num2 = 0
            res_num = 0
            if num1_len == 1:
                num1 = np.random.randint(0, 2)
            else:
                num1 = np.random.randint(2 ** (num1_len - 1), 2 ** num1_len)
            if num2_len == 1:
                num2 = np.random.randint(0, 2)
            else:
                num2 = np.random.randint(2 ** (num2_len - 1), 2 ** num2_len)
            res_num = num1 + num2
            seq = f'{num1:b}+{num2:b}#'
            answ = f'{res_num:b}#'
            #print(num1, num2, res_num, '|', seq, answ)

            inp = np.zeros((max_len_batch, self.symbols_amount))
            inp[:seq_len + 1] = self.ohe.fit_transform([[c] for c in seq])

            out = np.zeros((max_len_batch, self.symbols_amount))
            out[seq_len + 1:seq_len + 1 + len(answ)] = self.ohe.fit_transform([[c] for c in answ])

            mask = np.zeros(max_len_batch)
            mask[seq_len + 1:seq_len + 1 + len(answ)] = 1

            return inp.astype(np.float32), out.astype(np.float32), mask.astype(np.float32)
        elif self.task == '*':
            num1_len = np.random.randint(1, seq_len - 1)
            num2_len = seq_len - num1_len - 1
            num1 = 0
            num2 = 0
            res_num = 0
            if num1_len == 1:
                num1 = np.random.randint(0, 2)
            else:
                num1 = np.random.randint(2 ** (num1_len - 1), 2 ** num1_len)
            if num2_len == 1:
                num2 = np.random.randint(0, 2)
            else:
                num2 = np.random.randint(2 ** (num2_len - 1), 2 ** num2_len)
            res_num = num1 * num2
            seq = f'{num1:b}*{num2:b}#'
            answ = f'{res_num:b}#'
            #print(num1, num2, res_num, '|', seq, answ)

            inp = np.zeros((max_len_batch, self.symbols_amount))
            inp[:seq_len + 1] = self.ohe.fit_transform([[c] for c in seq])

            out = np.zeros((max_len_batch, self.symbols_amount))
            out[seq_len + 1:seq_len + 1 + len(answ)] = self.ohe.fit_transform([[c] for c in answ])

            mask = np.zeros(max_len_batch)
            mask[seq_len + 1:seq_len + 1 + len(answ)] = 1

            return inp.astype(np.float32), out.astype(np.float32), mask.astype(np.float32)

    def calc_batch_len(self, seq_len_batch):
        if self.task == '+' or self.task == '*':
            return seq_len_batch * 2 + 1


    def gen_batch(
            self,
            batch_size,
            min_len, max_len,
            distribution
    ):
        seq_len_batch = np.random.choice(
            np.arange(min_len, max_len + 1),
            size=batch_size, p=distribution
            )

        total_len_batch = self.calc_batch_len(seq_len_batch)
        max_len_batch = np.max(total_len_batch)

        inp = np.zeros((batch_size, max_len_batch, self.symbols_amount), dtype=np.float32)
        out = np.zeros((batch_size, max_len_batch, self.symbols_amount), dtype=np.float32)
        mask = np.zeros((batch_size, max_len_batch), dtype=np.float32)

        for i in range(batch_size):
            inp[i], out[i], mask[i] = self.gen_example(seq_len_batch[i], max_len_batch)

        inp = torch.tensor(inp).float()
        out = torch.tensor(out).float()
        mask = torch.tensor(mask).float()

        return inp, out, mask

    def __iter__(self):
        while True:
            distrib = np.ones(self.max_len - self.min_len + 1, dtype=np.float32)
            yield self.gen_batch(
                self.batch_size,
                self.min_len, self.max_len,
                distribution=distrib / distrib.sum()
            )

    @staticmethod
    def loss(prediction, target, mask):
        """Compute scalar NLL of target sequence.

        Irrelevant time steps are masked out by mask tensor.

        Args:
          prediction: batch first 3D tensor with predictions
          target: batch first 3D tensor with targets
          mask: batch first 2D tensor of {1, 0} to mask time steps
        """
        xent = F.binary_cross_entropy(prediction, target, reduction='none')
        loss_time_batch = xent.sum(-1)
        loss_batch = torch.sum(loss_time_batch * mask, dim=-1)
        return loss_batch.sum() / loss_batch.size(0)


if __name__ == '__main__':
    obj = Arithmetic(3, 5, 10, task='*', seed=2)
    inp, out, mask= obj.gen_batch(3, 5, 10, distribution=np.full(6, 1/6))
    print(inp)
    print(out)
    print(mask)
