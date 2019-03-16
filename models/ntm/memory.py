"""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        """
        super().__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias TODO: different initializations
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.memory * (1 - erase) + add

    def address(self, key, beta, gate, shift, gamma, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.
        """
        # Content focus
        w_content = self._similarity(key, beta)

        # Location focus
        w_gate = self._interpolate(w_prev, w_content, gate)
        w_shift = self._shift(w_gate, shift)
        w = self._sharpen(w_shift, gamma)

        return w

    def _similarity(self, key, beta):
        k = key.view(self.batch_size, 1, -1)
        w = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, w, g):
        return g * w + (1 - g) * w_prev

    def _shift(self, w, s):
        result = torch.zeros(w.size())
        for b in range(self.batch_size):
            result[b] = _convolve(w[b], s[b])
        return result

    def _sharpen(self, w, gamma):
        w_sharp = w ** gamma
        w_sharp = torch.div(w_sharp, torch.sum(w_sharp, dim=1).view(-1, 1) + 1e-16)
        return w
