"""Different ontrollers."""
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter


class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""

    def __init__(self, num_inputs, num_outputs, num_layers):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(
            self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(
            self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state


# TODO add support to NTM
class LinearController(nn.Module):
    "NTM controller based on FCNN."

    def __init__(self, layers):
        super().__init__()

        self.num_inputs = layers[0]
        self.num_outputs = layers[-1]
        self.num_layers = len(layers) - 1

        mlp = []

        for i, (in_features, out_features) in enumerate(zip(layers[:-1], layers[1:]), 1):
            mlp.append(nn.Linear(in_features, out_features))
            if i != self.num_layers:
                mlp.append(nn.LeakyReLU())

        self.mlp = nn.Sequential(*mlp)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim == 1:
                nn.init.xavier_uniform_(p, gain=1)
            else:
                nn.init.normal_(p, std=0.01)

    def forward(self, x):
        return self.mlp(x)
