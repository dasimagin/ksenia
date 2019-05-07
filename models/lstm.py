import torch
import numpy as np
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM for sequence tasks."""
    def __init__(self, n_inputs, n_outputs, n_hidden, n_layers):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(n_hidden, n_outputs)

        # The hidden state is a learned parameter
        self.lstm_h_bias = nn.Parameter(torch.randn(self.n_layers, 1, self.n_hidden) * 0.05)
        self.lstm_c_bias = nn.Parameter(torch.randn(self.n_layers, 1, self.n_hidden) * 0.05)
        self.reset_parameters()

    def reset_parameters(self):
        # Linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

        # LSTM
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.n_inputs + self.n_hidden))
                nn.init.uniform_(p, -stdev, stdev)

    def calculate_num_params(self):
        count = 0
        for p in self.parameters():
            count += p.data.view(-1).size(0)
        return count

    def forward(self, x):
        """Run forward on one sequence. Many to many setting (direct map).
        Inputs must be batch first
        """
        batch_size, seq_len, seq_width = x.shape

        h_0 = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        c_0 = self.lstm_c_bias.clone().repeat(1, batch_size, 1)

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        return self.fc(out)
