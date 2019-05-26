import functools

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def dict_append(d, name, tensor):
    if d is not None:
        values = d.get(name)
        if not values:
            values = []
            d[name] = values
        values.append(tensor.squeeze().detach().cpu().numpy())


def init_debug(debug, initial):
    if debug is not None and not debug:
        debug.update(initial)


def dict_get(dict, name):
    return dict.get(name) if dict is not None else None


def get_next_tensor_part(src, dims, prev_pos=0):
    if not isinstance(dims, list):
        dims = [dims]
    n = functools.reduce(lambda x, y: x * y, dims)
    data = src.narrow(-1, prev_pos, n)
    return data.contiguous().view(list(data.size())[:-1] + dims) if len(dims) > 1 else data, prev_pos + n


def split_tensor(src, shapes):
    pos = 0
    res = []
    for s in shapes:
        d, pos = get_next_tensor_part(src, s, pos)
        res.append(d)
    return res


def linear_reset(module, gain=1.0):
    if isinstance(module, torch.nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.zero_()


def circular_convolution(weights, shifts):
    """Perform circular convolution for each head's weightings.
    Args:
      weights: tensor of shape `[batch size, num heads, num cells]` weight distributions
      shifts: tensor of shape `[batch size, num heads, 3]` shift vectors
    """
    padded = torch.cat([weights[:, :, -1:], weights, weights[:, :, :1]], dim=-1)
    return F.conv2d(
        padded.view(1, -1, 1, padded.size(-1)),
        shifts.view(-1, 1, 1, shifts.size(-1)),
        groups=padded.size(0) * padded.size(1),
    ).view(weights.size())


def cosine_distance(memory, keys):
    memory = memory.unsqueeze(1)
    keys = keys.unsqueeze(-2)
    norm = keys.norm(dim=-1)
    norm = norm * memory.norm(dim=-1)
    scores = (memory * keys).sum(-1) / (norm + EPS)
    return scores


def address_memory(memory, keys, betas, gates, shifts, gammas, prev_weights, debug=None):
    # constraints
    keys = torch.tanh(keys)
    betas = F.softplus(betas)
    gates = torch.sigmoid(gates)
    shifts = F.softmax(shifts, dim=-1)
    gammas = 1.0 + F.softplus(gammas)

    # content based attention
    scores = cosine_distance(memory, keys)
    scores = scores * betas
    w_content = F.softmax(scores, scores.dim() - 1)

    # interpolate & shift weights
    w_interpolated = gates * w_content + (1 - gates) * prev_weights
    w_shift = circular_convolution(w_interpolated, shifts)

    # sharpen [batch size, num heads, cells count]
    w_sharp = w_shift ** gammas
    w_sharp = torch.div(w_sharp, w_sharp.sum(-1).unsqueeze(-1) + EPS)

    dict_append(debug, "gates", gates)
    dict_append(debug, "betas", betas)
    dict_append(debug, "shifts", shifts)
    dict_append(debug, "gammas", gammas)

    return w_sharp


class WriteHead(nn.Module):
    def __init__(self, n_heads, mem_word_length, n_cells):
        super().__init__()
        self.n_heads = n_heads
        self.mem_word_length = mem_word_length
        self.n_cells = n_cells
        self.input_partition = [self.mem_word_length] * 3 + [1, 1, 3, 1]
        self.input_size = self.n_heads * sum(self.input_partition)
        self.shapes = [[self.n_heads, size] for size in self.input_partition]
        self.write_dist = None
        self.write_data = None
        self.erase_data = None

        self.write_dist_bias = nn.Parameter(torch.randn(1, n_heads, n_cells) * 10)
        self.write_data_bias = nn.Parameter(torch.randn(1, n_heads, mem_word_length) * 0.05)

    def new_sequence(self):
        self.write_dist = None
        self.write_data = None
        self.erase_data = None

    @staticmethod
    def mem_update(memory, write_dist, erase_vector, write_vector):
        """"""
        erase_matrix = torch.prod(1.0 - write_dist.unsqueeze(-1) * erase_vector.unsqueeze(-2), dim=1)
        update_matrix = write_dist.transpose(1, 2) @ write_vector
        return memory * erase_matrix + update_matrix

    def get_prev_dist(self, memory):
        if self.write_dist is None:
            return F.softmax(
                self.write_dist_bias.clone().repeat(memory.size(0), 1, 1),
                dim=-1
            )
        return self.write_dist

    def get_prev_data(self, memory):
        if self.write_data is None:
            return torch.tanh(self.write_data_bias.clone().repeat(memory.size(0), 1, 1))
        return self.write_data

    def forward(self, memory, controls, debug):
        """Perform n_heads writes to memory given controls vector.

        Args:
          memory: NTM memory
          controls: controls vector

        Returns:
          torch.tensor: new NTM memory after n_heads writes
        """
        tensors = split_tensor(controls, self.shapes)
        keys, erase_vectors, write_vectors = tensors[:3]
        betas, gates, shifts, gammas = tensors[3:]

        self.write_dist = address_memory(
            memory, keys, betas, gates, shifts, gammas, self.get_prev_dist(memory), debug=debug
        )

        self.write_data = torch.tanh(write_vectors)
        self.erase_data = torch.sigmoid(erase_vectors)

        dict_append(debug, "write_weights", self.write_dist)
        dict_append(debug, "write_vectors", write_vectors)
        dict_append(debug, "erase_vectors", erase_vectors)

        return WriteHead.mem_update(memory, self.write_dist, self.erase_data, self.write_data)


class ReadHead(nn.Module):
    def __init__(self, n_heads, mem_word_length, n_cells):
        super().__init__()
        self.n_heads = n_heads
        self.mem_word_length = mem_word_length
        self.n_cells = n_cells
        self.input_partition = [self.mem_word_length, 1, 1, 3, 1]
        self.input_size = self.n_heads * sum(self.input_partition)
        self.shapes = [[self.n_heads, size] for size in self.input_partition]
        self.read_dist = None
        self.read_data = None
        self.read_dist_bias = nn.Parameter(torch.zeros(1, n_heads, n_cells))
        self.read_data_bias = nn.Parameter(torch.randn(1, n_heads, mem_word_length) * 0.05)

    def new_sequence(self):
        self.read_dist = None
        self.read_data = None

    def get_prev_dist(self, memory):
        if self.read_dist is None:
            return F.softmax(
                self.read_dist_bias.clone().repeat(memory.size(0), 1, 1),
                dim=-1,
            )
        return self.read_dist

    def get_prev_data(self, memory):
        if self.read_data is None:
            return torch.tanh(self.read_data_bias.clone().repeat(memory.size(0), 1, 1))
        return self.read_data

    def forward(self, memory, controls, debug):
        """Address NTM memory given controls vector

        Args:
          memory: memory tensor of shape `[batch size, cells count, cell width]`
          controls: controls tensor of shape `[batch size, controls size]`
        """
        keys, betas, gates, shifts, gammas = split_tensor(controls, self.shapes)

        self.read_dist = address_memory(
            memory, keys, betas, gates, shifts, gammas, self.get_prev_dist(memory), debug=debug)
        self.read_data = (memory.unsqueeze(1) * self.read_dist.unsqueeze(-1)).sum(-2)

        dict_append(debug, "read_weights", self.read_dist)
        dict_append(debug, "read_data", self.read_data)

        return self.read_data


class NTM(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            mem_word_length,
            mem_cells_count,
            n_writes,
            n_reads,
            controller_n_hidden,
            controller_n_layers,
            clip_value,
            controller='lstm',
            controller_output=None,
            layer_sizes=None,
            dropout=0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.write_head = WriteHead(n_writes, mem_word_length, mem_cells_count)
        self.read_head = ReadHead(n_reads, mem_word_length, mem_cells_count)

        controller_input = input_size + n_reads * mem_word_length
        controls_size = self.read_head.input_size + self.write_head.input_size

        self.controller = 0
        if controller is None or controller == 'lstm':
            self.controller = LSTMController(controller_input, controller_n_hidden, controller_n_layers)
        elif controller == 'feedforward':
            self.controller = FFController(controller_input, controller_n_hidden, layer_sizes)
        self.clip_value = clip_value
        self.controller_to_controls = nn.Linear(controller_n_hidden, controls_size)
        self.reads_to_output = nn.Linear(n_reads * mem_word_length, output_size)
        self.controller_to_output = nn.Linear(controller_n_hidden, output_size)

        self.mem_word_length = mem_word_length
        self.mem_cells_count = mem_cells_count
        self.memory = None
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        linear_reset(self.controller_to_controls)
        linear_reset(self.controller_to_output)
        linear_reset(self.reads_to_output)
        self.controller.reset_parameters()

    def calculate_num_params(self):
        count = 0
        for p in self.parameters():
            count += p.data.view(-1).size(0)
        return count

    def step(self, inp, debug):
        """Perform one NTM time step on a batch of vectors."""
        init_debug(debug, {
            'read_head': {},
            'write_head': {},
        })

        batch_size = inp.size(0)
        prev_read_data = self.read_head.get_prev_data(self.memory).view(batch_size, -1)
        controller_output = self.controller(torch.cat([inp, prev_read_data], -1))
        controls_vector = self.controller_to_controls(controller_output)
        controls_vector = controls_vector.clamp(-self.clip_value, self.clip_value)

        shapes = [[self.write_head.input_size], [self.read_head.input_size]]
        write_head_controls, read_head_controls = split_tensor(controls_vector, shapes)

        self.memory = self.write_head(
            self.memory,
            write_head_controls,
            dict_get(debug, "write_head"))
        reads = self.read_head(
            self.memory,
            read_head_controls,
            dict_get(debug, "read_head"))
        return (
            self.controller_to_output(self.dropout(controller_output)) +
            self.reads_to_output(reads.view(batch_size, -1))
        )

    def mem_init(self, batch_size, device):
        self.memory = torch.zeros(batch_size, self.mem_cells_count, self.mem_word_length).to(device)

    def forward(self, x, debug=None):
        """Run ntm on a batch of sequences.

        Args:
          x: 3-D tensor of shape `[batch size, seq len, word length]`
          debug: dict where debug information is stored

        Returns:
          3-D tensor of shape `[sequence length, batch size, output length]`
        """
        self.read_head.new_sequence()
        self.write_head.new_sequence()
        self.controller.new_sequence()
        self.mem_init(x.size(0), x.device)

        out = []
        for t in range(x.size(1)):
            out.append(self.step(x[:, t], debug))

        return torch.stack(out, dim=1)


class LSTMController(nn.Module):
    """NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(
            input_size=self.num_inputs,
            hidden_size=self.num_outputs,
            num_layers=self.num_layers,
        )

        self.prev_state = None

        self.lstm_h_bias = nn.Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = nn.Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.reset_parameters()

    def new_sequence(self):
        self.prev_state = None

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def forward(self, x):
        if self.prev_state is None:
            lstm_h = self.lstm_h_bias.clone().repeat(1, x.size(0), 1)
            lstm_c = self.lstm_c_bias.clone().repeat(1, x.size(0), 1)
            self.prev_state = (lstm_h, lstm_c)

        out, self.prev_state = self.lstm(x.unsqueeze(0), self.prev_state)
        return out.squeeze(0)


class FFController(nn.Module):
    """NTM controller based on feedforward network."""
    def __init__(self, num_inputs, num_outputs, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        layers = [nn.Linear(num_inputs, layer_sizes[0]), nn.ReLU()]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], num_outputs))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def new_sequence(self):
        pass

    def reset_parameters(self):
        self.net.apply(linear_reset)

    def forward(self, x):
        out = self.net(x)
        return out
