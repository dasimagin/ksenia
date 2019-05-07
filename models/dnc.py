бimport torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import functools


EPS = 1e-6


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
    assert isinstance(module, torch.nn.Linear)
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if module.bias is not None:
        module.bias.data.zero_()


class ContentAddressing(nn.Module):
    def __init__(self, disable_content_norm=False, mask_min=0.0, disable_key_masking=False):
        super().__init__()
        self.disable_content_norm = disable_content_norm
        self.mask_min = mask_min
        self.disable_key_masking = disable_key_masking

    def forward(self, memory, keys, betas, mask=None):
        # Memory shape [batch, cell count, word length]
        # Key shape [batch, n heads*, word length]
        # Betas shape [batch, n heads]
        if mask is not None and self.mask_min != 0:
            mask = mask * (1.0-self.mask_min) + self.mask_min

        single_head = keys.dim() == 2
        if single_head:
            # Single head
            keys = keys.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        memory = memory.unsqueeze(1)
        keys = keys.unsqueeze(-2)

        if mask is not None:
            mask = mask.unsqueeze(-2)
            memory = memory * mask
            if not self.disable_key_masking:
                keys = keys * mask

        # Shape [batch, n heads, cell count]
        norm = keys.norm(dim=-1)
        if not self.disable_content_norm:
            norm = norm * memory.norm(dim=-1)

        scores = (memory * keys).sum(-1) / (norm + EPS)
        scores *= betas.unsqueeze(-1)

        res = F.softmax(scores, scores.dim()-1)
        return res


class Allocator(torch.nn.Module):
    def __init__(self, enable_sort=True):
        super().__init__()
        self.enable_sort = enable_sort
        self.usages = None
        self.zero_usages = None
        self.debug_sequ_init = False
        self.one = None

    def _init_sequence(self, prev_read_distributions):
        # prev_read_distributions size is [batch, n_heads, cell count]
        s = prev_read_distributions.size()
        if self.zero_usages is None or list(self.zero_usages.size())!=[s[0],s[-1]]:
            self.zero_usages = torch.zeros(s[0], s[-1], device = prev_read_distributions.device)
            if self.debug_sequ_init:
                self.zero_usages += torch.arange(0, s[-1]).unsqueeze(0) * 1e-10

        self.usages = self.zero_usages

    def _init_consts(self, device):
        if self.one is None:
            self.one = torch.ones(1, device=device)

    def new_sequence(self):
        self.usages = None

    def update_usages(self, prev_write_distribution, prev_read_distributions, free_gates):
        # Read distributions shape: [batch, n_heads, cell count]
        # Free gates shape: [batch, n_heads]

        self._init_consts(prev_read_distributions.device)
        phi = torch.addcmul(self.one, -1, free_gates.unsqueeze(-1), prev_read_distributions).prod(-2)
        # Phi is the free tensor, sized [batch, cell count]

        # If memory usage counter if doesn't exists
        if self.usages is None:
            self._init_sequence(prev_read_distributions)
            # in first timestep nothing is written or read yet, so we don't need any further processing
        else:
            self.usages = torch.addcmul(self.usages, 1, prev_write_distribution.detach(), (1 - self.usages)) * phi

        return phi

    def forward(self, prev_write_distribution, prev_read_distributions, free_gates):
        phi = self.update_usages(prev_write_distribution, prev_read_distributions, free_gates)
        sorted_usage, free_list = (self.usages*(1.0-_EPS)+_EPS).sort(-1)

        u_prod = sorted_usage.cumprod(-1)
        one_minus_usage = 1.0 - sorted_usage
        sorted_scores = torch.cat([one_minus_usage[..., 0:1], one_minus_usage[..., 1:] * u_prod[..., :-1]], dim=-1)

        return sorted_scores.clone().scatter_(-1, free_list, sorted_scores), phi


class WriteHead(nn.Module):
    def __init__(self, n_heads, mem_word_length, enable_masking=False, enable_sort=True):
        super().__init__()
        self.n_heads = n_heads
        self.mem_word_length = mem_word_length
        self.enable_masking = enable_masking
        self.allocator = Allocator(enable_sort)

        if enable_masking:
            self.input_partition = [self.mem_word_length] * 4 + [1, 1, 3, 1]
        else:
            self.input_partition = [self.mem_word_length] * 3 + [1, 1, 3, 1]

        self.input_size = self.n_heads * sum(self.input_partition)
        self.shapes = [[self.n_heads, size] for size in self.input_partition]
        self.write_dist = None
        self.write_data = None
        self.erase_data = None

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
            return torch.zeros(memory.size(0), self.n_heads, memory.size(1)).to(memory)
        return self.write_dist

    def get_prev_data(self, memory):
        if self.write_data is None:
            return torch.zeros(memory.size(0), self.n_heads, memory.size(2)).to(memory)
        return self.write_data

    def forward(self, memory, controls):
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

        self.write_dist = None

        self.write_data = write_vectors
        self.erase_data = torch.sigmoid(erase_vectors)

        return WriteHead.mem_update(memory, self.write_dist, self.erase_data, self.write_data)


class ReadHead(nn.Module):
    def __init__(self, n_heads, mem_word_length):
        super().__init__()
        self.n_heads = n_heads
        self.mem_word_length = mem_word_length
        self.input_partition = [self.mem_word_length, 1, 1, 3, 1]
        self.input_size = self.n_heads * sum(self.input_partition)
        self.shapes = [[self.n_heads, size] for size in self.input_partition]
        self.read_dist = None
        self.read_data = None

    def new_sequence(self):
        self.read_dist = None
        self.read_data = None

    def get_prev_dist(self, memory):
        if self.read_dist is None:
            mem_shape = memory.shape
            return torch.zeros(mem_shape[0], self.n_heads, mem_shape[1]).to(memory)
        return self.read_dist

    def get_prev_data(self, memory):
        if self.read_data is None:
            mem_shape = memory.shape
            return torch.zeros(mem_shape[0], self.n_heads, mem_shape[2]).to(memory)
        return self.read_data

    def address(self, memory, controls):
        pass

    def forward(self, memory, controls):
        """Address NTM memory given controls vector

        Args:
          memory: memory tensor of shape `[batch size, cells count, cell width]`
          controls: controls tensor of shape `[batch size, controls size]`
        """
        keys, betas, gates, shifts, gammas = split_tensor(controls, self.shapes)

        self.read_dist = self.address(memory, controls)
        self.read_data = (memory.unsqueeze(1) * self.read_dist.unsqueeze(-1)).sum(-2)
        return self.read_data


class DNC(nn.Module):
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
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.write_head = WriteHead(n_writes, mem_word_length)
        self.read_head = ReadHead(n_reads, mem_word_length)

        controller_input = input_size + n_reads * mem_word_length
        controls_size = self.read_head.input_size + self.write_head.input_size

        self.controller = LSTMController(controller_input, controller_n_hidden, controller_n_layers)
        self.clip_value = clip_value
        self.controller_to_controls = nn.Linear(controller_n_hidden, controls_size)
        self.controller_to_output = nn.Linear(controller_n_hidden, output_size)
        self.reads_to_output = nn.Linear(mem_word_length * n_reads, output_size)

        self.register_buffer('mem_bias', torch.Tensor(mem_cells_count, mem_word_length))
        self.mem_word_length = mem_word_length
        self.mem_cells_count = mem_cells_count
        self.memory = None
        self.reset_parameters()

    def reset_parameters(self):
        linear_reset(self.controller_to_controls)
        linear_reset(self.controller_to_output)
        linear_reset(self.reads_to_output)
        self.controller.reset_parameters()
        nn.init.constant_(self.mem_bias, EPS)   # as in implementing NTM

    def calculate_num_params(self):
        count = 0
        for p in self.parameters():
            count += p.data.view(-1).size(0)
        return count

    def step(self, inp, debug):
        """Perform one NTM time step on a batch of vectors."""
        batch_size = inp.size(0)
        prev_read_data = self.read_head.get_prev_data(self.memory).view(batch_size, -1)
        controller_output = self.controller(torch.cat([inp, prev_read_data], -1))
        controls_vector = self.controller_to_controls(controller_output)
        controls_vector = controls_vector.clamp(-self.clip_value, self.clip_value)

        shapes = [[self.write_head.input_size], [self.read_head.input_size]]
        write_head_controls, read_head_controls = split_tensor(controls_vector, shapes)

        self.memory = self.write_head(self.memory, write_head_controls)
        reads = self.read_head(self.memory, read_head_controls)
        return (self.controller_to_output(controller_output) +
                self.reads_to_output(reads.view(batch_size, -1)))

    def mem_init(self, batch_size, device):
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1).to(device)

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

        return torch.stack(out, dim=1).clamp(-self.clip_value, self.clip_value)


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
