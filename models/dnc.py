import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

import functools
from collections import namedtuple


EPS = 1e-8


WriteArchive = namedtuple(
    'WriteArchive',
    ['write_weights', 'add_vector', 'erase_vector', 'write_gate', 'alloc_gate']
)


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
    assert isinstance(module, torch.nn.Linear)
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if module.bias is not None:
        module.bias.data.zero_()


class ContentAddressing(nn.Module):
    """Content based attention module.

    Content based attention with cosine distance and masking.
    """
    def __init__(self, mask_min=0.0):
        super().__init__()
        self.mask_min = mask_min

    def forward(self, memory, keys, betas, mask=None):
        """Produce content based attention vector

        Args:
          memory: memory tensor of shape [batch size, cells count, cell width]
          keys: keys of shape [batch size, num heads, cell width] or [batch size, cell width]
          betas: values for softmax temperature of shape [batch size, num heads, 1] or [batch size, 1]
          mask: (optional) mask for content based addresing [batch size, num heads, cell width]
           or [batc size, cell width]

        Returns:
          content weights: tensor of shape [batch size, num heads, cells count]
        """
        if mask is not None:
            mask = mask * (1.0 - self.mask_min) + self.mask_min

        single_head = keys.dim() == 2
        if single_head:
            keys = keys.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        memory = memory.unsqueeze(1)
        keys = keys.unsqueeze(-2)

        if mask is not None:
            mask = mask.unsqueeze(-2)
            memory = memory * mask
            keys = keys * mask

        norm = keys.norm(dim=-1)
        norm = norm * memory.norm(dim=-1)

        scores = (memory * keys).sum(-1) / (norm + EPS)
        scores *= betas.unsqueeze(-1)

        res = F.softmax(scores, dim=-1)
        return res.squeeze(1) if single_head else res


class AllocationAddressing(torch.nn.Module):
    """Dynamic memory allocation.

    Supports mulpitle write and read heads.
    """
    def __init__(self):
        super().__init__()
        self.zero_usages = None
        self.usages = None

    def new_sequence(self):
        self.usages = None

    def update_usages(self, write_weights, prev_reads, free_gates):
        """Usage vector update.

        Updates usage vector and returns memory retention vector phi.

        Args:
          prev_write: WriteArchive
          prev_reads: tensor of shape [batch size, num read heads, cells count]
          free_gates: tensor of shape [batch size, num read heads]

        Returns:
          phi: tensor of shape [batch size, cells count] memory retention vector
        """
        batch_size, n_reads, n_cells = prev_reads.shape
        device = prev_reads.device

        one = torch.ones(1, device=device)
        phi = torch.addcmul(
            one, -1, free_gates.view(batch_size, n_reads, 1), prev_reads).prod(-2)

        if self.usages is None or list(self.usages.shape) != [batch_size, n_cells]:
            self.usages = torch.zeros(batch_size, n_cells, device=device)
        else:
            self.usages = torch.addcmul(
                self.usages, 1, write_weights.detach(), 1 - self.usages) * phi
        return phi

    def forward(self, write_weights, read_weights, free_gates, write_gate, diff_alloc):
        """Computes allocation weightings for all write heads.

        Args:
          prev_writes: WriteArchive or None (if first step)
          read_weights: tensor of shape [batch size, num read heads, cells count]
          free_gates: tensor of shape [batch size, num read heads] with [0, 1] free gates
            from controller for each read head
          write_gates: tensor of shape [batch size, 1] with values in
            the range [0, 1] indicating how much each write head does writing
            based on the address returned here (and hence how much usage
            increases). (for multiple write heads)
          diff_alloc: tensor of shape [batch size, 1] sharpening parameter for differentiable
            memory allocation

        Returns:
          alloc_dist, phi: tensor(batch size, num write heads, cells count), tensor(batch size, cells count)
            allocation weightings and memory retention vector
        """
        # Calculation of usage is not
        # differentiable with respect to write weights.
        phi = self.update_usages(write_weights, read_weights, free_gates)

        if diff_alloc is None:
            sorted_usages, free_list = (self.usages * (1.0 - EPS) + EPS).sort(-1)

            u_prod = sorted_usages.cumprod(-1)
            one_minus_usage = 1.0 - sorted_usages
            sorted_scores = torch.cat(
                [one_minus_usage[..., 0:1], one_minus_usage[..., 1:] * u_prod[..., :-1]],
                dim=-1
            )
            alloc_dist = sorted_scores.clone().scatter_(-1, free_list, sorted_scores)
        else:
            # hacky initialization of first usage for better start allocation
            if write_weights is None:
                batch_size, _, n_cells = read_weights.shape
                one_minus_usage = torch.zeros(batch_size, n_cells, device=read_weights.device)
                one_minus_usage[:, 0] = 1
            else:
                one_minus_usage = 1.0 - self.usages

            alloc_dist = F.softmax(one_minus_usage * diff_alloc, dim=-1)

        return alloc_dist, phi


class WriteHead(nn.Module):
    """DNC write head.

    Perform one write to memory using 2 different attention mechanisms:
     content based attention (masking/no masking: option)
     dynamic memory allocation (sort/no sort: option)
    """
    def __init__(
            self,
            n_reads,
            cell_width,
            mask_min=0.0,
            masking=False,
            diff_alloc=False,
            dealloc=False,
    ):
        super().__init__()
        self.n_reads = n_reads
        self.cell_width = cell_width

        self.masking = masking
        self.diff_alloc = diff_alloc
        self.dealloc = dealloc

        self.content_addressing = ContentAddressing(mask_min=mask_min)
        self.allocation_addressing = AllocationAddressing()

        if masking:
            self.shapes = [cell_width] * 4
        else:
            self.shapes = [cell_width] * 3

        if diff_alloc:
            self.shapes += [n_reads] + [1] * 4
        else:
            self.shapes += [n_reads] + [1] * 3

        self.input_size = sum(self.shapes)
        self.last_write = None

    def new_sequence(self):
        self.last_write = None
        self.allocation_addressing.new_sequence()

    def _make_write_archive(
            self,
            write_weights,
            add_vector, erase_vector,
            write_gate, alloc_gate,
    ):
        return WriteArchive(write_weights, add_vector, erase_vector, write_gate, alloc_gate)

    @staticmethod
    def mem_update(memory, weights, erase_vector, add_vector, phi):
        weights = weights.unsqueeze(-1)

        erase_matrix = 1.0 - weights * erase_vector.unsqueeze(-2)

        if phi is not None:
            erase_matrix = erase_matrix * phi.unsqueeze(-1)

        update_matrix = weights * add_vector.unsqueeze(-2)
        return memory * erase_matrix + update_matrix

    def forward(self, memory, controls, read_weights, debug=None):
        """Perform n_heads writes to memory given controls vector and prev_reads.

        Args:
          memory: NTM memory  [batch size, cells count, cell width]
          controls: controls vector [batch size, self.input_size]
          prev_read_dist: previos reads weightings [batch size, heads count, cells count]

        Returns:
          torch.tensor: new NTM memory after write
        """
        tensors = split_tensor(controls, self.shapes)

        alloc_beta = None
        mask = None

        if self.masking:
            mask = torch.sigmoid(tensors[0])
            tensors = tensors[1:]

        if self.diff_alloc:
            alloc_beta = tensors[-1]
            alloc_beta = 1 + F.softplus(alloc_beta)
            tensors = tensors[:-1]

        key, erase_vector, add_vector = tensors[:3]
        free_gates, beta, alloc_gate, write_gate = tensors[3:]

        erase_vector = torch.sigmoid(erase_vector)
        beta = 1 + F.softplus(beta)
        free_gates = torch.sigmoid(free_gates)
        alloc_gate = torch.sigmoid(alloc_gate)
        write_gate = torch.sigmoid(write_gate)

        if self.last_write is None:
            last_write_weights = None
        else:
            last_write_weights = self.last_write.write_weights

        content_weights = self.content_addressing(memory, key, beta, mask)
        alloc_weights, phi = self.allocation_addressing(
            last_write_weights,
            read_weights,
            free_gates,
            write_gate,
            alloc_beta,
        )

        write_weights = write_gate * (
            alloc_gate * alloc_weights + (1 - alloc_gate) * content_weights
        )

        self.last_write = self._make_write_archive(
            write_weights,
            add_vector,
            erase_vector,
            write_gate,
            alloc_gate,
        )

        dict_append(debug, "write_weights", write_weights)
        dict_append(debug, "alloc_weights", alloc_weights)
        dict_append(debug, "usages", self.allocation_addressing.usages)
        dict_append(debug, "free_gates", free_gates)
        dict_append(debug, "write_beta", beta)
        dict_append(debug, "write_gate", write_gate)
        dict_append(debug, "alloc_gate", alloc_gate)
        dict_append(debug, "add_vector", add_vector)
        dict_append(debug, "erase_vector", erase_vector)
        if mask is not None:
            dict_append(debug, "write_mask", mask)
        if alloc_beta is not None:
            dict_append(debug, "alloc_beta", alloc_beta)

        return WriteHead.mem_update(
            memory, write_weights,
            erase_vector, add_vector,
            phi if self.dealloc else None
        )


class TemporalMemoryLinkage(nn.Module):
    def __init__(self):
        super().__init__()
        self.links = None
        self.precedence = None
        self.diag_mask = None

        self.initial_links = None
        self.initial_precedence = None
        self.initial_diag_mask = None
        self.initial_shape = None

    def new_sequence(self):
        self.links = None
        self.precedence = None
        self.diag_mask = None

    def _init_link(self, write_weights):
        batch_size, n_cells = write_weights.shape
        device = write_weights.device
        if self.initial_shape is None or self.initial_shape != [batch_size, n_cells, n_cells]:
            self.initial_links = torch.zeros(batch_size, n_cells, n_cells).to(device)
            self.initial_precedence = torch.zeros(batch_size, n_cells).to(device)
            self.initial_diag_mask = (1.0 - torch.eye(n_cells).unsqueeze(0).to(device)).detach()
            self.initial_shape = [batch_size, n_cells, n_cells]

        self.links = self.initial_links
        self.precedence = self.initial_precedence
        self.diag_mask = self.initial_diag_mask

    def _update_precedence(self, write_weights):
        self.precedence = (
            1 - write_weights.sum(-1, keepdim=True)) * self.precedence + write_weights

    def _update_links(self, write_weights):
        """Update link matrix from write weights

        Link matrix is stored for each (maybe multiple) write head. It has
        shape of [batch size, num write heads, memory size, memory size]

        Args:
          w_dist: tensor of shape [batch size, num write heads, memory size]
            containing the memory addresses of the different write heads.
        """

        if self.links is None:
            self._init_link(write_weights)

        wt_i = write_weights.unsqueeze(-1)
        wt_j = write_weights.unsqueeze(-2)
        pt_j = self.precedence.unsqueeze(-2)

        self.links = ((1 - wt_i - wt_j) * self.links + wt_i * pt_j) * self.diag_mask

    def forward(self, write_weights, read_weights, debug=None):
        """Update links and find forward and backward weightings.

        Args:
          w_dist: tensor of last write weightings of shape [batch size, num write heads, cells count]
          prev_reads: tensor of previous read weightings of shape [batch size, num read heads, cells count]

        Returns:
          forward_dist: tensor with future weightings
            of shape [batch size, num read heads, num write heads, cells count]
          backward_dist: tensor with previous weightings
            of shape [batch size, num read heads, num write heads, cells count]
        """
        self._update_links(write_weights)
        self._update_precedence(write_weights)

        links_multi_head = self.links.unsqueeze(1)

        forward_weights = (links_multi_head * read_weights.unsqueeze(-2)).sum(-1)
        backward_weights = (links_multi_head * read_weights.unsqueeze(-1)).sum(-2)

        return forward_weights, backward_weights


class ReadHead(nn.Module):
    def __init__(
            self,
            n_reads,
            cell_width,
            masking=False,
            mask_min=0.0,
            links=True,
            links_sharpening=False,
    ):
        super().__init__()
        self.n_reads = n_reads
        self.cell_width = cell_width
        self.masking = masking
        self.links = links
        self.links_sharpening = links_sharpening

        self.content_addressing = ContentAddressing(mask_min)
        self.temporal_addressing = TemporalMemoryLinkage()

        if masking:
            self.shapes = [[n_reads, cell_width]] * 2
        else:
            self.shapes = [[n_reads, cell_width]]

        self.shapes += [n_reads]

        if links:
            self.shapes += [[n_reads, 3]]

        if links_sharpening:
            self.shapes += [[n_reads, 2]]

        self.input_size = sum(
            part[0] * part[1] if isinstance(part, list) else part
            for part in self.shapes
        )
        self.read_weights = None
        self.read_vector = None

    def new_sequence(self):
        self.read_weights = None
        self.read_vector = None
        self.temporal_addressing.new_sequence()

    def get_last_weights(self, memory):
        if self.read_weights is None:
            mem_shape = memory.shape
            return torch.zeros(mem_shape[0], self.n_reads, mem_shape[1]).to(memory)
        return self.read_weights

    def get_last_vector(self, memory):
        if self.read_vector is None:
            mem_shape = memory.shape
            return torch.zeros(mem_shape[0], self.n_reads, mem_shape[2]).to(memory)
        return self.read_vector

    def _sharpen_weights(self, weights, sharpening):
        weights += EPS
        weights = weights / weights.max(dim=-1, keepdim=True)[0]
        weights = weights.pow(sharpening)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights

    def forward(self, memory, controls, write_weights, debug=None):
        """Address NTM memory given controls vector

        Args:
          memory: memory tensor of shape `[batch size, cells count, cell width]`
          controls: controls tensor of shape `[batch size, controls size]`
        """
        tensors = split_tensor(controls, self.shapes)

        # Options
        masks = None
        gates = None
        sharpening = None

        # Masking optional params
        if self.masking:
            masks = torch.sigmoid(tensors[0])
            tensors = tensors[1:]

        # Links optional params
        if self.links_sharpening:
            sharpening = 1 + F.softplus(tensors[-1])
            tensors = tensors[:-1]

        if self.links:
            gates = F.softmax(tensors[-1], dim=-1)
            tensors = tensors[:-1]

        keys, betas = tensors
        betas = 1 + F.softplus(betas)

        # Addressing
        content_weights = self.content_addressing(memory, keys, betas, masks)

        if self.links:
            forward_weights, backward_weights = self.temporal_addressing(
                write_weights,
                self.get_last_weights(memory),
            )

            # Gates for each mode for each write head
            forward_mode = gates[..., 0:1]
            backward_mode = gates[..., 1:2]
            content_mode = gates[..., 2:]

            if self.links_sharpening:
                forward_weights = self._sharpen_weights(forward_weights, sharpening[..., :1])
                backward_weights = self._sharpen_weights(backward_weights, sharpening[..., 1:])

            self.read_weights = (
                forward_mode * forward_weights +
                backward_mode * backward_weights +
                content_mode * content_weights
            )

            dict_append(debug, "forward_weights", forward_weights)
            dict_append(debug, "bacward_weights", backward_weights)
            dict_append(debug, "read_modes", gates)
        else:
            self.read_weights = content_weights

        self.read_vector = (memory.unsqueeze(1) * self.read_weights.unsqueeze(-1)).sum(-2)

        dict_append(debug, "read_weights", self.read_weights)
        if masks is not None:
            dict_append(debug, "read_masks", masks)

        return self.read_vector


class DNC(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            n_cells,
            cell_width,
            n_reads,
            controller_n_hidden,
            controller_n_layers,
            clip_value,
            masking=False,
            mask_min=0.0,
            dealloc=False,
            diff_alloc=False,
            links=True,
            links_sharpening=False,
            normalization=False,
            dropout=0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.clip_value = clip_value
        self.normalization = normalization
        self.dropout = nn.Dropout(p=dropout)

        self.write_head = WriteHead(
            n_reads,
            cell_width,
            masking=masking, mask_min=mask_min,
            diff_alloc=diff_alloc,
            dealloc=dealloc,
        )

        self.read_head = ReadHead(
            n_reads,
            cell_width,
            masking=masking,
            mask_min=mask_min,
            links=links,
            links_sharpening=links_sharpening,
        )

        controller_input = input_size + n_reads * cell_width
        controls_size = self.read_head.input_size + self.write_head.input_size

        if normalization:
            self.normalization = nn.LayerNorm(normalized_shape=controls_size)
        else:
            self.normalization = None

        self.controller = LSTMController(controller_input, controller_n_hidden, controller_n_layers)
        self.controller_to_controls = nn.Linear(controller_n_hidden, controls_size)
        self.controller_to_output = nn.Linear(controller_n_hidden, output_size)
        self.reads_to_output = nn.Linear(cell_width * n_reads, output_size)

        self.cell_width = cell_width
        self.n_cells = n_cells
        self.memory = None
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

        # Run controller
        last_reads = self.read_head.get_last_vector(self.memory).view(batch_size, -1)
        controller_output = self.controller(torch.cat([inp, last_reads], -1))
        controls_vector = self.controller_to_controls(controller_output)
        controls_vector = controls_vector.clamp(-self.clip_value, self.clip_value)

        if self.normalization is not None:
            controls_vector = self.normalization(controls_vector)

        # Split controls
        shapes = [[self.write_head.input_size], [self.read_head.input_size]]
        write_head_controls, read_head_controls = split_tensor(controls_vector, shapes)

        # Write and Read
        self.memory = self.write_head(
            self.memory,
            write_head_controls,
            self.read_head.get_last_weights(self.memory),
            dict_get(debug, 'write_head'),
        )

        reads = self.read_head(
            self.memory,
            read_head_controls,
            self.write_head.last_write.write_weights,
            dict_get(debug, 'read_head'),
        )

        return (
            self.controller_to_output(self.dropout(controller_output)) +
            self.reads_to_output(reads.view(batch_size, -1))
        )

    def mem_init(self, batch_size, device):
        self.memory = torch.zeros(batch_size, self.n_cells, self.cell_width).to(device)

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
