"""Neural turing machine"""
import torch
from torch import nn

from .controller import BaseNTMController
from .memory import NTMMemory


class NTM(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 controller, heads, memory):
        """Initialize an EncapsulatedNTM.
        """
        super().__init__()

        assert isinstance(controller, BaseNTMController), "Controller must be a descendant of BaseNTMController."
        assert isinstance(memory, NTMMemory), "Memory object must be an instance of NTMMemory"

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.heads = nn.ModuleList(heads)
        self.memory = memory

        # Initialize sizes
        controller_inp, controller_outp = self.controller.size()
        N, M = self.memory.size()
        self.controller_outp = controller_outp
        self.N = N
        self.M = M

        # Initialize the initial previous read values to random biases
        device = next(self.parameters()).device
        self.num_read_heads = 0
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M, device=device) * 0.01
                self.register_buffer(f"read{self.num_read_heads}_bias", init_r_bias.data)
                self.num_read_heads += 1

        # Perform checks
        assert self.num_read_heads > 0, "Heads list must contain at least a single read head."

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_outp + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)

    def create_new_state(self, batch_size):
        init_r = []
        for name, r in self.named_buffers(recurse=True):
            if name.startswith('read'):
                init_r.append(r.clone().repeat(batch_size, 1))
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        return init_r, heads_state, controller_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def forward(self, x, prev_state):
        prev_reads, prev_heads_states, prev_controller_state = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                head_state, r = head(controller_outp, prev_head_state)
                reads.append(r)
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states.append(head_state)

        # Generate output
        controller_and_reads = torch.cat([controller_outp] + reads, dim=1)
        outp = torch.sigmoid(self.fc(controller_and_reads))

        # Pack the current state
        state = (reads, heads_states, controller_state)

        return outp, state
