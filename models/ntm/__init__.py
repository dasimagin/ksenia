"""Module with all NTM components."""

from .ntm import NTM
from .controller import LSTMController, LinearController
from .memory import NTMMemory
from .heads import NTMReadHead, NTMWriteHead


def create_ntm(
        num_inputs, num_outputs, N=128, M=20,
        controller_type='lstm',
        lstm_size=100, lstm_layers=1,
        linear_layers=None,
        num_heads=1,
):
    """Create model class with specified parameters."""

    if controller_type == 'lstm':
        controller = LSTMController(num_inputs + M * num_heads, lstm_size, lstm_layers)
    if controller_type == 'linear':
        if not linear_layers:
            linear_layers = [num_inputs + M * num_heads, 300, 400, 400]
        controller = LinearController(linear_layers)

    memory = NTMMemory(N, M)
    _, controller_size = controller.size()

    heads = []
    for i in range(num_heads):
        heads.append(NTMReadHead(memory, controller_size))
        heads.append(NTMWriteHead(memory, controller_size))

    return NTM(num_inputs, num_outputs, controller, heads, memory)


__all__ = [
    "NTM",
    "LSTMController",
    "LinearController",
    "NTMMemory",
    "NTMReadHead",
    "NTMWriteHead",
]
