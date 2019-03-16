import numpy as np
import torch


class CopyDataLoader:
    def __init__(self, num_batches, batch_size, seq_width, min_len, max_len):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_width = seq_width
        self.min_len = min_len
        self.max_len = max_len

    def generate(self):
        """Generator of random sequences for the copy task.

        Creates random batches of "bits" sequences.

        All the sequences within each batch have the same length.
        The length is [`min_len`, `max_len`]
        """
        for batch_num in range(self.num_batches):

            # All batches have the same sequence length
            seq_len = np.random.randint(self.min_len, self.max_len + 1)
            seq = np.random.binomial(1, 0.5, (seq_len, self.batch_size, self.seq_width))
            seq = torch.from_numpy(seq)

            # The input includes an additional channel used for the delimiter
            inp = torch.zeros(seq_len + 1, self.batch_size, self.seq_width + 1)
            inp[:seq_len, :, :self.seq_width] = seq
            inp[seq_len, :, self.seq_width] = 1.0  # delimiter in our control channel
            outp = seq.clone()

            yield batch_num + 1, inp.float(), outp.float()
