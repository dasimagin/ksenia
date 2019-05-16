"""Dataloaders for different tasks with bit vectors.
"""
import logging

import torch
import numpy as np
import torch.nn.functional as F

import utils


class BitmapTask:
    @staticmethod
    def loss(prediction, target, mask):
        """Compute scalar NLL of target sequence.

        Irrelevant time steps are masked out by mask tensor.

        Args:
          prediction: batch first 3D tensor with predictions
          target: batch first 3D tensor with targets
          mask: batch first 2D tensor of {1, 0} to mask time steps
        """
        xent = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        loss_time_batch = xent.sum(-1)
        loss_batch = torch.sum(loss_time_batch * mask, dim=-1)
        return loss_batch.sum() / loss_batch.size(0)

    def _gen_batch(self):
        raise NotImplementedError

    def _gen_name(self, param):
        raise NotImplementedError

    def _get_examples(self, params):
        params = [dict(zip(params, i)) for i in zip(*params.values())]
        examples = [self._gen_batch(batch_size=1, **p) for p in params]
        return examples, params

    def generalization(self, model, step, writer, config):
        """Evaluate model generalization on longer sequences"""
        with torch.no_grad():
            inp, tar, mask = self._gen_batch(**config.evaluate.generalization)

            # Run model and collect debug info
            if config.evaluate.fit_memory:
                model.n_cells = config.evaluate.generalization.n_cells

            device = 'cuda' if config.gpu and torch.cuda.is_available() else 'cpu'
            inp = inp.to(device)
            tar = tar.to(device)
            mask = mask.to(device)
            pred = model(inp)

            if config.evaluate.fit_memory:
                model.n_cells = config.model.n_cells

            pred_binarized = (pred.clone().data > 0).float()
            cost_time_batch = torch.sum(torch.abs(pred_binarized - tar.data), dim=-1)
            cost_batch = torch.sum(cost_time_batch * mask, dim=-1)
            cost = cost_batch.sum() / inp.size(0)
            writer.add_scalar('test/cost', cost.item(), global_step=step*config.task.batch_size)

    def visualize(self, model, step, writer, config):
        """Evaluate model on few longer examples with visualizations"""
        device = 'cuda' if config.gpu and torch.cuda.is_available() else 'cpu'
        examples, params = self._get_examples(config.evaluate.visualization)

        # Evaluate with io and memory visualizations
        with torch.no_grad():
            for example, param in zip(examples, params):
                inp, tar, mask = example
                info = {}

                # Run model and collect debug info
                if config.evaluate.fit_memory:
                    model.n_cells = param['n_cells']

                out = model(
                    inp.to(device),
                    debug=info,
                )

                if config.evaluate.fit_memory:
                    model.n_cells = config.model.n_cells

                out = torch.sigmoid(out)
                out = out.detach().cpu().numpy()[0].T
                tar = tar.detach().cpu().numpy()[0].T
                mask = mask.detach().cpu().numpy()[0]
                start = np.flatnonzero(mask)[0]

                io = utils.input_output_img(tar[:, start:], out[:, start:])

                writer.add_image(
                    'io/' + self._gen_name(param),
                    io,
                    global_step=step * config.task.batch_size,
                )

                if config.model.name == 'dnc':
                    mem = utils.dnc_img(info)
                    writer.add_image(
                        'mem/' + self._gen_name(param),
                        mem,
                        global_step=step * config.task.batch_size,
                    )

    def evaluate(self, model, step, writer, config):
        self.visualize(model, step, writer, config)
        self.generalization(model, step, writer, config)


class CopyTask(BitmapTask):
    """Data generator for copy and repeat copy tasks.
    """
    def __init__(self, batch_size, min_len, max_len, bit_width=8, seed=1):
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.bit_width = bit_width
        self.full_input_width = bit_width + 1
        self.full_output_width = bit_width + 1
        self.rand = np.random.RandomState(seed)
        self.special_chars = 1

    def _gen_name(self, param):
        return f"len_{param['max_len']}"

    def _gen_batch(
            self,
            batch_size,
            min_len, max_len,
            **kwargs,
    ):
        full_input_width = self.full_input_width
        full_output_width = self.full_output_width
        bit_width = self.bit_width

        seq_len_batch = self.rand.randint(min_len, max_len + 1, size=batch_size)

        total_len_batch = seq_len_batch * 2 + 2
        max_len_batch = np.max(total_len_batch)

        inp = np.zeros((batch_size, max_len_batch, full_input_width))
        out = np.zeros((batch_size, max_len_batch, full_output_width))
        mask = np.zeros((batch_size, max_len_batch))

        # generate random vectors
        for i in range(batch_size):
            seq_len = seq_len_batch[i]
            total_len = total_len_batch[i]

            seq = self.rand.binomial(1, 0.5, size=(seq_len, bit_width)).astype(float)

            inp[i, :seq_len, :bit_width] = seq
            inp[i, seq_len, bit_width] = 1.0

            out[i, seq_len+1:total_len - 1, :bit_width] = seq
            out[i, total_len - 1, bit_width] = 1.0

            mask[i, seq_len + 1:total_len] = 1.0

        inp = torch.tensor(inp).float()
        out = torch.tensor(out).float()
        mask = torch.tensor(mask).float()

        return inp, out, mask

    def __iter__(self):
        while True:
            yield self._gen_batch(
                self.batch_size,
                self.min_len, self.max_len,
            )


class RepeatCopyTask(BitmapTask):
    def __init__(
            self,
            batch_size,
            bit_width,
            min_len,
            max_len,
            min_rep,
            max_rep,
            norm_max,
            seed,
    ):
        self.batch_size = batch_size
        self.bit_width = bit_width
        self.full_input_width = bit_width + 2
        self.full_output_width = bit_width + 1
        self.min_len = min_len
        self.max_len = max_len
        self.min_rep = min_rep
        self.max_rep = max_rep
        self.norm_max = norm_max
        self.rand = np.random.RandomState(seed)

    def _normalize(self, x):
        return x / self.norm_max

    def _gen_name(self, param):
        return f"len_{param['max_len']}_rep_{param['max_rep']}"

    def _gen_batch(
            self,
            batch_size,
            min_len, max_len,
            min_rep, max_rep,
            **kwargs,
    ):
        full_input_width = self.full_input_width
        full_output_width = self.full_output_width
        bit_width = self.bit_width

        seq_len_batch = self.rand.randint(min_len, max_len + 1, size=batch_size)
        num_rep_batch = self.rand.randint(min_rep, max_rep + 1, size=batch_size)

        total_len_batch = seq_len_batch * (num_rep_batch + 1) + 3
        max_len_batch = np.max(total_len_batch)

        inp = np.zeros((batch_size, max_len_batch, full_input_width))
        out = np.zeros((batch_size, max_len_batch, full_output_width))
        mask = np.zeros((batch_size, max_len_batch))

        # generate random vectors
        for i in range(batch_size):
            seq_len = seq_len_batch[i]
            total_len = total_len_batch[i]
            num_rep = num_rep_batch[i]

            seq = self.rand.binomial(1, 0.5, size=(seq_len, bit_width)).astype(float)

            inp[i, :seq_len, :bit_width] = seq
            inp[i, seq_len, bit_width] = 1.0
            inp[i, seq_len + 1, bit_width + 1] = self._normalize(num_rep)

            out[i, seq_len + 2:total_len - 1, :bit_width] = np.tile(seq, (num_rep, 1))
            out[i, total_len - 1, bit_width] = 1.0

            mask[i, seq_len + 2:total_len] = 1.0

        inp = torch.tensor(inp).float()
        out = torch.tensor(out).float()
        mask = torch.tensor(mask).float()

        return inp, out, mask

    def __iter__(self):
        while True:
            yield self._gen_batch(
                self.batch_size,
                self.min_len, self.max_len,
                self.min_rep, self.max_rep,
            )


class AssociativeRecallTask(BitmapTask):
    def __init__(
            self,
            batch_size,
            bit_width,
            item_len,
            min_cnt,
            max_cnt,
            seed,
    ):
        self.batch_size = batch_size
        self.bit_width = bit_width
        self.full_input_width = bit_width + 2
        self.full_output_width = bit_width + 1
        self.item_len = item_len
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        self.rand = np.random.RandomState(seed)

    def _gen_name(self, param):
        return f"cnt_{param['max_cnt']}_len_{param['item_len']}"

    def _gen_batch(
            self, batch_size,
            min_cnt, max_cnt,
            **kwargs,
    ):
        full_input_width = self.full_input_width
        full_output_width = self.full_output_width
        bit_width = self.bit_width
        item_len = self.item_len

        item_cnt_batch = self.rand.randint(min_cnt, max_cnt+1, size=batch_size)
        input_len_batch = item_cnt_batch * (item_len + 1) + 1
        total_len_batch = input_len_batch + 2 * (item_len + 1)
        max_len_batch = np.max(total_len_batch)

        inp = np.zeros((batch_size, max_len_batch, full_input_width))
        out = np.zeros((batch_size, max_len_batch, full_output_width))
        mask = np.zeros((batch_size, max_len_batch))

        for i in range(batch_size):
            item_cnt = item_cnt_batch[i]
            input_len = input_len_batch[i]
            total_len = total_len_batch[i]

            inp[i, :input_len, :bit_width] = np.random.binomial(1, 0.5, size=(input_len, bit_width)).astype(float)
            inp[i, :input_len-1:item_len+1, :-2] = 0
            inp[i, :input_len-1:item_len+1, -2] = 1

            # choose one block
            inp[i, input_len-1, -1] = 1.
            inp[i, input_len-1, :-1] = 0.
            inp[i, input_len + item_len, -1] = 1.
            idx = np.random.choice(item_cnt - 1)
            start = idx * (item_len + 1) + 1
            end = start + item_len
            inp[i, input_len:input_len+item_len, :bit_width] = inp[i, start:end, :bit_width]

            start = end + 1
            end = start + item_len
            out[i, input_len+item_len + 1:total_len - 1, :bit_width] = inp[i, start:end, :bit_width]
            out[i, total_len - 1, -1] = 1.

            mask[i, input_len+item_len+1:total_len] = 1.

        inp = torch.tensor(inp).float()
        out = torch.tensor(out).float()
        mask = torch.tensor(mask).float()

        return inp, out, mask

    def __iter__(self):
        while True:
            yield self._gen_batch(
                self.batch_size,
                self.min_cnt, self.max_cnt,
            )
