#!/usr/bin/env python3
import argparse
import datetime
import pathlib
import logging

import torch
import numpy as np
import tensorboardX

from models.ntm import create_ntm
from tasks.copytask import CopyDataLoader
# TODO implement dnc in pytorch


LOGGER = logging.getLogger(__name__)


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip_grads(net):
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        torch.nn.utils.clip_grad_value_(p, 10)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def train_batch(model, criterion, optimizer, x, y):
    """Trains a single batch."""
    model.train()
    optimizer.zero_grad()
    inp_seq_len = x.size(0)
    outp_seq_len, batch_size, _ = y.size()

    # New sequence
    model.init_sequence(batch_size)
    prev_state = model.create_new_state(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        _, prev_state = model(x[i], prev_state)

    # Collect sequence
    y_out = torch.zeros(y.size())
    x_inp = torch.zeros(batch_size, x.size(2))
    for i in range(outp_seq_len):
        y_out[i], prev_state = model(x_inp, prev_state)

    loss = criterion(y_out, y)
    loss.backward()
    clip_grads(model)
    optimizer.step()

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - y.data))

    return loss.item(), cost.item() / batch_size


def save_checkpoint(model, name, batch_num, checkpoint_path):
    progress_clean()
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_fname = f"{name}-batch-{batch_num}.model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(model.state_dict(), checkpoint_path/model_fname)


def train(model, criterion, optimizer, dataloader, args):
    """Train model generic function."""
    LOGGER.info("Training model for %d batches (batch_size=%d)...", args.num_batches, args.batch_size)
    start_time = datetime.datetime.now()
    writer = tensorboardX.SummaryWriter(log_dir=args.tensorboard_logs)
    mean_loss = 0.
    mean_cost = 0.

    for batch_num, x, y in dataloader.generate():
        if not args.no_gpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        loss, cost = train_batch(model, criterion, optimizer, x, y)
        mean_loss = (mean_loss * (batch_num - 1) + loss) / batch_num
        mean_cost = (mean_cost * (batch_num - 1) + cost) / batch_num

        progress_bar(batch_num, args.report_interval, loss)
        writer.add_scalar('loss', loss)
        writer.add_scalar('cost', cost)

        # Report
        if batch_num % args.report_interval == 0:
            LOGGER.info(
                "Batch %d Loss: %.6f Cost: %.2f Time: %d seconds",
                batch_num, mean_loss, mean_cost,
                (datetime.datetime.now() - start_time).total_seconds()
            )

        # Checkpoint
        if args.checkpoint_interval != 0 and batch_num % args.checkpoint_interval == 0:
            save_checkpoint(model, type(model).__name__, batch_num, args.checkpoint_path)

    LOGGER.info("Done training.")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')

    # Common training parameters
    parser.add_argument(
        '--seed', type=int, default=1000,
        help="Seed value for RNGs",
    )

    parser.add_argument(
        '--no-gpu', action='store_true',
        help='Do not use gpu for training',
    )

    parser.add_argument(
        '--task', action='store', choices=['copy'], default='copy',
        help="Choose the task to train (default: copy)"
    )

    parser.add_argument(
        '--report-interval', type=int, default=200,
        help="Reporting interval"
    )

    parser.add_argument(
        '--checkpoint-interval', type=int, default=1000,
        help="Checkpoint interval (default: 1000). "
        "Use 0 to disable checkpointing"
    )

    parser.add_argument(
        '--checkpoint-path', action='store', default='./',
        help="Path for saving checkpoint data (default: './')"
    )

    parser.add_argument(
        '--tensorboard-logs', type=str, action='store', default='./',
        help='Path for saving tensorboard logs (default: ./)'
    )

    # Task or model specific TODO better organize different tasks and models
    parser.add_argument('--num-batches', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--seq-width', type=int, default=8)
    parser.add_argument('--min-len', type=int, default=1)
    parser.add_argument('--max-len', type=int, default=20)
    parser.add_argument('--memory-n', type=int, default=128)
    parser.add_argument('--memory-m', type=int, default=20)

    args = parser.parse_args()
    return args


def init_logging():
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
        level=logging.DEBUG
    )


def main():
    init_logging()
    args = init_arguments()
    init_seed(args.seed)

    model = create_ntm(
        args.seq_width + 1, args.seq_width, args.memory_n, args.memory_m,
    )

    if not args.no_gpu and torch.cuda.is_available():
        model = model.cuda()

    dataloader = CopyDataLoader(
        args.num_batches,
        args.batch_size,
        args.seq_width,
        args.min_len,
        args.max_len,
    )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.95, momentum=0.9)

    LOGGER.info("Total number of parameters: %d", model.calculate_num_params())
    train(model, criterion, optimizer, dataloader, args)


if __name__ == "__main__":
    main()
