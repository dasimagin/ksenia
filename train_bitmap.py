#!/usr/bin/env python3

import argparse
import logging
import pathlib
import time
import signal

import yaml
import tensorboardX
import torch
import torch.utils.data

import utils
from tasks.bitmap import RepeatCopyTask, CopyTask, BitBatchSampler
from models.lstm import LSTM
from models.ntm import NTM


def setup():
    parser = argparse.ArgumentParser(
        prog='Train/Eval script',
        description=('Script for training and evaluating memory models on various bitmap tasks. '
                     'All parameters should be given throug the config.json file.'),
    )
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        help='Name of the current experiment.'
    )

    args = parser.parse_args()
    assert args.name, 'Must supply a name of the experiment'
    path = pathlib.Path('experiments')/args.name
    assert (path/'config.yaml').exists(), 'Must supply configuration file.'

    with open(path/'config.yaml') as f:
        config = yaml.safe_load(f)

    (path/'tensorboard').mkdir(exist_ok=True)
    (path/'checkpoints').mkdir(exist_ok=True)

    config['path'] = path
    config['tensorboard'] = path/'tensorboard'
    config['checkpoints'] = path/'checkpoints'

    return utils.DotDict(config)


def train(config):
    # Setup manual seed
    torch.manual_seed(config.seed)
    device = torch.device('cuda' if config.gpu and torch.cuda.is_available() else 'cpu')

    # Load Model
    if config.model.name == 'lstm':
        model = LSTM(
            n_inputs=config.task.bit_width + 1,
            n_outputs=config.task.bit_width if config.task.name == 'copy' else config.task.bit_width + 1,
            n_hidden=config.model.n_hidden,
            n_layers=config.model.n_layers,
        )
    elif config.model.name == 'ntm':
        model = NTM(
            input_size=config.task.bit_width + 1,
            output_size=config.task.bit_width if config.task.name == 'copy' else config.task.bit_width + 1,
            mem_word_length=config.model.mem_word_length,
            mem_cells_count=config.model.mem_cells_count,
            n_writes=config.model.n_writes,
            n_reads=config.model.n_reads,
            controller_n_hidden=config.model.controller_n_hidden,
            controller_n_layers=config.model.controller_n_layers,
            controller_clip=config.model.controller_clip,
        )
    else:
        raise Exception('Unknown task')

    if config.gpu and torch.cuda.is_available():
        model = model.cuda()

    logging.info('Loaded model')
    logging.info('Total number of parameters %d', model.calculate_num_params())

    # Optimizers
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    if config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Load data
    if config.task.name == 'copy':
        dataset = CopyTask(bit_width=config.task.bit_width, seed=config.seed)
    elif config.task.name == 'repeat':
        dataset = RepeatCopyTask(bit_width=config.task.bit_width, seed=config.seed)
    else:
        raise Exception('Unknown task')

    batch_sampler = BitBatchSampler(
        batch_size=config.task.batch_size,
        min_len=config.task.min_len,
        max_len=config.task.max_len,
        min_rep=config.task.min_rep,
        max_rep=config.task.max_rep,
        seed=config.seed,
    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=config.gpu)
    writer = tensorboardX.SummaryWriter(log_dir=str(config.tensorboard))
    iter_start_time = time.time()
    loss_sum = 0.0
    cost_sum = 0.0

    # fix 3 different (input, target) pairs for testing
    # also test generalization on longer sequences
    if config.report_interval:
        valid_dataset = CopyTask(bit_width=config.task.bit_width, seed=config.seed)
        examples = [(valid_dataset[j], j) for j in (20, 40, 100)]

    for i, (x, y) in enumerate(data_loader, 1):
        model.train()
        seq_len, batch_size, seq_width = x.shape
        if config.gpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        pred = model(x)
        loss = dataset.loss(pred, y)
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        optimizer.step()

        pred_binarized = (pred.clone().data > 0.5).float()
        cost = torch.sum(torch.abs(pred_binarized - y.data)) / (seq_len * seq_width * batch_size)

        loss_sum += loss.item()
        cost_sum += cost.item()

        if i % config.verbose_interval == 0:
            time_now = time.time()
            time_per_iter = (time_now - iter_start_time) / config.verbose_interval * 1000.0
            loss_avg = loss_sum / config.verbose_interval
            cost_avg = cost_sum / config.verbose_interval

            message = f"Sequences: {i * config.task.batch_size}, loss: {loss_avg:.2f}, cost: {cost_avg:.2f} "
            message += f"({time_per_iter:.2f} ms/iter)"
            logging.info(message)

            iter_start_time = time_now
            loss_sum = 0.0
            cost_sum = 0.0

        if i % config.report_interval == 0:
            logging.info('Validating model on longer sequences')
            model.eval()
            for ex, ex_len in examples:
                # Store io plots in tensorboard
                ex_input, ex_target = ex
                ex_output = model(torch.tensor(ex_input, device=device).unsqueeze(0))
                ex_output = ex_output.detach().to('cpu').numpy()[0].T
                ex_target = ex_target.T

                fig = utils.plot_input_output(
                    ex_target[:, ex_target.shape[1] // 2 + 1:],
                    ex_output[:, ex_output.shape[1] // 2 + 1:],
                )
                writer.add_figure(f"io/{ex_len}", fig, global_step=i * config.task.batch_size)

        if i % config.checkpoint_interval == 0:
            utils.save_checkpoint(model, config.checkpoints, loss.item(), cost.item())

        # Write scalars to tensorboard
        writer.add_scalar('train/loss', loss.item(), global_step=i * config.task.batch_size)
        writer.add_scalar('train/cost', cost.item(), global_step=i * config.task.batch_size)

        # Stopping
        if not running:
            return
        if config.exit_after and i * config.task.batch_size > config.exit_after:
            return


def signal_handler(signal, frame):
    global running
    print('You pressed Ctrl+C!')
    running = False


def main():
    global running
    running = True
    signal.signal(signal.SIGINT, signal_handler)

    # Setup and load configuration from file
    config = setup()
    utils.set_logger(config['path']/'train.log')

    logging.info('Loaded config')
    for section in config.items():
        logging.info(section)

    logging.info('Start training')
    train(config)


if __name__ == "__main__":
    main()
