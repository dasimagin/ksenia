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
    if config.model == 'lstm':
        model = LSTM(
            n_inputs=config.bit_width + 1,
            n_outputs=config.bit_width if config.task == 'copy' else config.bit_width + 1,
            n_hidden=config.n_hidden,
            n_layers=config.n_layers,
        )

    if config.gpu and torch.cuda.is_available():
        model = model.cuda()

    logging.info('Loaded model')
    logging.info('Total number of parameters %d', model.calculate_num_params())

    # Optimizers
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    if config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=config.momentum)
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Load data
    if config.task == 'copy':
        dataset = CopyTask(bit_width=config.bit_width, seed=config.seed)
    if config.task == 'repeat':
        dataset = RepeatCopyTask(bit_width=config.bit_width, seed=config.seed)

    batch_sampler = BitBatchSampler(
        batch_size=config.batch_size,
        min_len=config.min_len,
        max_len=config.max_len,
        min_rep=config.min_rep,
        max_rep=config.max_rep,
        seed=config.seed,
    )

    iter_start_time = time.time()
    writer = tensorboardX.SummaryWriter(log_dir=str(config.tensorboard))
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=config.gpu)

    # fix 3 different (input, target) pairs for testing
    # also test generalization on longer sequences
    if config.valid_interval:
        valid_dataset = CopyTask(bit_width=config.bit_width, seed=config.seed)
        # TODO add validation cost to tensorboard
        # valid_batch_sampler = BitBatchSampler(
        #     batch_size=config.batch_size,
        #     min_len=21,  # TODO add config params
        #     max_len=100,
        #     min_rep=1,
        #     max_rep=1,
        #     seed=config.seed,
        # )
        # valid_data_loader = torch.utils.data.DataLoader(
        #     valid_dataset, batch_sampler=valid_batch_sampler, pin_memory=config.gpu)

        # using length hack
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        pred_binarized = (pred.clone().data > 0.5).float()
        cost = torch.sum(torch.abs(pred_binarized - y.data)) / (seq_len * seq_width * batch_size)

        if i % config.info_interval == 0:
            time_now = time.time()
            time_per_iter = (time_now - iter_start_time) / config.info_interval * 1000.0
            message = f"Sequences: {i * config.batch_size}, loss: {loss.item():.2f}, cost: {cost.item():.2f} "
            message += f"({time_per_iter:.2f} ms/iter)"
            iter_start_time = time_now

            logging.info(message)

        if i % config.valid_interval == 0:
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
                writer.add_figure(f"io/{ex_len}", fig, global_step=i * config.batch_size)

        if i % config.save_interval == 0:
            utils.save_checkpoint(model, config.checkpoints, loss.item(), cost.item())

        # Write scalars to tensorboard
        writer.add_scalar('train/loss', loss.item(), global_step=i * config.batch_size)
        writer.add_scalar('train/cost', cost.item(), global_step=i * config.batch_size)

        # Stopping
        if not running:
            return
        if config.exit_after and i * config.batch_size > config.exit_after:
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
