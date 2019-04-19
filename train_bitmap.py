#!/usr/bin/env python3

import argparse
import logging
import pathlib
import time
import signal
import shutil

import yaml
import tensorboardX
import torch
import torch.utils.data

import utils
from tasks.bitmap import CopyTask, RepeatCopyTask
from models.lstm import LSTM
from models.ntm import NTM


def train(model, optimizer, criterion, train_data, validation_data, config):
    if config.scheduler is not None:
        optimizer, scheduler = optimizer

    writer = tensorboardX.SummaryWriter(log_dir=str(config.tensorboard))
    iter_start_time = time.time()
    loss_sum = 0
    cost_sum = 0

    for i, (x, y, m) in enumerate(train_data, 1):
        model.train()
        batch_size, seq_len, seq_width = x.shape
        if config.gpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            m = m.cuda()

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y, m)
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        optimizer.step()

        pred_binarized = (pred.clone().data > 0.5).float()
        cost_time_batch = torch.sum(torch.abs(pred_binarized - y.data), dim=-1)
        cost_batch = torch.sum(cost_time_batch * m, dim=-1)
        cost = cost_batch.sum() / batch_size

        loss_sum += loss.item()
        cost_sum += cost.item()

        if i % config.verbose_interval == 0:
            time_now = time.time()
            time_per_iter = (time_now - iter_start_time) / config.verbose_interval * 1000.0
            loss_avg = loss_sum / config.verbose_interval
            cost_avg = cost_sum / config.verbose_interval

            message = f"Iter: {i}, Sequences: {i * config.task.batch_size}, "
            message += f"loss: {loss_avg:.2f}, cost: {cost_avg:.2f}, "
            message += f"({time_per_iter:.2f} ms/iter)"
            logging.info(message)

            iter_start_time = time_now
            loss_sum = 0
            cost_sum = 0

        if i % config.checkpoint_interval == 0:
            logging.info('Saving checkpoint')
            utils.save_checkpoint(model, config.checkpoints, loss.item(), cost.item())

            logging.info('Validating model on longer sequences')
            evaluate(i, model, validation_data, writer, config)

        if config.scheduler is not None and i % config.scheduler.interval == 0:
            logging.info('Learning rate scheduler')
            scheduler.step(cost.item())

        # Write scalars to tensorboard
        writer.add_scalar('train/loss', loss.item(), global_step=i * config.task.batch_size)
        writer.add_scalar('train/cost', cost.item(), global_step=i * config.task.batch_size)

        # Stopping
        if not running:
            return
        if config.exit_after and i * config.task.batch_size > config.exit_after:
            return


def evaluate(step, model, data, writer, config):
    model.eval()
    device = 'cuda' if config.gpu and torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for instance in data:
            if config.task.name == 'copy':
                example, length = instance
                start = length + 1
                name = f"io/len_{length}"
            elif config.task.name == 'repeat':
                example, length, num_rep = instance
                start = length + 2
                name = f"io/len_{length}_rep_{num_rep}"

            inp, target, mask = example
            output = model(inp.to(device))
            output = output.data.to('cpu').numpy()[0].T
            target = target.data.numpy()[0].T

            img = utils.input_output_img(
                target[:, start:],
                output[:, start:],
            )

            writer.add_image(
                name,
                img, global_step=step * config.task.batch_size
            )


def setup_model(config):
    # Load data
    if config.task.name == 'copy':
        train_data = CopyTask(
            batch_size=config.task.batch_size,
            min_len=config.task.min_len,
            max_len=config.task.max_len,
            bit_width=config.task.bit_width,
            seed=config.task.seed,
        )

        params = [20, 40, 100]
        validation_data = []

        for length in params:
            example = train_data.gen_batch(
                batch_size=1,
                min_len=length, max_len=length,
            )
            validation_data.append((example, length))
        loss = CopyTask.loss
    elif config.task.name == 'repeat':
        train_data = RepeatCopyTask(
            batch_size=config.task.batch_size,
            bit_width=config.task.bit_width,
            min_len=config.task.min_len,
            max_len=config.task.max_len,
            min_rep=config.task.min_rep,
            max_rep=config.task.max_rep,
            norm_max=10,
            seed=config.task.seed,
        )

        params = [(10, 10), (20, 10), (10, 20)]
        validation_data = []

        for length, num_rep in params:
            example = train_data.gen_batch(
                batch_size=1,
                min_len=length, max_len=length,
                min_rep=num_rep, max_rep=num_rep,
            )

            validation_data.append((example, length, num_rep))
        loss = RepeatCopyTask.loss
    else:
        logging.info('Unknown task')
        exit(0)

    # Setup model
    torch.manual_seed(config.model.seed)
    if config.model.name == 'lstm':
        model = LSTM(
            n_inputs=train_data.full_input_width,
            n_outputs=train_data.full_output_width,
            n_hidden=config.model.n_hidden,
            n_layers=config.model.n_layers,
        )
    elif config.model.name == 'ntm':
        model = NTM(
            input_size=train_data.full_input_width,
            output_size=train_data.full_output_width,
            mem_word_length=config.model.mem_word_length,
            mem_cells_count=config.model.mem_cells_count,
            n_writes=config.model.n_writes,
            n_reads=config.model.n_reads,
            controller_n_hidden=config.model.controller_n_hidden,
            controller_n_layers=config.model.controller_n_layers,
            controller_clip=config.model.controller_clip,
        )
    else:
        logging.info('Unknown model')
        exit(0)

    if config.gpu and torch.cuda.is_available():
        model = model.cuda()

    logging.info('Loaded model')
    logging.info('Total number of parameters %d', model.calculate_num_params())

    # Setup optimizer
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )
    if config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
        )
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

    if config.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            verbose=config.scheduler.verbose,
            threshold=config.scheduler.threshold,
        )
        optimizer = (optimizer, scheduler)

    return model, optimizer, loss, train_data, validation_data


def read_config():
    parser = argparse.ArgumentParser(
        prog='Train/Eval script',
        description=('Script for training and evaluating memory models on various bitmap tasks. '
                     'All parameters should be given throug the config file.'),
    )
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        required=True,
        help='Name of the current experiment. Can also provide name/with/path for grouping'
    )
    parser.add_argument(
        '-k',
        '--keep',
        action='store_true',
        help='Keep logs from previous run.'
    )

    args = parser.parse_args()
    path = pathlib.Path('experiments')/args.name

    assert args.name, f'No such directory: {str(path)}.'
    assert (path/'config.yaml').exists(), 'No configuration file found.'

    with open(path/'config.yaml') as f:
        config = utils.DotDict(yaml.safe_load(f))

    if not args.keep:
        (path/'tensorboard').exists() and shutil.rmtree(path/'tensorboard')
        (path/'checkpoints').exists() and shutil.rmtree(path/'checkpoints')
        open(path/'train.log', 'w').close()

    (path/'tensorboard').mkdir(exist_ok=True)
    (path/'checkpoints').mkdir(exist_ok=True)

    config.path = path
    config.tensorboard = path/'tensorboard'
    config.checkpoints = path/'checkpoints'

    return config


def signal_handler(signal, frame):
    global running
    print('You pressed Ctrl+C!')
    running = False


def main():
    global running
    running = True
    signal.signal(signal.SIGINT, signal_handler)

    config = read_config()
    utils.set_logger(config.path/'train.log')

    logging.info('Loaded config:\n')
    logging.info('=' * 30 + '\n')
    with open(config.path/'config.yaml') as conf:
        logging.info(conf.read())
    logging.info('=' * 30 + '\n')
    logging.info('Start training')

    model, optimizer, loss, train_data, validation_data = setup_model(config)
    train(model, optimizer, loss, train_data, validation_data, config)


if __name__ == "__main__":
    main()
