#!/usr/bin/env python3

import argparse
import logging
import pathlib
import time
import signal
import shutil
import numpy as np

import yaml
import tensorboardX
import torch
import torch.utils.data

import utils
from tasks.arithmetic import Arithmetic
from models.lstm import LSTM
from models.ntm import NTM


def choose_complexity(min_len, max_len, cur_complexity):
    rnd = np.random.choice([0, 1, 2], p=[0.1, 0.25, 0.65])
    e = np.random.geometric(1/2)
    res = np.zeros(max_len - min_len + 1)
    if rnd == 0:
        res = 1 / (max_len - min_len + 1)
        return res
    elif rnd == 1:
        max_complexity = min(max_len - min_len + 1, cur_complexity + 1 + e)
        res[:max_complexity] = 1
        res /= res.sum()
        return res
    else:
        max_complexity = min(max_len - min_len, cur_complexity + e)
        res[max_complexity] = 1
        return res


def train(model, optimizer, criterion, train_data, validation_data, config):
    if config.scheduler is not None:
        optimizer, scheduler = optimizer

    writer = tensorboardX.SummaryWriter(log_dir=str(config.tensorboard))
    iter_start_time = time.time()
    loss_sum = 0
    cost_sum = 0
    cur_step = 0
    last_curriculum_update = 0
    cur_complexity = 0
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

        pred_idx = pred.data.max(-1)[1]
        true_idx = y.data.max(-1)[1]
        cost = ((pred_idx != true_idx).float() * m).sum() / batch_size

        loss_sum += loss.item()
        cost_sum += cost.item()
        
        val_tensors = []
        for (vx, vy, vm), vlength in validation_data:
            val_tensors.append((vx.cuda(), vy.cuda(), vm.cuda(), vlength))

        if i % config.verbose_interval == 0:
            time_now = time.time()
            time_per_iter = (time_now - iter_start_time) / config.verbose_interval * 1000.0
            loss_avg = loss_sum / config.verbose_interval
            cost_avg = cost_sum / config.verbose_interval

            message = f"Iter: {i}, Sequences: {cur_step}, "
            message += f"loss: {loss_avg:.2f}, cost: {cost_avg:.2f}, "
            message += f"({time_per_iter:.2f} ms/iter)"
            logging.info(message)

            iter_start_time = time_now
            loss_sum = 0
            cost_sum = 0

        if i % config.checkpoint_interval == 0:
            logging.info('Saving checkpoint')
            utils.save_checkpoint(
                model, 
                optimizer,
                cur_step,
                train_data,
                None,
                config.checkpoints)

            logging.info('Validating model on longer sequences')
            for vx, vy, vm, vlength in val_tensors:
                vpred = model(vx)
                vpred_idx = vpred.data.max(-1)[1]
                vtrue_idx = vy.data.max(-1)[1]
                vcost = ((vpred_idx != vtrue_idx).float() * vm).sum() / 50
                if vlength == 20:
                    vloss = criterion(vpred, vy, vm)
                    writer.add_scalar(f'val/loss{vlength}', vloss.item(), global_step=cur_step)
                writer.add_scalar(f'val/cost{vlength}', vcost.item(), global_step=cur_step)

        if config.scheduler is not None and i % config.scheduler.interval == 0:
            logging.info('Learning rate scheduler')
            scheduler.step(cost.item())

        # Write scalars to tensorboard
        writer.add_scalar('train/loss', loss.item(), global_step=cur_step)
        writer.add_scalar('train/cost', cost.item(), global_step=cur_step)
        if loss.item() < config.curriculum.threshold \
            and (i - last_curriculum_update) >= config.curriculum.update_step:
            cur_complexity += 1
            last_curriculum_update = i

        train_data.distribution = choose_complexity(train_data.min_len, train_data.max_len, cur_complexity)


        # Stopping
        if not running:
            return
        if config.exit_after and cur_step > config.exit_after:
            return
        
        cur_step += batch_size


def setup_model(config):
    # Load data
    if config.task.name == 'arithmetic':
        train_data = Arithmetic(
            batch_size=config.task.batch_size,
            min_len=config.task.min_len,
            max_len=config.task.max_len,
            task=config.task.task,
            seed=config.seed,
        )

        params = [20, 30, 40, 60]
        validation_data = []

        for length in params:
            example = train_data.gen_batch(
                batch_size=50,
                min_len=length, max_len=length,
                distribution=np.array([1,])
            )
            validation_data.append((example, length))
        loss = Arithmetic.loss
    else:
        logging.info('Unknown task')
        exit(0)

    # Setup model
    torch.manual_seed(config.seed)
    if config.model.name == 'lstm':
        model = LSTM(
            n_inputs=train_data.symbols_amount,
            n_outputs=train_data.symbols_amount,
            n_hidden=config.model.n_hidden,
            n_layers=config.model.n_layers,
        )
    elif config.model.name == 'ntm':
        model = NTM(
            input_size=train_data.symbols_amount,
            output_size=train_data.symbols_amount,
            mem_word_length=config.model.mem_word_length,
            mem_cells_count=config.model.mem_cells_count,
            n_writes=config.model.n_writes,
            n_reads=config.model.n_reads,
            controller_n_hidden=config.model.controller_n_hidden,
            controller_n_layers=config.model.controller_n_layers,
            clip_value=config.model.clip_value,
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
    print(config.path)
    print(config.tensorboard)

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
