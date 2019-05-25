#!/usr/bin/env python3

import logging
import time
import signal

import torch
import torch.utils.data
import tensorboardX

import utils
from tasks.bitmap import (
    CopyTask,
    RepeatCopyTask,
    AssociativeRecallTask,
)
from models.lstm import LSTM
from models.ntm import NTM
from models.dnc import DNC


def setup(config):
    if config.task.name == 'copy':
        task = CopyTask(
            batch_size=config.task.batch_size,
            min_len=config.task.min_len,
            max_len=config.task.max_len,
            bit_width=config.task.bit_width,
            seed=config.task.seed,
        )
    elif config.task.name == 'repeat':
        task = RepeatCopyTask(
            batch_size=config.task.batch_size,
            bit_width=config.task.bit_width,
            min_len=config.task.min_len,
            max_len=config.task.max_len,
            min_rep=config.task.min_rep,
            max_rep=config.task.max_rep,
            norm_max=config.task.norm_max,
            seed=config.task.seed,
        )
    elif config.task.name == 'recall':
        task = AssociativeRecallTask(
            batch_size=config.task.batch_size,
            bit_width=config.task.bit_width,
            item_len=config.task.item_len,
            min_cnt=config.task.min_cnt,
            max_cnt=config.task.max_cnt,
            seed=config.task.seed,
        )
    else:
        logging.info('Unknown task')
        exit(0)

    torch.manual_seed(config.model.seed)
    if config.model.name == 'lstm':
        model = LSTM(
            n_inputs=task.full_input_width,
            n_outputs=task.full_output_width,
            n_hidden=config.model.n_hidden,
            n_layers=config.model.n_layers,
        )
    elif config.model.name == 'ntm':
        model = NTM(
            input_size=task.full_input_width,
            output_size=task.full_output_width,
            mem_word_length=config.model.mem_word_length,
            mem_cells_count=config.model.mem_cells_count,
            n_writes=config.model.n_writes,
            n_reads=config.model.n_reads,
            controller_n_hidden=config.model.controller_n_hidden,
            controller_n_layers=config.model.controller_n_layers,
            clip_value=config.model.clip_value,
        )
    elif config.model.name == 'dnc':
        model = DNC(
            input_size=task.full_input_width,
            output_size=task.full_output_width,
            cell_width=config.model.cell_width,
            n_cells=config.model.n_cells,
            n_reads=config.model.n_reads,
            controller_n_hidden=config.model.controller_n_hidden,
            controller_n_layers=config.model.controller_n_layers,
            clip_value=config.model.clip_value,
            masking=config.model.masking,
            mask_min=config.model.mask_min,
            dealloc=config.model.dealloc,
            diff_alloc=config.model.diff_alloc,
            links=config.model.links,
            links_sharpening=config.model.links_sharpening,
            normalization=config.model.normalization,
            dropout=config.model.dropout,
        )
    else:
        logging.info('Unknown model')
        exit(0)

    if config.gpu and torch.cuda.is_available():
        model = model.cuda()

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

    step = 0
    if config.load:
        logging.info('Restoring model from checkpoint')
        model, optimizer, task, step = utils.load_checkpoint(
            model, optimizer, task, config.load,
        )

    return model, optimizer, task, step


def train(model, optimizer, task, step, config):
    writer = tensorboardX.SummaryWriter(logdir=str(config.tensorboard))
    criterion = task.loss
    iter_start_time = time.time()
    loss_sum = 0
    cost_sum = 0

    for i, (x, y, m) in enumerate(task, step + 1):
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

        torch.nn.utils.clip_grad_value_(model.parameters(), config.gradient_clipping)
        optimizer.step()

        pred_binarized = (pred.clone().data > 0).float()
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

        writer.add_scalar('train/loss', loss.item(), global_step=i * config.task.batch_size)
        writer.add_scalar('train/cost', cost.item(), global_step=i * config.task.batch_size)

        if i % config.evaluate_interval == 0:
            logging.info('Evaluating model')
            task.evaluate(model, i, writer, config)

            logging.info('Saving checkpoint')
            utils.save_checkpoint(
                model,
                optimizer,
                i, task,
                config.checkpoints,
            )

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

    config = utils.read_config()
    utils.set_logger(config.path/'train.log')

    with open(config.path/'config.yaml') as conf:
        config_text = conf.read()

    logging.info('Loaded config:\n')
    logging.info('='*30 + '\n')
    logging.info(config_text)
    logging.info('='*30 + '\n')
    logging.info('Start training')

    # Multiple training runs with different seeds
    if isinstance(config.model.seed, list):
        seeds = config.model.seed
    else:
        seeds = [config.model.seed]

    for i, seed in enumerate(seeds):
        logging.info('\n' + '='*30)
        logging.info('Running with seed %d', seed)
        logging.info('='*30 + '\n')

        config.model.seed = seed
        config.tensorboard = config.path/'tensorboard'/f'seed_{seed}'
        config.checkpoints = config.path/'checkpoints'/f'seed_{seed}'
        config.tensorboard.mkdir(exist_ok=True, parents=True)
        config.checkpoints.mkdir(exist_ok=True, parents=True)

        model, optimizer, task, step = setup(config)

        if i == 0:
            logging.info('Configured model')
            logging.info('Total number of parameters %d', model.calculate_num_params())

        train(model, optimizer, task, step, config)
        running = True


if __name__ == "__main__":
    main()
