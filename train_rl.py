#!/usr/bin/env python3

import argparse
import logging
import pathlib
import time
import signal

import yaml
import tensorboardX
import torch
import torch.nn as nn
import torch.utils.data

import utils
from tasks.rl_envs import *
from models.lstm import LSTM
from models.ntm import NTM
from rl_utils.q_learning import *
from rl_utils.memory import *
from copy import copy
from sklearn.metrics import accuracy_score

def setup():
    parser = argparse.ArgumentParser(
        prog='Train/Eval script',
        description=
        ('Script for training and evaluating memory models on various bitmap tasks. '
         'All parameters should be given throug the config.json file.'),
    )
    parser.add_argument(
        '-n', '--name', type=str, help='Name of the current experiment.')

    args = parser.parse_args()
    assert args.name, 'Must supply a name of the experiment'
    path = pathlib.Path('experiments') / args.name
    assert (path / 'config.yaml').exists(), 'Must supply configuration file.'

    with open(path / 'config.yaml') as f:
        config = yaml.safe_load(f)

    (path / 'tensorboard').mkdir(exist_ok=True)
    (path / 'checkpoints').mkdir(exist_ok=True)

    config['path'] = path
    config['tensorboard'] = path / 'tensorboard'
    config['checkpoints'] = path / 'checkpoints'

    return utils.DotDict(config)


def train(config):
    # Setup manual seed
    torch.manual_seed(config.seed)
    device = torch.device(
        'cuda' if config.gpu and torch.cuda.is_available() else 'cpu')

    # Load Model
    if config.model.name == 'lstm':
        model = LSTM(
            n_inputs=config.task.len_alphabet,
            n_outputs=config.task.len_alphabet + 2,
            n_hidden=config.model.n_hidden,
            n_layers=config.model.n_layers,
        )
    elif config.model.name == 'ntm':
        model = NTM(
            input_size=config.task.len_alphabet,
            output_size=config.task.len_alphabet + 2,
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
    if config.optim.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optim.learning_rate,
            momentum=config.optim.momentum)
    if config.optim.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.optim.learning_rate,
            momentum=config.optim.momentum)
    if config.optim.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optim.learning_rate)

    # Load data
    if config.task.name == 'copy':
        env = CopyEnv(seed=config.seed, len_alphabet=config.task.len_alphabet, n_copies=config.task.n_copies)
    # elif config.task.name == 'repeat':
    #     dataset = RepeatCopyTask(bit_width=config.task.bit_width, seed=config.seed)
    else:
        raise Exception('Unknown task')

    writer = tensorboardX.SummaryWriter(log_dir=str(config.tensorboard))
    iter_start_time = time.time()
    loss_sum = 0.0
    cost_sum = 0.0

    curricua = Curriculum(config)

    # fix 3 different (input, target) pairs for testing
    # also test generalization on longer sequences
    if config.train.report_interval:
        valid_env = copy(env)

    if config.train.loss == 'mse':
        loss = nn.MSELoss()
    elif config.train.loss == 'mse_penalty':
        loss = mse_and_penalty
    else:
        raise ValueError('Unknown loss')

    iter_start_time = time.time()
    config.iteration = 0

    for i in range(config.train.train_iterations):
        model.train()

        episode_total_rewards = 0.
        total_loss = 0.
        for _ in range(config.train.batch_size):
            acc_loss, episode_total_reward = learn_episode(curricua, env, model, optimizer, loss, device, config)
            episode_total_rewards += episode_total_reward
            total_loss += acc_loss
        episode_total_rewards /= config.train.batch_size
        total_loss /= config.train.batch_size
        total_loss = float(total_loss)

        if i % config.train.verbose_interval == 0:
            time_now = time.time()
            time_per_iter = (
                time_now - iter_start_time) / config.train.verbose_interval * 1000.0
            loss_avg = total_loss / config.train.verbose_interval
            episode_total_rewards_avg = episode_total_rewards / config.train.verbose_interval

            message = f"Sequences: {i * config.train.batch_size}, Mean episode_total_rewards: {episode_total_rewards_avg}, loss_avg: {loss_avg}"
            message += f", epsilon: {temp_epsilon(config)} curr_temp_size: {curricua.temp_size} ({time_per_iter:.2f} ms/iter)"
            logging.info(message)

            iter_start_time = time_now

        if i % config.train.update_interval == 0:
            logging.info('Validating model on same-size sequences')
            model.eval()
            validation_rewards = 0.
            acc = 0.
            for _ in range(config.train.batch_size):
                env = valid_env.reset(len_input_seq=curricua.temp_size)
                validation_rewards += validate(model, env, device)
                acc += accuracy_score(env.true_output, env.output_panel)
            config = curricua.update(config, acc / config.train.batch_size)
            writer.add_scalar(
                'validation/reward', validation_rewards / config.train.batch_size, global_step=i)
            writer.add_scalar(
                'validation/accuracy', acc / config.train.batch_size, global_step=i)

        if i % config.train.report_interval == 0:
            logging.info('Validating model on longer sequences')
            model.eval()
            for len_val_seq in config.validation.len_val_seqs:
                validation_rewards = 0.
                acc = 0.
                for _ in range(config.validation.iterations):
                    # Store io plots in tensorboard
                    env = valid_env.reset(len_input_seq=len_val_seq)
                    validation_rewards += validate(model, env, device)
                    acc += accuracy_score(env.true_output, env.output_panel)
                    ex_output = env.output_panel
                    ex_target = env.true_output

                    fig = utils.plot_input_output(
                        np.array(ex_target).reshape(1, -1),
                        np.array(ex_output).reshape(1, -1),
                    )
                    writer.add_figure(
                        f"io/{len_val_seq}",
                        fig,
                        global_step=i)
                writer.add_scalar(
                    f'big_validation/{len_val_seq}/reward', validation_rewards / config.validation.iterations, global_step=i)
                writer.add_scalar(
                    f'big_validation/{len_val_seq}/accuracy', acc / config.validation.iterations, global_step=i)

        if i % config.train.checkpoint_interval == 0:
            utils.save_checkpoint(model, config.checkpoints, total_loss,
                                  episode_total_rewards)

        # Write scalars to tensorboard
        writer.add_scalar(
            'train/loss', total_loss, global_step=i)
        writer.add_scalar(
            'train/reward', episode_total_rewards, global_step=i)
        writer.add_scalar(
            'train/size', curricua.temp_size, global_step=i)

        global running
        if not running:
            break


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
    utils.set_logger(config['path'] / 'train.log')

    logging.info('Loaded config')
    for section in config.items():
        logging.info(section)

    logging.info('Start training')
    train(config)


if __name__ == "__main__":
    main()
