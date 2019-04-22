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
from tasks.rl_envs import *
from models.lstm import LSTM
from models.ntm import NTM
from rl_utils.q_learning import *


def validate(model, env, device):
    model.init_sequence(1, device)
    temp_env = env.reset()
    while not temp_env.finished:
        readed = np.eye(temp_env.len_alphabet)[temp_env.read()]
        action_probas = model.step(readed)
        _ = temp_env.step(action_probas.argmax())
    return temp_env.episode_total_reward


def optimize_model(model, memory, optimizer, loss, device, config, q_learning=q_learning, **q_params):
    transitions = memory.reverse_sample()
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    with torch.no_grad():
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = model(non_final_next_states.unsqueeze(1)).max(1)[0].detach()
    state_values = model(state_batch.unsqueeze(1)).gather(1, action_batch)

    true_state_values = q_learning(state_values, action_batch, reward_batch, next_state_values, **q_params)

    # Compute loss
    loss = loss(state_values, true_state_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                   config.optim.gradient_clipping)
    optimizer.step()

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
    if config.optimizer == 'adam':
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

    curricua = Curriculum(
        seed=config.seed,
        min_rep=config.task.min_len,
        max_rep=config.task.max_len,
        update_limit=config.task.update_limit)

    # fix 3 different (input, target) pairs for testing
    # also test generalization on longer sequences
    if config.train.report_interval:
        valid_env = copy(env).reset(task_len=config.validation.task_len)

    if config.train.loss == 'mse':
        loss = nn.MSELoss
    elif config.train.loss == 'mse_penalty':
        loss = mse_and_penalty
    else:
        raise ValueError('Unknown loss')

    memory = ReplayMemory(10000)
    iter_start_time = time.time()

    for i in range(config.train.train_iterations):
        model.train()
        memory.reset()

        episode_total_rewards = []
        for _ in range(config.train.batch_size):
            episode_total_rewards.append(save_episode(memory, curricua, env, model))
        episode_total_rewards = np.array(episode_total_rewards)

        optimize_model(model, memory, optimizer, loss, device, config)

        if i % config.train.verbose_interval == 0:
            time_now = time.time()
            time_per_iter = (
                time_now - iter_start_time) / config.train.verbose_interval * 1000.0
            loss_avg = loss_sum / config.train.verbose_interval
            cost_avg = cost_sum / config.train.verbose_interval

            message = f"Sequences: {i * config.train.batch_size}, Mean episode_total_rewards: {episode_total_rewards.mean()} "
            message += f"({time_per_iter:.2f} ms/iter)"
            logging.info(message)

            iter_start_time = time_now

        if i % config.train.report_interval == 0:
            logging.info('Validating model on longer sequences')
            model.eval()
            validation_rewards = []
            for _ in range(config.validation.iterations):
                # Store io plots in tensorboard
                env = valid_env.reset(task_len=config.validation.task_len)
                validation_rewards.append(validate(model, env, device))
                ex_output = env.output_panel
                ex_target = env.true_output

                fig = utils.plot_input_output(
                    ex_target[:, ex_target.shape[1] // 2 + 1:],
                    ex_output[:, ex_output.shape[1] // 2 + 1:],
                )
                writer.add_figure(
                    f"io/{ex_len}",
                    fig,
                    global_step=i)
            writer.add_scalar(
                'validation/reward', np.array(validation_rewards).mean(), global_step=i)

        if i % config.train.checkpoint_interval == 0:
            utils.save_checkpoint(model, config.checkpoints, loss.item(),
                                  episode_total_rewards.mean())

        # Write scalars to tensorboard
        writer.add_scalar(
            'train/loss', loss.item(), global_step=i)
        writer.add_scalar(
            'train/reward', episode_total_rewards.mean(), global_step=i)


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
