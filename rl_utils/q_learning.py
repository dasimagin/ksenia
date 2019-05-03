import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import random

def q_learning(state_values, action_batch, reward_batch, next_state_values, device, config):
    state_action_values = state_values.gather(1, action_batch)
    expected_state_action_values = (next_state_values.max(1)[0] * config.q_learning.gamma) + reward_batch
    true_state_values = state_action_values.clone().detach()
    true_state_values = (1 - config.q_learning.alpha) * state_action_values + config.q_learning.alpha * expected_state_action_values
    return state_action_values, true_state_values

def mse_l1(input, target):
    return ((input - target) ** 2 + torch.abs(input)).sum()

def mse_l2(input, target):
    return ((input - target) ** 2 + (input ** 2)).sum()

def mse_lsize(input, target, size):
    return ((input - target) ** 2 + (input - size) ** 2).sum()

def validate(model, env, device):
    model.init_sequence(1, device)
    temp_env = env.reset()
    while not temp_env.finished:
        readed = torch.eye(temp_env.len_alphabet, device=device)[temp_env.read()]
        action_probas = model.step(readed)
        _ = temp_env.step(action_probas.argmax())
    return temp_env.episode_total_reward

def temp_epsilon(config):
    return config.q_learning.eps_end + (config.q_learning.eps_start - config.q_learning.eps_end) * \
            math.exp(-1. * config.iteration / config.q_learning.eps_decay)

def select_action(action_probas, device, config):
    sample = random.random()
    eps_threshold = temp_epsilon(config)
    if sample > eps_threshold:
        with torch.no_grad():
            return action_probas.argmax(1)
    else:
        return torch.tensor([random.randrange(config.task.len_alphabet + 2)], device=device, dtype=torch.long)

def learn_episode(
         curricua,
         env,
         model,
         optimizer,
         loss,
         device,
         config,
         q_learning=q_learning
    ):
    task_len = curricua.sample()
    temp_env = env.reset(task_len)
    model.init_sequence(1, device)
    readed = torch.eye(temp_env.len_alphabet, device=device)[temp_env.read()]
    action_probas = model.step(readed)
    acc_loss = torch.zeros(1, device=device)
    while not temp_env.finished:
        action = select_action(action_probas.view(1, -1), device, config)
        reward = temp_env.step(action)
        new_readed = torch.eye(temp_env.len_alphabet, device=device)[temp_env.read()] if not temp_env.finished else None
        new_action_probas = model.step(new_readed) if new_readed is not None else torch.zeros(1, device=device)
        reward_tensored = torch.tensor([reward], device=device, dtype=torch.float32).view(1, -1)
        acc_loss += loss(*q_learning(action_probas.view(1, -1), action_probas.argmax().view(1, -1), reward_tensored, new_action_probas.view(1, -1), device, config))
        action_probas = new_action_probas
    optimizer.zero_grad()
    acc_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                   config.optim.gradient_clipping)
    optimizer.step()
    return acc_loss / len(temp_env.output_panel), temp_env.episode_total_reward / len(temp_env.output_panel)

def optimize_model(model, memory, optimizer, loss, device, config, q_learning=q_learning, **q_params):
    # TODO: fix it
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
        next_state_values = torch.zeros(len(reward_batch), device=device)
        next_state_values[non_final_mask] = model(non_final_next_states.unsqueeze(1)).max(1)[0].detach()
    state_values = model(state_batch.unsqueeze(1)).gather(1, action_batch)

    true_state_values = q_learning(state_values, action_batch, reward_batch, next_state_values, device, **q_params)

    # Compute loss
    batch_loss = loss(state_values, true_state_values)

    # Optimize the model
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                   config.optim.gradient_clipping)
    optimizer.step()
