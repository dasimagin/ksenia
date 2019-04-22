import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

def q_learning(state_values, action_batch, reward_batch, next_state_values, device, gamma=0.97, alpha=0.5):
    state_action_values = state_values.gather(1, action_batch)
    expected_state_action_values = (next_state_values.max(1)[0] * gamma) + reward_batch
    true_state_values = state_action_values.clone().detach()
    true_state_values = (1 - alpha) * state_action_values + alpha * expected_state_action_values
    return state_action_values, true_state_values

def mse_and_penalty(input, target):
    pass

def validate(model, env, device):
    model.init_sequence(1, device)
    temp_env = env.reset()
    while not temp_env.finished:
        readed = torch.eye(temp_env.len_alphabet)[temp_env.read()]
        action_probas = model.step(readed)
        _ = temp_env.step(action_probas.argmax())
    return temp_env.episode_total_reward

def learn_episode(
         curricua,
         env,
         model,
         optimizer,
         loss,
         device,
         config,
         q_learning=q_learning,
         **q_params
    ):
    task_len = curricua.sample()
    temp_env = env.reset(task_len)
    model.init_sequence(1, device)
    readed = torch.eye(temp_env.len_alphabet)[temp_env.read()]
    action_probas = model.step(readed)
    acc_loss = torch.zeros(1, device=device)
    while not temp_env.finished:
        reward = temp_env.step(action_probas.argmax())
        new_readed = torch.eye(temp_env.len_alphabet)[temp_env.read()] if not temp_env.finished else None
        new_action_probas = model.step(new_readed) if new_readed is not None else torch.zeros(1, device=device)
        acc_loss += loss(*q_learning(action_probas.view(1, -1), action_probas.argmax().view(1, -1), torch.Tensor([reward]).view(1, -1), new_action_probas.view(1, -1), device, **q_params))
        action_probas = new_action_probas
    optimizer.zero_grad()
    acc_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                   config.optim.gradient_clipping)
    optimizer.step()
    return acc_loss, temp_env.episode_total_reward

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
