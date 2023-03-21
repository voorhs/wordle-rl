# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/train_agent.py

import torch
import numpy as np

from collections import deque
from time import time
from datetime import datetime

from environment.environment import Environment
from dqn.agent import Agent


def train(
        env: Environment, agent: Agent, logging_level=None,
        n_episodes=2000, eps_start=1.0, eps_end=0.001, eps_decay=0.995,
        end_of_episode=None, nickname=None):
    """
    Params
    ------
        n_episodes (int):   maximum number of training episodes
        eps_start (float):  starting value of epsilon, for epsilon-greedy action selection
        eps_end (float):    minimum value of epsilon
        eps_decay (float):  multiplicative factor (per episode) for decreasing epsilon
    """
    # to leave space for customizing end of episode
    if end_of_episode is None:
       end_of_episode = lambda env: env.isover()
    
    # for saving net checkpoints as "<nickname>-<i>-<timestamp>.pth"
    if nickname is None:
        nickname = 'checkpoint'
    timestamp = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
    
    # logging_level defines size of sliding window of last episodes
    # to calculate statistics from and display stats during training
    if logging_level is None:
        logging_level = n_episodes // 10

    # rewards
    scores = deque(maxlen=logging_level)

    # indicators of success
    successes = deque(maxlen=logging_level)

    # for "time" vs "win rate" testing
    timers = []
    win_rates = []

    # exploration chance
    eps = eps_start

    start_time = time()
    
    # collect enough experience before training
    print(f'Playing {agent.n_warm} initial games (warm start)...', end=' ')
    for i_episode in range(agent.n_warm):
        # begin new episode
        state = env.reset()
        score = 0

        # until end of episode
        while not end_of_episode(env):
            # agent-environment interaction
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)

            # collect replay and learn from buffer
            agent.step(state, action, reward, next_state, done)

            # collect reward and
            score += reward
            state = next_state
    print(f'Collected! Time: {time() - start_time:.1f} s')

    # begin training
    for i_episode in range(1, n_episodes+1):
        # begin new episode
        state = env.reset()
        score = 0

        # until end of episode
        while not end_of_episode(env):
            # agent-environment interaction
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)

            # collect replay and learn from buffer
            agent.step(state, action, reward, next_state, done)

            # collect reward and
            score += reward
            state = next_state

        # collect statistics

        scores.append(score)
        score = np.mean(scores)

        successes.append(env.wordle.win)
        success = np.mean(successes)

        elapsed_time = time() - start_time
        timers.append(elapsed_time)
        win_rates.append(success)

        # decrease exploration chance
        eps = max(eps_end, eps_decay * eps)

        if i_episode % logging_level == 0:
            # progress print
            print(
                f'\nEpisode {i_episode:4d}',
                f'Score: {score:.2f}',
                f'Success Rate: {100*success:.1f}%',
                f'RMSE: {agent.loss:.3f}' if agent.loss else f'RMSE: None',
                f'Time: {elapsed_time:.1f} s',
                sep='\t'
            )

        # save net params
        if i_episode % 5000 == 0:
            print('\nSaving checkpoint...', end=' ')
            num = i_episode // 5000
            filename = f'{nickname}-{num}-{timestamp}.pth'
            torch.save(agent.qnetwork_local.state_dict(), filename)
            print(f'Saved to {filename}')

    # save final net params
    timestamp = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
    torch.save(agent.qnetwork_local.state_dict(), nickname + '-final-' + timestamp + '.pth')

    return timers, win_rates


def test(env: Environment, agent: Agent, return_result=False):
    # output file for collecting episodes in text format
    timestamp = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
    output = 'test' + timestamp + '.txt'

    # number of test words
    n_episodes = len(env.wordle.answers)

    # rewards and indicators of success
    scores, success, steps = np.zeros((3, n_episodes))
    success_count = 0

    # hard reset pre-generated answers list
    env.wordle.current_answer = -1

    start_time = time()

    # run through all answers
    for i_episode in range(n_episodes):

        # begin new episode
        state = env.reset(replace=False)
        with open(output, 'a') as f:
            f.write(f'Episode {i_episode+1}\tAnswer {state.answer}\n')

        # until end of episode
        while not env.isover():
            # agent-environment interaction (prints guess with pattern)
            action = agent.act(state)
            next_state, reward, _ = env.step(action, output)

            # collect statistics
            scores[i_episode] += reward
            steps[i_episode] += 1

            state = next_state

        with open(output, 'a') as f:
            f.write(f"Score {scores[i_episode]}\t{'WIN' if env.wordle.win else 'LOSE'}\n\n")

        # collect statistics
        success[i_episode] = env.wordle.win
        success_count += env.wordle.win

    elapsed_time = time() - start_time

    # final statistics
    message = '\t'.join([
        f'Success: {success_count} / {n_episodes} ({100*success_count/n_episodes:.4f}%)',
        f'Steps: {steps[success.astype(bool)].mean():.4f}',
        f'Time: {elapsed_time:.1f} s',
        f'Saved to: {output}\n'
    ])

    # print in console and in txt file
    print('\n' + message)
    with open(output, 'a') as f:
        f.write(message)
        f.write(f'\nscores: {scores}\nsucesses: {success}')

    if return_result:
        return scores, success
