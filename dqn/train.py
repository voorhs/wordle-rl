# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/train_agent.py

import torch
import numpy as np

from collections import deque
from time import time
from datetime import datetime

from environment.environment import Environment
from dqn.agent import Agent


def train(
        env: Environment, agent: Agent, logging_level=100,
        n_episodes=2000, eps_start=1.0, eps_end=0.001, eps_decay=0.995):
    """
    Params
    ------
        n_episodes (int):   maximum number of training episodes
        eps_start (float):  starting value of epsilon, for epsilon-greedy action selection
        eps_end (float):    minimum value of epsilon
        eps_decay (float):  multiplicative factor (per episode) for decreasing epsilon
    """
    # rewards
    scores = []
    scores_window = deque(maxlen=100)

    # indicators of success
    success = []
    success_window = deque(maxlen=100)

    # exploration chance
    eps = eps_start

    start_time = time()

    for i_episode in range(1, n_episodes+1):
        # begin new episode
        state = env.reset()
        score = 0

        # until end of episode
        while not env.isover():
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
        scores_window.append(score)
        score_recent = np.mean(scores_window)

        success.append(env.wordle.win)
        success_window.append(env.wordle.win)
        success_recent = np.mean(success_window)

        # decrease exploration chance
        eps = max(eps_end, eps_decay * eps)

        if i_episode % logging_level == 0:
            # progress print
            elapsed_time = time() - start_time
            print(
                f'\nEpisode {i_episode:4d}',
                f'Last {logging_level} Score: {score_recent:.2f}',
                f'Last {logging_level} Success Rate: {100*success_recent:.1f}%',
                f'Duration: {elapsed_time:.1f}',
                sep='\t'
            )

        # save net params
        if i_episode % 5000 == 0:
            torch.save(agent.qnetwork_local.state_dict(),
                       f'checkpoint{i_episode//5000}.pth')

    elapsed_time = time() - start_time
    print('\n=====================================')
    print(
        f'\nAverage Score: {np.mean(score):.2f}',
        f'Success Rate: {100*success_recent:.1f}%',
        f'Duration: {elapsed_time:.1f}',
        sep='\t'
    )

    # save final net params
    timestamp = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
    torch.save(agent.qnetwork_local.state_dict(), 'net' + timestamp + '.pth')

    return scores, success


def test(env: Environment, agent: Agent):
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

        if i_episode % 100 == 0:
            # progress print
            elapsed_time = time() - start_time
            print(
                f'\nWords Tested {i_episode+1:4d} / {n_episodes:4d}',
                f'Success Rate: {success_count:4d} / {n_episodes:4d}',
                f'Duration: {elapsed_time:.1f}',
                sep='\t'
            )

    elapsed_time = time() - start_time

    print('\n===============================')
    print(
        f'\nSuccess Rate: {success_count:4d} / {n_episodes:4d} ({100*success_count/n_episodes:.4f}%)',
        f'Mean steps number: {steps[success.astype(bool)].mean():.4f}',
        f'Elapsed Time: {elapsed_time:.1f} s',
        sep='\t'
    )

    return scores, success
