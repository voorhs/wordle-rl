import torch
import numpy as np

from collections import deque
import time
from datetime import datetime

from environment.environment import Environment
from dqn.agent import Agent


def train(
        env: Environment, agent: Agent,
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

    start_time = time.time()

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

        # progress print
        print(
            f'\nEpisode {i_episode:4d}',
            f'Recent Average Score: {score_recent:.2f}',
            f'Recent Success Rate: {100*success_recent:.1f}%',
            sep='\t'
        )

        if i_episode % 100 == 0:
            elapsed_time = time.time() - start_time
            print("Duration:", elapsed_time)

        # save net params
        if i_episode % 5000 == 0:
            torch.save(agent.qnetwork_local.state_dict(),
                       f'checkpoint{i_episode//5000}.pth')

    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)

    # save final net params
    timestamp = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
    torch.save(agent.qnetwork_local.state_dict(), 'net' + timestamp + '.pth')

    return scores, success
