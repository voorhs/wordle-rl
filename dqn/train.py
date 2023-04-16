# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/train_agent.py

import torch
import numpy as np

from collections import deque
from time import time
from datetime import datetime
from typing import List

from environment.environment import Environment
from dqn.agent import Agent

class Trainer:
    def __init__(
            self,
            env: List[Environment] | Environment, agent: Agent,
            logging_interval=None, checkpoint_interval=None,
            play_batch_size=8, is_parallel=True,
            n_batches=32, n_batches_warm=8,
        ):
        """
        Params
        ------
            logging_level (int): defines size of sliding window of batches to calculate and print stats from
            n_batches (int): number of batches to play and learn from during training
            n_batches_warm (int): size of initial experience to be collected before training
            play_batch_size (int): to strike a balance between the amount of incoming replays and size of the training batch
            is_parallel (bool): if True, then self.play_batch_parallel() is used, self.play_batch_successively() otherwise
            checkpoint_level (int): defines the interval of saving net params
        """
        if isinstance(env, list):
            self.env_list = env
        else:
            self.env = env
        self.agent = agent

        self.is_parallel = is_parallel
        self.n_batches = n_batches
        self.n_batches_warm = n_batches_warm
        self.play_batch_size = play_batch_size

        if checkpoint_interval is None:
            checkpoint_interval = n_batches
        self.checkpoint_interval = checkpoint_interval

        if logging_interval is None:
            logging_interval = max(n_batches // 8, 1)
        self.logging_interval = logging_interval

    def train(self, eps_start=1.0, eps_end=0.05, eps_decay=0.999, nickname=None):
        """Requires params to define exploration during training"""
        
        # for saving net checkpoints as "<nickname>-<num>.pth"
        if nickname is None:
            nickname = 'checkpoint' + datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
        self.nickname = nickname
        
        # for saving text reports of gameplays (self.test(log_game=True))
        self.t = 0

        # rewards and indicators of win; used for sliding window progress print
        self.scores = deque(maxlen=self.logging_interval)
        self.wins = deque(maxlen=self.logging_interval)

        self.start_time = time()

        if self.is_parallel:
            play_batch = self.play_batch_parallel
        else:
            play_batch = self.play_batch_successively

        # ======= COLLECT INITIAL EXPERIENCE =======

        # don't update net and 100% explore
        self.agent.eps = 1
        self.agent.eval = True
        for _ in range(self.n_batches_warm):
            play_batch()

        # ======= TRAINING =======

        # these guys are the main goal of training (because we aim to experiment with all methods)
        self.train_timers = []
        self.test_timers = []
        self.train_win_rates = []
        self.test_win_rates = []

        # slowly decreasing exporation
        self.agent.eps = eps_start

        for i_batch in range(1, self.n_batches+1):
            # collect batch of replays
            self.agent.eval = False
            batch_scores, batch_wins = play_batch()

            # decrease exploration chance
            self.agent.eps = max(eps_end, eps_decay * self.agent.eps)

            # collect statistics (update those "guys")
            elapsed_time = time() - self.start_time
            self.log_train(batch_scores, batch_wins, elapsed_time)
            self.agent.eval = True
            self.log_test(i_batch, elapsed_time)

            if i_batch % self.checkpoint_interval == 0:
                # save net params
                print('\nSaving checkpoint...', end=' ')
                num = i_batch // self.checkpoint_interval
                filename = f'{self.nickname}-{num}.pth'
                torch.save(self.agent.qnetwork_local.state_dict(), filename)
                print(f'Saved to {filename}')
        
        # return "guys"
        return self.train_timers, self.train_win_rates, self.test_timers, self.test_win_rates

    def play_batch_parallel(self):
        envs_number = len(self.env_list)
        
        state_size = self.env_list[0].state.size
        states = np.empty((envs_number, state_size))

        # reset all environments
        for env in self.env_list:
            env.reset()

        # to store stats
        batch_scores = np.zeros(envs_number)
        batch_wins = np.zeros(envs_number, dtype=bool)
        
        all_is_over = False
        while not all_is_over:
            # collect batch of states from envs that are not finished yet
            indexes = []
            for i, env in enumerate(self.env_list):
                if env.isover():
                    continue

                states[i] = env.state.value
                indexes.append(i)
            
            # feed batch to agent
            actions = self.agent.act_batch(states[indexes])

            all_is_over = True
            for i, action in zip(indexes, actions):
                # send action to env
                next_state, reward, done = self.env_list[i].step(action)

                # save replay to agent's buffer
                self.agent.add(states[i], action, reward, next_state, done)
            
                # collect stats
                batch_scores[i] += reward

                if done:
                    batch_wins[i] = self.env_list[i].wordle.win
                else:
                    all_is_over = False
        
        return batch_scores, batch_wins

    def play_batch_successively(self):
        """
        Play episodes to get at least `self.play_batch_size` of replays.
        
        It is necessary to strike a balance between the amount of
        incoming experience (replays) and the size of the training batch.
        - If there is less than one batch of experience, then the network
        will be trained on the same examples many times.
        - If there is more --- the network will simply not have time to learn
        from all the examples and the memory will be filled with examples that
        the network will never use for training.

        Return
        ------
            scores (list): total reward from each episode played
            wins (list): results of each episode
        """
        # to store stats
        batch_scores = []
        batch_wins = []

        # to track the amount of collected replays
        before = self.agent.memory.n_seen

        # play until batch of replays is collected
        while (self.agent.memory.n_seen - before < self.play_batch_size):
            
            # play single episode
            episode_score, episode_win = self.play_episode()
            
            # collect stats
            batch_scores.append(episode_score)
            batch_wins.append(episode_win)

        # return stats
        return batch_scores, batch_wins

    def play_episode(self):
        """
        Return
        ------
            score: total reward from this episode
            win: result of episode (True/False)
        """
        # to store stats
        score = 0

        # begin new episode
        state = self.env.reset()

        # until end of episode
        while not self.env.isover():

            # agent-environment interaction
            action = self.agent.act_single(state)
            next_state, reward, done = self.env.step(action)

            # collect replay
            self.agent.add(state, action, reward, next_state, done)

            # collect reward
            score += reward
            state = next_state
        
        # return stats
        return score, self.env.wordle.win

    def log_train(self, batch_scores, batch_wins, elapsed_time):
        # move sliding window
        self.scores.extend(batch_scores)
        self.wins.extend(batch_wins)

        self.train_timers.append(elapsed_time)
        self.train_win_rates.append(100 * np.mean(self.wins))

    def log_test(self, i_batch, elapsed_time):
        if i_batch % self.logging_interval != 0:
            return
        
        # test agent without exploration
        eps = self.agent.eps

        self.agent.eps = 0
        test_scores, test_win_rate, mean_steps = self.test(return_result=True)
        self.agent.eps = eps

        self.test_timers.append(elapsed_time)
        self.test_win_rates.append(test_win_rate)

        print(
            f'\nBatch {i_batch:4d}',
            f'Time: {elapsed_time:.0f} s',
            # f'RMSE: {self.agent.loss:.4f}' if self.agent.loss else f'RMSE: None',
            f'Agent Eps: {self.agent.eps:.2f}',
            # f'Train Score: {np.mean(self.scores):.2f}',
            f'Train Win Rate: {self.train_win_rates[-1]:.2f}%',
            # f'Test Score: {test_scores:.2f}',
            f'Test Win Rate: {test_win_rate:.2f}%',
            f'Test Mean Steps: {mean_steps:.2f}',
            sep='\t'
        )

    def test(self, log_game=True, return_result=False):
        env = None
        if hasattr(self, 'env_list'):
            env = self.env_list[0]
        else:
            env = self.env

        output = None
        if log_game:
            # output file to show how agent played (text report of gameplay)
            output = f'{self.nickname}-{self.t}.txt'

        # number of test words
        n_episodes = len(env.wordle.answers)

        # rewards and indicators of success
        scores, success, steps = np.zeros((3, n_episodes))
        success_count = 0

        # hard reset pre-generated answers list
        env.wordle.current_answer = -1

        # run through all answers
        for i_episode in range(n_episodes):

            # begin new episode
            state = env.reset(replace=False)
            
            if log_game:
                with open(output, 'a') as f:
                    f.write(f'Episode {i_episode+1}\tAnswer {state.answer}\n')

            # until end of episode
            while not env.isover():

                # agent-environment interaction (`output` stands for printing report to txt file)
                action = self.agent.act_single(state)
                next_state, reward, _ = env.step(action, output)

                # collect statistics
                scores[i_episode] += reward
                steps[i_episode] += 1

                state = next_state

            if log_game:
                # end of episode
                with open(output, 'a') as f:
                    f.write(f"Score {scores[i_episode]}\t{'WIN' if env.wordle.win else 'LOSE'}\n\n")

            # collect statistics
            success[i_episode] = env.wordle.win
            success_count += env.wordle.win

        # final statistics
        scores = np.mean(scores)
        win_rate = 100 * success_count / n_episodes
        mean_steps = steps[success.astype(bool)].mean()
        
        if log_game:
            # end of episode
            with open(output, 'a') as f:
                f.write('\n'.join([
                        f'Test Win Rate: {success_count} / {n_episodes} ({win_rate:.2f}%)',
                        f'Test Mean Steps: {mean_steps:.2f}',
                        f'Test Score: {scores}'
                ]))

        self.t += 1
        if return_result:
            return scores, win_rate, mean_steps
