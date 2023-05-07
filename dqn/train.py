# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/train_agent.py

import torch
import numpy as np

from tqdm.notebook import tqdm_notebook as tqdm
from collections import deque
from functools import partial
from time import time
from datetime import datetime
import os
from typing import List, Union

from environment.environment import Environment, StateYesNo
from environment.action import ActionCombLetters, ActionWagons, ActionVocabulary
from wordle.wordlenp import Wordle
from replay_buffer.cpprb import ReplayBuffer, PrioritizedReplayBuffer
from dqn.agent import Agent
from dqn.model import SkipConnectionQNetwork


class Trainer:
    def __init__(
            self, agent: Agent,
            train_env: Union[List[Environment], Environment],
            test_env: Environment,
            logging_interval=None,
            n_batches=32, n_batches_warm=8, play_batch_size=8, is_parallel=True,
            nickname=None, test_first=True
        ):
        """
        Params
        ------
            logging_interval    (int): defines size of sliding window of batches to calculate and print stats from
            checkpoint_interval (int): defines the interval for saving agent's net params
            n_batches           (int): number of batches to play and learn from during training
            n_batches_warm      (int): size of initial experience to be collected before training
            play_batch_size     (int): to strike a balance between the amount of incoming replays and size of the training batch
            is_parallel        (bool): if True, then self.play_batch_parallel() is used, self.play_batch_successively() otherwise
            nickname            (str): directory name for saving checkpoints and other files
        """
        self.agent = agent

        if isinstance(train_env, list):
            self.train_env_list = train_env
        else:
            self.train_env = train_env
        self.test_env = test_env

        self.is_parallel = is_parallel
        self.n_batches = n_batches
        self.n_batches_warm = n_batches_warm
        self.play_batch_size = play_batch_size

        if logging_interval is None:
            logging_interval = max(n_batches // 8, 1)
        self.logging_interval = logging_interval

        if self.is_parallel:
            self.play_batch = self.play_batch_parallel
        else:
            self.play_batch = self.play_batch_successively
        
        # create directory named `nickname`
        if nickname is None:
            nickname = datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")
        self.nickname = nickname
        i = 1
        while os.path.exists(self.nickname):
            if self.nickname.endswith(f'-{i-1}'):
                self.nickname = self.nickname.replace(f'-{i-1}', f'-{i}')
            elif i == 1:
                self.nickname += f'-{i}'
            i += 1

        os.mkdir(self.nickname)

        self.test_first = test_first
        self.train_output = None

    def train(self, eps_start=1.0, eps_end=0.05, eps_decay=0.999):
        """Requires params to define exploration during training"""
        # for saving text reports of gameplays (self.test(log_game=True))
        self.t = 0

        # rewards and indicators of win; used for sliding window progress print
        self.scores = deque(maxlen=self.logging_interval)
        self.wins = deque(maxlen=self.logging_interval)

        self.start_time = time()

        # ======= COLLECT INITIAL EXPERIENCE =======

        # don't update net and 100% explore
        self.agent.eps = 1
        self.agent.eval = True
        for _ in tqdm(range(self.n_batches_warm), desc='WARM BATCHES'):
            self.play_batch()

        # ======= TRAINING =======

        # these guys are the main goal of training (because we aim to experiment with all methods)
        self.train_timers = []
        self.test_timers = []
        self.train_win_rates = []
        self.test_win_rates = []

        if self.test_first:
            self.agent.eval = True
            self.log_test(0, 0)

        # slowly decreasing exporation
        self.agent.eps = eps_start

        for i_batch in tqdm(range(1, self.n_batches+1), desc='TRAIN BATCHES'):
            # collect batch of replays
            self.agent.eval = False
            batch_scores, batch_wins = self.play_batch()

            # decrease exploration chance
            self.agent.eps = max(eps_end, eps_decay * self.agent.eps)

            # collect statistics (update those "guys")
            elapsed_time = time() - self.start_time
            self.save_checkpoint(i_batch)
            self.log_train(batch_scores, batch_wins, elapsed_time)
            self.agent.eval = True
            self.log_test(i_batch, elapsed_time)
        
        # return "guys"
        return self.train_timers, self.train_win_rates, self.test_timers, self.test_win_rates

    def save_checkpoint(self, i_batch):
        if i_batch % self.logging_interval != 0:
            return
            
        self.agent.dump(self.nickname, self.t) 

    def play_batch_parallel(self):
        envs_number = len(self.train_env_list)
        
        state_size = self.train_env_list[0].state.size
        states = np.empty((envs_number, state_size))

        # reset all environments
        for env in self.train_env_list:
            env.reset()

        # to store stats
        batch_scores = np.zeros(envs_number)
        batch_wins = np.zeros(envs_number, dtype=bool)
        
        all_is_over = False
        while not all_is_over:
            # collect batch of states from envs that are not finished yet
            indexes = []
            for i, env in enumerate(self.train_env_list):
                if env.isover():
                    continue

                states[i] = env.state.value
                indexes.append(i)
            
            # feed batch to agent
            actions = self.agent.act_batch(states[indexes])

            all_is_over = True
            for i, action in zip(indexes, actions):
                # send action to env
                next_state, reward, done = self.train_env_list[i].step(action, self.train_output)

                # save replay to agent's buffer
                self.agent.add(states[i], action, reward, next_state, done)
            
                # collect stats
                batch_scores[i] += reward

                if done:
                    batch_wins[i] = self.train_env_list[i].wordle.win
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
        self.success, test_win_rate, mean_steps = self.test(return_result=True)
        self.agent.eps = eps

        self.test_timers.append(elapsed_time)
        self.test_win_rates.append(test_win_rate)

        if i_batch == 0:
            print(
                f'\nBatch {i_batch:4d}',
                f'Test Win Rate: {test_win_rate:.2f}%',
                f'Test Mean Steps: {mean_steps:.2f}',
                sep='\t'
            )
        else:
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
            self.t += 1

        self.train_output = f'{self.nickname}/train-{self.t}.txt'

    def test(self, log_game=True, return_result=False):
        env = None
        if hasattr(self, 'test_env_list'):
            env = self.test_env_list[0]
        else:
            env = self.test_env

        output = None
        if log_game:
            # output file to show how agent played (text report of gameplay)
            output = f'{self.nickname}/test-{self.t}.txt'

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
            state = env.reset(for_test=True)

            # until end of episode
            while not env.isover():

                # agent-environment interaction (`output` stands for printing report to txt file)
                action = self.agent.act_single(state)
                next_state, reward, _ = env.step(action, output)

                # collect statistics
                scores[i_episode] += reward
                steps[i_episode] += 1

                state = next_state

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

        if return_result:
            return success, win_rate, mean_steps

    def train_comb_letters(
        rewards, ohe1, ohe2, wordle_list, tasks_results, rb_size,
        eps_start, eps_end, eps_decay, lr=5e-4, alpha=0, k=1,
        logging_interval=None, method_name=None, fine_tune=False, backbone_path=None, *,
        data, n_envs, optimize_interval, agent_path, n_batches, n_batches_warm
    ):
        train_answers, test_answers, guesses = data
        
        # create train list of parallel games 
        train_env_list = []
        for _ in range(n_envs):
            env = Environment(
                rewards=rewards,
                wordle=Wordle(vocabulary=guesses, answers=train_answers),
                state_instance=StateYesNo()
            )
            train_env_list.append(env)
        state_size = train_env_list[0].state.size

        # test env 
        test_env = Environment(
            rewards=rewards,
            wordle=Wordle(vocabulary=guesses, answers=test_answers),
            state_instance=StateYesNo()
        )

        replay_buffer = None
        if alpha == 0:
            replay_buffer = ReplayBuffer(state_size=state_size, buffer_size=rb_size)
        else:
            replay_buffer = PrioritizedReplayBuffer(state_size=state_size, alpha=alpha, buffer_size=rb_size)

        # create agent with weights from `agent_path`
        agent = Agent(
            state_size=state_size,
            action_instance=ActionCombLetters(
                k=k, vocabulary=guesses,
                ohe_matrix= ohe1 if k == 1 else ohe2,
                wordle_list=wordle_list
            ),
            replay_buffer=replay_buffer,
            optimize_interval=optimize_interval,
            agent_path=agent_path,
            optimizer=partial(torch.optim.Adam, lr=lr),
            model=SkipConnectionQNetwork
        )

        if fine_tune:
            if agent_path is None:
                raise ValueError('Fine tune is possible only with provided weights')
            agent.freeze_layers()
        
        if backbone_path is not None:
            agent.load_backbone(backbone_path)

        # to track experiments
        problem_name = f'{len(test_answers)}-{len(guesses)}'
        if method_name is None:
            method_name = 'comb-letters'
        nickname = f'{method_name}-{problem_name}'

        # training and evaluating utilities
        trainer = Trainer(
            agent=agent,
            train_env=train_env_list,
            test_env=test_env,
            play_batch_size=len(train_env_list),
            n_batches=n_batches,
            n_batches_warm=n_batches_warm,
            nickname=nickname,
            logging_interval=logging_interval
        )

        res = trainer.train(
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
        )

        # save metrics
        tasks_results[problem_name][method_name] = res
        
        return trainer.nickname
    
    def train_test_split(n_guesses, overfit, guesses, indices, in_answers):
        guesses = np.array(guesses)
        guesses_cur = guesses[indices[:n_guesses]]
        
        train_indices = []
        test_indices = []
        for i_guess in indices[:n_guesses]:
            if i_guess in in_answers:
                test_indices.append(i_guess)
            else:
                train_indices.append(i_guess)

        if overfit:
            train_answers_cur = guesses[test_indices]
        else:
            train_answers_cur = guesses[train_indices]
        
        test_answers_cur = guesses[test_indices]

        print(
            f'guesses: {len(guesses_cur)}',
            f'train answers: {len(train_answers_cur)}',
            f'test answers: {len(test_answers_cur)}' + (' (overfit strategy)' if overfit else ''),
            sep='\n'
        )

        return train_answers_cur, test_answers_cur, guesses_cur
    
    def train_wagons(
        rewards, ohe, wordle_list, tasks_results, rb_size,
        k, eps_start, eps_end, eps_decay, lr=5e-4, alpha=0, n_hidden_layers=1,
        logging_interval=None, method_name=None, fine_tune=False, backbone_path=None, *,
        data, n_envs, optimize_interval, agent_path, n_batches, n_batches_warm
    ):
        train_answers, test_answers, guesses = data
        
        # create train list of parallel games 
        train_env_list = []
        for _ in range(n_envs):
            env = Environment(
                rewards=rewards,
                wordle=Wordle(vocabulary=guesses, answers=train_answers),
                state_instance=StateYesNo()
            )
            train_env_list.append(env)
        state_size = train_env_list[0].state.size

        # test env 
        test_env = Environment(
            rewards=rewards,
            wordle=Wordle(vocabulary=guesses, answers=test_answers),
            state_instance=StateYesNo()
        )

        replay_buffer = None
        if alpha == 0:
            replay_buffer = ReplayBuffer(state_size=state_size, buffer_size=rb_size)
        else:
            replay_buffer = PrioritizedReplayBuffer(state_size=state_size, alpha=alpha, buffer_size=rb_size)

        # create agent with weights from `agent_path`
        agent = Agent(
            state_size=state_size,
            action_instance=ActionWagons(
                k=k, vocabulary=guesses,
                ohe_matrix=ohe,
                wordle_list=wordle_list
            ),
            replay_buffer=replay_buffer,
            optimize_interval=optimize_interval,
            agent_path=agent_path,
            optimizer=partial(torch.optim.Adam, lr=lr),
            n_hidden_layers=n_hidden_layers
        )

        if backbone_path is not None:
            agent.load_backbone(backbone_path)

        if fine_tune:
            agent.freeze_layers()

        # to track experiments
        problem_name = f'{len(test_answers)}-{len(guesses)}'
        if method_name is None:
            method_name = 'wagons'
        nickname = f'{method_name}-{problem_name}'

        # training and evaluating utilities
        trainer = Trainer(
            agent=agent,
            train_env=train_env_list,
            test_env=test_env,
            play_batch_size=len(train_env_list),
            n_batches=n_batches,
            n_batches_warm=n_batches_warm,
            nickname=nickname,
            logging_interval=logging_interval
        )

        res = trainer.train(
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
        )

        # save metrics
        tasks_results[problem_name][method_name] = res
        
        return trainer.nickname
    
    def train_vocabulary(
        rewards, tasks_results, rb_size,
        eps_start, eps_end, eps_decay, lr=5e-4, alpha=0,
        logging_interval=None, method_name=None, fine_tune=False, backbone_path=None, *,
        data, n_envs, optimize_interval, agent_path, n_batches, n_batches_warm
    ):
        train_answers, test_answers, guesses = data
        
        # create train list of parallel games 
        train_env_list = []
        for _ in range(n_envs):
            env = Environment(
                rewards=rewards,
                wordle=Wordle(vocabulary=guesses, answers=train_answers),
                state_instance=StateYesNo()
            )
            train_env_list.append(env)
        state_size = train_env_list[0].state.size

        # test env 
        test_env = Environment(
            rewards=rewards,
            wordle=Wordle(vocabulary=guesses, answers=test_answers),
            state_instance=StateYesNo()
        )

        replay_buffer = None
        if alpha == 0:
            replay_buffer = ReplayBuffer(state_size=state_size, buffer_size=rb_size)
        else:
            replay_buffer = PrioritizedReplayBuffer(state_size=state_size, alpha=alpha, buffer_size=rb_size)

        # create agent with weights from `agent_path`
        agent = Agent(
            state_size=state_size,
            action_instance=ActionVocabulary(vocabulary=guesses),
            replay_buffer=replay_buffer,
            optimize_interval=optimize_interval,
            agent_path=agent_path,
            optimizer=partial(torch.optim.Adam, lr=lr)
        )

        if fine_tune:
            if agent_path is None:
                raise ValueError('Fine tune is possible only with provided weights')
            agent.freeze_layers()
        
        if backbone_path is not None:
            agent.load_backbone(backbone_path)

        # to track experiments
        problem_name = f'{len(test_answers)}-{len(guesses)}'
        if method_name is None:
            method_name = 'wagons'
        nickname = f'{method_name}-{problem_name}'

        # training and evaluating utilities
        trainer = Trainer(
            agent=agent,
            train_env=train_env_list,
            test_env=test_env,
            play_batch_size=len(train_env_list),
            n_batches=n_batches,
            n_batches_warm=n_batches_warm,
            nickname=nickname,
            logging_interval=logging_interval
        )

        res = trainer.train(
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
        )

        # save metrics
        tasks_results[problem_name][method_name] = res
        
        return trainer.nickname
    