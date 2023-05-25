# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/train_agent.py

import torch
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from collections import deque, defaultdict
from functools import partial
from time import time
from datetime import datetime
import os
from typing import List, Union, Literal

from environment.environment import Environment, StateYesNo, EnvironmentDordle, StateYesNoDordle
from environment.action import ActionWagons, ActionVocabulary, ActionLetters, ActionCombLetters
from wordle.wordle import Wordle, Dordle
from replay_buffer.cpprb import ReplayBuffer, PrioritizedReplayBuffer
from dqn.agent import Agent
from dqn.model import QNetwork, OldQNetwork


class RLFramework:
    """Basic utilities with agent, environment and game in one framework for experiments with training and testing."""

    def __init__(
            self, agent: Agent,
            train_env: List[Environment],
            test_word_list: List[str],
            nickname,
            detailed_logging=False,
            logging_interval=None,
            n_episodes=80000,
            n_episodes_warm=80,
            play_batch_size=8,
            eps_start=1.0, eps_end=0.05, eps_decay=0.999
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

        self.train_env_list = train_env
        self.test_word_list = test_word_list

        self.n_episodes = n_episodes
        self.n_episodes_warm = n_episodes_warm
        self.play_batch_size = play_batch_size

        self.detailed_logging = detailed_logging
        if logging_interval is None:
            logging_interval = max(n_episodes // 8, 1)
        self.logging_interval = logging_interval
        
        # directory for txt, pth, npz
        i = 1
        path = nickname
        while os.path.exists(path):
            path = nickname + f' ({i})'
            i += 1
        nickname = path
        os.mkdir(nickname)

        self.nickname = nickname

        self.train_output = None

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

    def train(self):
        # duplicate std output to file
        self.logs_output = f'{self.nickname}/logs.txt'
        self.n_episodes_played = 0
        self.elapsed_time = 0

        # ======= COLLECT INITIAL EXPERIENCE =======

        # don't update net and 100% explore
        self.agent.eps = 1
        self.agent.eval = True
        pbar = tqdm(total=self.n_episodes_warm, desc='WARM EPISODES')
        self.play_train(self.n_episodes_warm, pbar=pbar)
        pbar.close()

        # ======= TRAINING =======

        # firstly, test initial agent
        self.agent.eval = True
        self.test_initial()

        # slowly decreasing exporation
        self.agent.eps = self.eps_start

        # for i_batch in tqdm(range(1, self.n_batches+1), desc='TRAIN BATCHES'):
        pbar = tqdm(total=self.n_episodes, desc='TRAIN EPISODES')
        for i_epoch in range(self.n_episodes//self.logging_interval):
            test_output = f'{self.nickname}/test-{i_epoch}.txt'
            train_output = f'{self.nickname}/train-{i_epoch}.txt'

            self.agent.eval = False
            train_stats = self.play_train(epoch_size=self.logging_interval, pbar=pbar, output=train_output)
            
            self.agent.eval = True
            test_stats = self.play_test(output=test_output)
            
            self.save_checkpoint(i_epoch)
            self.log(train_stats, test_stats, game_output=test_output)
        pbar.close()

    def save_checkpoint(self, i):
        self.agent.dump(self.nickname, i) 

    def play_batch_parallel(self, output):
        envs_number = len(self.train_env_list)
        
        state_size = self.train_env_list[0].state.size
        states = np.empty((envs_number, state_size))

        # reset all environments
        for env in self.train_env_list:
            env.reset()
            env.disable_reward_logs()

        # to store stats
        batch_scores = np.zeros(envs_number)
        batch_wins = np.zeros(envs_number, dtype=bool)
        
        all_is_over = False
        while not all_is_over:
            # collect batch of states from envs that are not finished yet
            indexes = []
            for i, env in enumerate(self.train_env_list):
                if env.game.isover():
                    continue

                states[i] = env.state.value
                indexes.append(i)
            
            # feed batch to agent
            actions = self.agent.act_batch(states[indexes])

            all_is_over = True
            for i, action in zip(indexes, actions):
                # send action to env
                next_state, reward, done = self.train_env_list[i].step(action, output)

                # save replay to agent's buffer
                self.agent.add(states[i], action, reward, next_state, done)
            
                # collect stats
                batch_scores[i] += reward

                if done:
                    batch_wins[i] = self.train_env_list[i].game.iswin()
                else:
                    all_is_over = False
        
        return batch_scores, batch_wins

    def play_train(self, epoch_size, pbar=None, output=None):
        n_batches = np.ceil(epoch_size / self.play_batch_size).astype(int)
        if n_batches == 0:
            return
        
        scores = []
        wins = []
        start = time()
        
        for _ in range(n_batches):
            batch_scores, batch_wins = self.play_batch_parallel(output)
            
            if pbar is not None:
                pbar.update(self.play_batch_size)
            
            scores.append(batch_scores)
            wins.append(batch_wins)

            # decrease exploration chance
            self.agent.eps = max(self.eps_end, self.eps_decay * self.agent.eps)

        n_episodes_played = n_batches * self.play_batch_size
        scores = np.concatenate(scores)
        wins = np.concatenate(wins)
        elapsed_time = time() - start

        return n_episodes_played, scores, wins, elapsed_time

    def play_test(self, output=None):
        # test agent without exploration
        eps = self.agent.eps
        self.agent.eps = 0

        # list of game states
        envs_number = len(self.train_env_list)
        state_size = self.train_env_list[0].state.size
        states = np.empty((envs_number, state_size))

        # to store stats
        scores = []
        wins = []
        steps = []

        # reset all environments
        n_words_guessed = 0
        for env in self.train_env_list:
            env.reset(self.test_word_list[n_words_guessed])
            env.enable_reward_logs()
            n_words_guessed += 1
            env.score = 0
            env.steps = 0
        
        epoch_size = len(self.test_word_list)
        i_word = 0
        while i_word < epoch_size:
            indexes = []
            # collect batch of states from envs
            for i, env in enumerate(self.train_env_list):
                if env.game.isover():
                    # start new game
                    if n_words_guessed == epoch_size:
                        continue

                    env.reset(self.test_word_list[n_words_guessed])
                    n_words_guessed += 1
                    env.score = 0
                    env.steps = 0

                states[i] = env.state.value
                indexes.append(i)
            
            # feed batch to agent
            actions = self.agent.act_batch(states[indexes])

            for i, action in zip(indexes, actions):
                env = self.train_env_list[i]

                # send action to env
                next_state, reward, done = env.step(action, output)
                
                # collect stats
                env.score += reward
                env.steps += 1

                # save replay to agent's buffer
                self.agent.add(states[i], action, reward, next_state, done)

                if done:
                    # collect stats
                    scores.append(env.score)
                    wins.append(env.game.iswin())
                    steps.append(env.steps)
                    i_word += 1
        
        self.agent.eps = eps
        return scores, wins, steps

    def log(self, train_stats, test_stats, game_output):
        # collected stats
        n_episodes_played, train_scores, train_wins, elapsed_time = train_stats
        test_scores, test_wins, test_steps = test_stats

        # simple aggregations
        train_win_rate = 100 * np.mean(train_wins)
        test_win_rate = 100 * np.mean(test_wins)
        test_mean_steps = np.mean(test_steps)
        test_mean_score = np.mean(test_scores)

        if self.detailed_logging:
            # top 3 hardest and easiest words
            hard_inds = np.argpartition(self.train_env_list[0].game.loses, kth=range(-3, 0))[range(-3, 0)]
            easy_inds = np.argpartition(self.train_env_list[0].game.wins, kth=range(-3, 0))[range(-3, 0)]
            hard_words = [self.test_word_list[i] for i in hard_inds]
            easy_words = [self.test_word_list[i] for i in easy_inds]

            # distribution of games by number of steps made
            n_letters = self.train_env_list[0].game.n_letters
            test_steps_distribution = np.bincount(test_steps, minlength=n_letters+1)[1:]
            tsd = ', '.join([f'({i}) {val}' for i, val in enumerate(test_steps_distribution, start=1)])
            
            # distribution of games by total reward gained
            test_reward_distribution = np.histogram(test_scores, bins=10)
            counts, bins = test_reward_distribution
            trd = ', '.join([f'[{bins[i]:.1f},{bins[i+1]:.1f}): {val}' for i, val in enumerate(counts)])

        # will be returned in the end of training
        def to_txt(txt, string):
            open(f'{self.nickname}/{txt}', 'a+').write(str(string)+',')
        to_txt('train_scores.txt', ','.join([str(a) for a in train_scores]))
        to_txt('train_win_rates.txt', train_win_rate)
        to_txt('test_mean_scores.txt', test_mean_score)
        to_txt('test_win_rates.txt', test_win_rate)
        
        # rewards distribution
        trtd = self.flush_reward_logs()

        self.n_episodes_played += n_episodes_played
        self.elapsed_time += elapsed_time

        # to train logs
        message = '\t'.join([
            f'\nEpisodes: {self.n_episodes_played:4d}',
            f'Time: {self.elapsed_time:.0f} s',
            f'Agent Eps: {self.agent.eps:.2f}',
            f'Train Win Rate: {train_win_rate:.2f}%',
            f'Test Win Rate: {test_win_rate:.2f}%',
            f'Test Mean Steps: {test_mean_steps:.4f}',
        ])
        if self.detailed_logging:
            message += '\n\n' + '\t'.join([
                f'Hard Words: {", ".join(hard_words)}',
                f'Easy Words: {", ".join(easy_words)}',
            ]) + '\n' + '\n'.join([
                f'Test Games Distribution by Steps: {tsd}',
                f'Test Games Distribution by Reward: {trd}',
                f'Test Rewards Contributions: {trtd}'
            ])
        self.print(message, self.logs_output)

        # to game report
        message = '\n'.join([
                f'Test Win Rate: {sum(test_wins)} / {len(test_wins)} ({test_win_rate:.2f}%)',
                f'Test Mean Steps: {test_mean_steps:.4f}',
        ])
        if self.detailed_logging:
            message += '\n' + '\n'.join([
                f'Hard Words: {", ".join(hard_words)}',
                f'Easy Words: {", ".join(easy_words)}',
                f'Test Games Distribution by Steps: {tsd}',
                f'Test Games Distribution by Reward: {trd}',
                f'Test Rewards Contributions: {trtd}'
            ])
        open(game_output, 'a+').write(message + '\n')

    def flush_reward_logs(self):
        # collate dicts from all environments
        collated = defaultdict(int)
        total = 0
        for env in self.train_env_list:
            for key, val in env.reward_stats.items():
                collated[key] += val
                total += val
        
        # clear all reward stats
        for env in self.train_env_list:
            for key in env.reward_stats.keys():
                env.reward_stats[key] = 0

        # compute fraction of each reward type
        res = {}
        for key, val in collated.items():
            res[key] = val / total

        return ', '.join([f'{key}: {100*val:.2f}%' for key, val in res.items()])

    def print(self, message, output):
        print(message)
        open(output, 'a+').write(message + '\n')

    def test_initial(self):
        # collected stats
        test_scores, test_wins, test_steps = self.play_test(f'{self.nickname}/test-initial.txt')
        
        # simple aggregations
        test_win_rate = 100 * np.mean(test_wins)
        test_mean_steps = np.mean(test_steps)

        # distribution of games by number of steps made
        n_letters = self.train_env_list[0].game.n_letters
        test_steps_distribution = np.bincount(test_steps, minlength=n_letters+1)[1:]
        tsd = ', '.join([f'{i}: {val}' for i, val in enumerate(test_steps_distribution, start=1)])
        
        # distribution of games by total reward gained
        test_reward_distribution = np.histogram(test_scores, bins=10)
        counts, bins = test_reward_distribution
        trd = ', '.join([f'[{bins[i]:.1f},{bins[i+1]:.1f}): {val}' for i, val in enumerate(counts)])

        # rewards distribution
        trtd = self.flush_reward_logs()

        message = '\t'.join([
            f'Initial Stats. ',
            f'Test Win Rate: {test_win_rate:.2f}%',
            f'Test Mean Steps: {test_mean_steps:.4f}',
        ]) + '\n\n' + '\n'.join([
            f'Test Games Distribution by Steps: {tsd}',
            f'Test Games Distribution by Reward: {trd}',
            f'Test Rewards Distribution by Type: {trtd}'
        ])

        # print train stats
        self.print(message, self.logs_output)


def exp_with_action(
    action_type: Literal['vocabulary', 'letters', 'comb_letters', 'wagons'],
    rewards,
    eps_start, eps_end, eps_decay, rb_size=int(1e6), n_letters=5, n_steps=6,
    lr=5e-4, combine_method='concat', hidden_size=256, n_hidden_layers=1, alpha=0, negative_weights=False, positive_weights=False,
    logging_interval=None, fine_tune=False, backbone_path=None, *, method_name,
    data, n_envs, optimize_interval, agent_path, n_episodes, n_episodes_warm, **action_specs
):
    """(Non-generic) experiment configurations. Operates with RLFramework."""

    train_answers, test_answers, guesses_set = data
    guesses_list = list(guesses_set)
    
    # create train list of parallel games 
    train_env_list = []
    for _ in range(n_envs):
        env = Environment(
            rewards=rewards,
            wordle=Wordle(
                vocabulary=guesses_set, answers=train_answers,
                negative_weights=negative_weights,
                positive_weights=positive_weights
            ),
            state_instance=StateYesNo(n_letters=n_letters, n_steps=n_steps)
        )
        train_env_list.append(env)
    state_size = train_env_list[0].state.size

    # synchronize pointers of all env instances
    for env in train_env_list[1:]:
        env.game.wins = train_env_list[0].game.wins
        env.game.loses = train_env_list[0].game.loses

    if alpha == 0:
        replay_buffer = ReplayBuffer(state_size=state_size, buffer_size=rb_size)
    else:
        replay_buffer = PrioritizedReplayBuffer(state_size=state_size, alpha=alpha, buffer_size=rb_size)

    if action_type == 'vocabulary':
        action = ActionVocabulary(
            vocabulary=guesses_list
        )
    elif action_type == 'letters':
        action = ActionLetters(
            vocabulary=guesses_list,
            ohe_matrix=action_specs['ohe_matrix'],
            wordle_list=action_specs['wordle_list'],
        )
    elif action_type == 'comb_letters':
        action = ActionCombLetters(
            vocabulary=guesses_list,
            ohe_matrix=action_specs['ohe_matrix'],
            k=action_specs['k'],
            wordle_list=action_specs['wordle_list']
        )
    elif action_type == 'wagons':
        action = ActionWagons(
            vocabulary=guesses_list,
            ohe_matrix=action_specs['ohe_matrix'],
            k=action_specs['k'],
            wordle_list=action_specs['wordle_list']
        )

    # create agent with weights from `agent_path`
    agent = Agent(
        state_size=state_size,
        action_instance=action,
        replay_buffer=replay_buffer,
        optimize_interval=optimize_interval,
        agent_path=agent_path,
        lr=lr,
        model=OldQNetwork,
        combine_method=combine_method,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers
    )

    if fine_tune:
        if agent_path is None:
            raise ValueError('Fine tune is possible only with provided weights')
        agent.fine_tune()
    
    if backbone_path is not None:
        agent.load_backbone(backbone_path)

    # to track experiments
    problem_name = f'{len(test_answers)}-{len(guesses_list)}'
    nickname = f'{method_name}-{problem_name}'

    # training and evaluating utilities
    exp = RLFramework(
        agent=agent,
        train_env=train_env_list,
        test_word_list=test_answers,
        play_batch_size=len(train_env_list),
        n_episodes=n_episodes,
        n_episodes_warm=n_episodes_warm,
        nickname=nickname,
        logging_interval=logging_interval,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay
    )

    exp.train()
    
    return exp.nickname

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

    return list(train_answers_cur), list(test_answers_cur), set(guesses_cur)


def get_dordle_data(n_guesses, n_boards, guesses, indices, in_answers):
    np.random.seed(0)
    
    guesses_cur = np.array(guesses)[indices[:n_guesses]]
    
    test_indices = []
    for i_guess in indices[:n_guesses]:
        if i_guess in in_answers:
            test_indices.append(i_guess)

    train_answers_cur = np.array(guesses)[test_indices]
    
    boardwise_answers = []
    for _ in range(n_boards):
        boardwise_answers.append(np.random.permutation(train_answers_cur))
    
    print(
        f'guesses: {len(guesses_cur)}',
        f'answers: {len(train_answers_cur)}',
        sep='\n'
    )

    return boardwise_answers, set(guesses_cur)

def play_dordle(
        rewards, ohe_matrix, wordle_list,
        eps_start, eps_end, eps_decay, rb_size=int(1e6), n_letters=5, n_steps=6,
        lr=5e-4, alpha=0,
        logging_interval=None, fine_tune=False, backbone_path=None, *, method_name,
        data, n_envs, optimize_interval, agent_path, n_episodes, n_episodes_warm
    ):
        boardwise_answers, guesses_set = data
        guesses_list = list(guesses_set)
        
        # create train list of parallel games 
        train_env_list = []
        for _ in range(n_envs):
            env = Environment(
                rewards=rewards,
                wordle=Dordle(
                    vocabulary=guesses_set, boardwise_answers=boardwise_answers,
                ),
                state_instance=StateYesNo(n_letters=n_letters, n_steps=n_steps)
            )
            train_env_list.append(env)
        state_size = train_env_list[0].state.size

        replay_buffer = None
        if alpha == 0:
            replay_buffer = ReplayBuffer(state_size=state_size, buffer_size=rb_size)
        else:
            replay_buffer = PrioritizedReplayBuffer(state_size=state_size, alpha=alpha, buffer_size=rb_size)

        # create agent with weights from `agent_path`
        agent = Agent(
            state_size=state_size,
            action_instance=ActionLetters(
                vocabulary=guesses_list,
                ohe_matrix=ohe_matrix,
                wordle_list=wordle_list
            ),
            replay_buffer=replay_buffer,
            optimize_interval=optimize_interval,
            agent_path=agent_path,
            optimizer=partial(torch.optim.Adam, lr=lr),
            model=QNetwork
        )

        if fine_tune:
            if agent_path is None:
                raise ValueError('Fine tune is possible only with provided weights')
            agent.fine_tune()
        
        if backbone_path is not None:
            agent.load_backbone(backbone_path)

        # to track experiments
        problem_name = f'{len(boardwise_answers)}-{len(guesses_list)}'
        nickname = f'{method_name}-{problem_name}'

        # training and evaluating utilities
        trainer = RLFramework(
            agent=agent,
            train_env=train_env_list,
            test_word_list=boardwise_answers,
            play_batch_size=len(train_env_list),
            n_episodes=n_episodes,
            n_episodes_warm=n_episodes_warm,
            nickname=nickname,
            logging_interval=logging_interval,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay
        )

        trainer.train()
        
        return trainer.nickname