import numpy as np
import torch
from collections import defaultdict
from wordle.wordle import Wordle, Dordle
from environment.action import BaseAction
from copy import copy as PythonCopy
from typing import List


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseState:
    def step(self, action, pattern):
        """Update state based on agent guess and wordle pattern."""
        raise NotImplementedError()

    def reset(self):
        """Begin new episode."""
        raise NotImplementedError()

    @property
    def value(self):
        """Return vector view to put it into Q network."""
        raise NotImplementedError()

    def copy(self):
        """Return deep copy of state instance."""
        raise NotImplementedError()


class StateYesNo(BaseState):
    # inspired by https://github.com/andrewkho/wordle-solver/blob/master/deep_rl/wordle/state.py
    def __init__(self, n_letters, n_steps):
        self.n_letters = n_letters
        self.steps_left = n_steps
        self.init_steps = n_steps

        # 26 indicators that color of this letter is known
        self.isknown = np.zeros(26)

        # 26 indicators that letter is in answer
        self.isin = np.zeros(26)

        # for each 26 letters of alphabet:
        #   for each 5 letters of word:
        #       no/yes;
        self.coloring = np.zeros((26, self.n_letters, 2))

    @property
    def size(self):
        ans = 1  # number of steps left
        for arr in [self.isknown, self.isin, self.coloring]:
            ans += arr.size
        return ans

    @property
    def value(self):
        # this vector is supposed to be input of DQN network
        return np.r_[self.isknown, self.isin, self.coloring.ravel(), self.steps_left]

    def step(self, action:BaseAction, pattern, done):
        self.steps_left -= 1
        guess = action.word
        yes_letters = []

        # mark all green letters as 'yes'
        for pos, g in enumerate(guess):
            if pattern[pos] != 'G':
                continue

            let = self._getord(g)
            self.isknown[let] = 1
            self.isin[let] = 1

            # green letter strikes all the
            # alphabet letters out of this position
            self.coloring[:, pos] = [1, 0]   # 'no'
            self.coloring[let, pos] = [0, 1]  # 'yes'

            yes_letters.append(g)  # to check for duplicate black letters

        maybe_letters = []

        # for Y and B the logic is more complicated
        for pos, g in enumerate(guess):
            if pattern[pos] == 'G':  # already marked
                continue

            let = self._getord(g)
            self.isknown[let] = 1

            if pattern[pos] == 'Y':
                self.isin[let] = 1
                maybe_letters.append(g)  # to check for duplicate black letters
                self.coloring[let, pos] = [1, 0]  # 'no'

            elif pattern[pos] == 'B':
                # Wordle colors duplicate of yellow and green letters with black
                # if true number of duplicates is already met in guess
                if g in maybe_letters:
                    # this case we don't need to
                    # strike this letter out of whole word
                    # but in this position
                    self.coloring[let, pos] = [1, 0]  # 'no'
                elif g in yes_letters:
                    # this case we don't need to
                    # strike this letter out of whole word
                    # but in positions where it's not green:
                    # ('no' xor 'yes') is equivalent for ('no' is only where its not 'yes')
                    self.coloring[let, :, 0] = (
                        self.coloring[let, :, 0] != self.coloring[let, :, 1])
                else:
                    # this case we strike this letter
                    # out of whole word
                    self.coloring[let, :] = [1, 0]

    @staticmethod
    def _getord(letter):
        return ord(letter.upper()) - 65

    # start new episode
    def reset(self):
        self.isknown *= 0
        self.isin *= 0
        self.steps_left = self.init_steps
        self.coloring *= 0

    def copy(self):
        res = StateYesNo(n_letters=self.n_letters, n_steps=self.init_steps)
        res.isknown = self.isknown.copy()
        res.isin = self.isin.copy()
        res.steps_left = self.steps_left
        res.coloring = self.coloring.copy()
        return res


class StateVocabulary(StateYesNo):
    def __init__(self, answers_mask=None, answer: str = 'hello', steps=6):
        """`answers_mask` is a mask of possible answers"""
        super().__init__(answer=answer, steps=steps)
        self.init_answers_mask = answers_mask.copy()
        self.answers_mask = answers_mask.copy()

    @property
    def size(self):
        return super().size + self.answers_mask.size

    def step(self, action: BaseAction, pattern, done):
        super().step(action, pattern)
        self.answers_mask[action.value] = int(done)

    @property
    def value(self):
        return np.r_[super().value, self.answers_mask]

    def reset(self, answer):
        super().reset(answer=answer)
        self.answers_mask = self.init_answers_mask.copy()

    def copy(self):
        copy = StateVocabulary(
            answers_mask=self.answers_mask, answer=self.answer, steps=self.init_steps)
        copy.isknown = self.isknown.copy()
        copy.isin = self.isin.copy()
        copy.steps = self.init_steps
        copy.coloring = self.coloring.copy()
        return copy


class StateYesNoDordle(BaseState):
    def __init__(self, n_letters, n_boards, n_steps=None, states_list=None, freeze_list=None):
        self.n_letters = n_letters
        if n_steps is None:
            n_steps = n_boards + 5
        self.init_steps = n_steps
        self.n_boards = n_boards

        if states_list is None:
            states_list = []
            for _ in range(self.n_boards):
                states_list.append(StateYesNo(n_letters, n_steps))
        
        self.states_list: List[StateYesNo] = states_list

        # indicators of finished games
        if freeze_list is None:
            freeze_list = [False for _ in range(self.n_boards)]
        self.freeze_list = freeze_list

    @property
    def size(self):
        res = 0
        for state in self.states_list:
            res += state.size
        return res
    
    @property
    def value(self):
        res = []
        for state in self.states_list:
            res.append(state.value)
        return np.concatenate(res)

    def step(self, action, pattern_list, done_list):
        for i in range(self.n_boards):
            if self.freeze_list[i]:
                continue
            self.states_list[i].step(action, pattern_list[i])
            self.freeze_list[i] |= done_list[i]
    
    def reset(self):
        for i in range(self.n_boards):
            self.states_list[i].reset()
            self.freeze_list[i] = False
    
    def copy(self):
        states_list = []
        for state in self.states_list:
            states_list.append(state.copy())

        return StateYesNoDordle(
            n_letters=self.n_letters,
            n_steps=self.init_steps,
            n_boards=self.n_boards,
            states_list=states_list,
            freeze_list=PythonCopy(self.freeze_list)
        )


class Environment:
    def __init__(
        self, rewards: defaultdict, wordle: Wordle = None, state_instance: BaseState = None
    ):
        # supposed to be dict with keys 'B', 'Y', 'G', 'win', 'lose', 'step'
        self.rewards = rewards

        # instance of Wordle game, which we use for getting color pattern
        if wordle is None:
            wordle = Wordle()
        self.game = wordle

        # instance of envorinment state, which we use for getting input for DQN network
        self.state = state_instance
        if state_instance is None:
            self.state = StateYesNo(self.game.answer)

        # it's better to remember letters
        self.collected = {color: set() for color in ['B', 'Y', 'G']}

    def step(self, action: BaseAction, output):
        # convert action to str guess
        guess = action.word

        # send guess to Wordle instance
        pattern = self.game.send_guess(guess, output)

        # compute reward from pattern
        reward = self._reward(guess, pattern)

        # get to next state of environment
        self.state.step(action, pattern, self.game.isover())

        return self.state.copy(), reward, self.game.isover()

    def _reward(self, guess, pattern):
        # reward (supposed to be negative) for any guess
        result = self.rewards['step']

        # reward for each letter
        for i, color in enumerate(pattern):
            if guess[i] not in self.collected[color]:
                result += self.rewards[color]
                self.collected[color].add(guess[i])
            elif 'repeat' in self.rewards.keys() and color == 'G':
                result += self.rewards['repeat']
        
        # if end of episode
        if self.game.isover():
            result += self.rewards['win'] if self.game.iswin() else self.rewards['lose']
        
        return result

    def reset(self, for_test=None):
        self.game.reset(for_test)
        self.state.reset()
        self.collected = {color: set() for color in ['B', 'Y', 'G']}
        
        return self.state.copy()
    
    def get_test_size(self):
        return len(self.game.answers)


class EnvironmentDordle:
    def __init__(self, rewards, n_boards, dordle: Dordle, state_instance: BaseState):
        self.rewards = rewards
        self.n_boards = n_boards
        self.game = dordle
        self.state = state_instance

        self.collected = self._empty_collected_list()
    
    def _empty_collected_list(self):
        return [{color: set() for color in ['B', 'Y', 'G']} for _ in range(self.n_boards)]

    def step(self, action: BaseAction, output):
        guess = action.word

        # send to all boards
        pattern_list, isover_list = self.game.send_guess(guess, output)
        
        # compute reward
        reward = self._reward(guess, pattern_list)
        
        # change state
        self.state.step(action, pattern_list, isover_list)

        return self.state.copy(), reward, self.game.isover()
    
    def _reward(self, guess, pattern_list):
        result = self.rewards['step']

        for collected, pattern in zip(self.collected, pattern_list):
            # if board is over not to give any reward
            if pattern is None:
                continue
            
            for letter, color in zip(guess, pattern):
                if letter not in collected[color]:
                    # reward for new letters
                    result += self.rewards[color]
                    collected[color].add(letter)
                elif 'repeat' in self.rewards.keys() and color == 'G':
                    # negative reward for repetition of green letters
                    result += self.rewards['repeat']
        
        if self.game.isover():
            result += self.rewards['win'] if self.game.iswin() else self.rewards['lose']
        
        return result
    
    def reset(self, for_test=False):
        self.game.reset(for_test)
        self.state.reset()
        self.collected = self._empty_collected_list()
        return self.state.copy()

    def get_test_size(self):
        return len(self.game.boardwise_answers)
