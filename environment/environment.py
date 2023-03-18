import numpy as np
import torch
from collections import defaultdict
from wordle.wordlenp import Wordle


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseState:
    def step(self, guess, pattern):
        """Update state based on agent guess and wordle pattern."""
        raise NotImplementedError()

    def reset(self, answer):
        """Begin new episode."""
        raise NotImplementedError()

    def tovector(self):
        """Return vector view to put it into Q network."""
        raise NotImplementedError()

    def copy(self):
        """Return deep copy of state instance."""
        raise NotImplementedError()


class BaseAction:
    def size(self):
        """Return size of action space."""
        raise NotImplementedError()

    def set_action(self, nn_output):
        raise NotImplementedError()

    def get_word(self):
        """Return guess corresponding to action instance"""
        raise NotImplementedError()

    def copy_like(self, nn_output):
        """Make new action instance of the same class."""
        raise NotImplementedError()


class StateYesNo(BaseState):
    # inspired by https://github.com/andrewkho/wordle-solver/blob/master/deep_rl/wordle/state.py
    def __init__(self, answer: str = 'hello', steps=6):
        self.answer = answer

        # 26 indicators that color of this letter is known
        self.isknown = np.zeros(26)

        # 26 indicators that letter is in answer
        self.isin = np.zeros(26)

        # steps left
        self.init_steps = steps
        self.steps = steps

        # for each 26 letters of alphabet:
        #   for each 5 letters of word:
        #       no/yes;
        self.coloring = np.zeros((26, 5, 2))

    @property
    def size(self):
        ans = 1  # number of steps left
        for arr in [self.isknown, self.isin, self.coloring]:
            ans += arr.size
        return ans

    def tovector(self):
        # this vector is supposed to be input of DQN network
        ans = np.r_[self.isknown, self.isin, self.coloring.ravel(), self.steps]
        ans = torch.from_numpy(ans).float().to(DEVICE)
        return ans

    def step(self, guess, pattern):
        self.steps -= 1

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

    # start new episode with new word
    def reset(self, answer):
        self.answer = answer
        self.isknown *= 0
        self.isin *= 0
        self.steps = self.init_steps
        self.coloring *= 0

    def copy(self):
        copy = StateYesNo(answer=self.answer, steps=self.init_steps)
        copy.isknown = self.isknown.copy()
        copy.isin = self.isin.copy()
        copy.steps = self.init_steps
        copy.coloring = self.coloring.copy()
        return copy


class StateVocabulary(StateYesNo):
    def __init__(self, answers_mask=None, answer: str = 'hello', steps=6):
        """`answers_mask` is a mask of possible answers"""
        super(StateVocabulary, self).__init__(answer=answer, steps=steps)
        self.answers_mask = answers_mask

    @property
    def size(self):
        return super().size + self.answers_mask.size

    def tovector(self):
        return torch.cat([super().tovector(), torch.Tensor(self.answers_mask).to(DEVICE)])

    def copy(self):
        copy = StateVocabulary(
            answers_mask=self.answers_mask, answer=self.answer, steps=self.init_steps)
        copy.isknown = self.isknown.copy()
        copy.isin = self.isin.copy()
        copy.steps = self.init_steps
        copy.coloring = self.coloring.copy()
        return copy


VOCABULARY = Wordle._load_vocabulary('wordle/guesses.txt', astype=list)


class ActionVocabulary(BaseAction):
    """Action is an index of word in list of possible answers."""

    def __init__(self, nn_output, vocabulary=None):
        self.value = np.argmax(nn_output)
        self.vocabulary = vocabulary
        if vocabulary is None:
            self.vocabulary = VOCABULARY

    def size(self):
        return len(self.vocabulary)

    def set_action(self, nn_output):
        # nn_output is vector of size as number of all possible guesses
        # which entries contain estimated q values
        self.value = np.argmax(nn_output)

    def get_word(self):
        return self.vocabulary[self.value]

    def copy_like(self, nn_output):
        return ActionVocabulary(nn_output, self.vocabulary)


# разделить методы на основные и вспомогательные
# чтобы стороннему человеку было легче вникнуть в алгоритм
class Environment:
    def __init__(
        self, rewards: defaultdict, wordle: Wordle = None, state_instance: BaseState = None
    ):
        # supposed to be dict with keys 'B', 'Y', 'G', 'win', 'lose', 'step'
        self.rewards = rewards

        # instance of Wordle game, which we use for getting color pattern
        self.wordle = wordle
        if wordle is None:
            self.wordle = Wordle()

        # instance of envorinment state, which we use for getting input for DQN network
        self.state = state_instance
        if state_instance is None:
            self.state = StateYesNo(self.wordle.answer)

        # it's better to remember letters
        self.collected = {color: set() for color in ['B', 'Y', 'G']}

    def step(self, action: BaseAction, output=None):
        # convert action to str guess
        guess = action.get_word()

        # send guess to Wordle instance
        pattern = self.wordle.send_guess(guess)

        # compute reward from pattern
        reward = self._reward(guess, pattern)

        # get to next state of environment
        self.state.step(guess, pattern)

        if output is not None:
            # print coloring to output file
            self._print_coloring(output, guess, pattern, reward)

        return self.state.copy(), reward, self.isover()

    def _reward(self, guess, pattern):
        # if end of episode
        if self.isover():
            return self.rewards['win'] if self.wordle.win else self.rewards['lose']

        # reward (supposed to be negative) for any guess
        result = self.rewards['step']

        # reward for each letter
        for i, color in enumerate(pattern):
            if guess[i] not in self.collected[color]:
                result += self.rewards[color]
                self.collected[color].add(guess[i])
        return result

    def reset(self, replace=True):
        self.wordle.reset(replace)
        self.state.reset(self.wordle.answer)
        self.collected = {color: set() for color in ['B', 'Y', 'G']}
        return self.state.copy()

    # indicator of terminal state (end of episode)
    def isover(self):
        return self.wordle.isover()

    def _print_coloring(self, output, guess, pattern, reward):
        with open(output, mode='a') as f:
            for i, p in enumerate(pattern):
                if p == 'B':
                    f.write('{:^7}'.format(guess[i].upper()))
                elif p == 'Y':
                    f.write('{:^7}'.format('*'+guess[i].upper()+'*'))
                elif p == 'G':
                    f.write('{:^7}'.format('**'+guess[i].upper()+'**'))
            f.write(f'\treward: {reward}\n')    # end of word line
            if self.isover():
                f.write('\n')   # end of wordle board
