from copy import copy
import numpy as np


class Wordle:
    """Main Wordle game taking optional arguments of the answer and whether to check against the dictionary.
        random_daily changes answer every day (see github/preritdas/wordle)."""

    def __init__(
        self, answer: str = 'hello',
        real_words: bool = True,
        vocabulary: set = None,
        max_guesses: int = 6
    ):
        # for fast operations
        self.answer = self._tonumpy(answer)

        # check `vocabulary` and `answer` consistency
        if real_words:
            if vocabulary is None:
                raise ValueError(
                    '`vocabulary` must be provided in case `real_words` is True')
            if answer not in vocabulary:
                raise ValueError('`answer` must be in `vocabulary`')
        self.real_words = real_words
        self.vocabulary = copy(vocabulary)

        # Individual guesses
        self.guesses = []
        if max_guesses <= 0:
            raise ValueError(f'`max_guesses` must be positive: {max_guesses}')
        self.max_guesses = max_guesses

        # None: game is in progress
        # True: answer is guessed
        # False: ran out of attempts
        self.win: bool = None

    # Individual guesses
    def send_guess(
        self,
        guess: np.ndarray,
        logging: bool = True
    ):
        # if game is over
        if self.win is not None:
            raise StopIteration(
                f"You have already {'won' if self.win else 'lost'}.")

        # guess logging
        if logging:
            self.guesses.append(guess)

        # validation
        self._validate(guess)

        # comparing with answer
        pattern, iscorrect = self._getpattern(guess)

        # change game state
        if iscorrect:
            self.win = True
        elif len(self.guesses) == self.max_guesses:
            self.win = False

        return pattern, iscorrect

    @staticmethod
    def _tonumpy(word: str):
        return np.array(list(word.lower()), dtype=np.object_)
    
    @staticmethod
    def _tostr(array: np.ndarray):
        return ''.join(array)

    def _validate(self, word: np.ndarray):
        if not isinstance(word, np.ndarray):
            raise ValueError(f"Not an array was given: {word}")
        
        iterator = (len(c) != 1 for c in word)
        char_len_errors = np.fromiter(iterator, bool)
        if char_len_errors.any():
            raise ValueError(f"All elements must be non empty chars: {word}")

        if " " in word:
            raise ValueError("Multiple words in guess are not allowed.")

        if len(word) != len(self.answer):
            raise ValueError(
                f"Guess must be {len(self.answer)} letters length.")

        if self.real_words and self._tostr(word) not in self.vocabulary:
            raise ValueError(f"{word} is not in vocabulary.")

    def _getpattern(self, guess: np.ndarray):
        # initialize pattern
        pattern = np.empty_like(self.answer, dtype=np.object_)

        # find green letters
        is_green = self.answer == guess
        pattern[is_green] = 'G'
        letters_left = list(self.answer[~is_green])

        # find yellow letters
        for i, g in enumerate(guess):
            if pattern[i] is not None:   # green
                continue
            if g in letters_left:
                pattern[i] = 'Y'
                letters_left.remove(g)
            else:
                pattern[i] = 'B'

        # return pattern and flag that guess is equal to answer
        return pattern, is_green.all()

    # Reset individual guesses
    def reset_guesses(self):
        """Removes all guesses from guess logging to allow 6 more attempts."""
        self.individual_guesses = []
