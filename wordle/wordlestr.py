from copy import copy
import numpy as np
from wordle.wordlenp import Wordle as BaseWordle


class Wordle(BaseWordle):
    def _validate(self, word: str):
        if not isinstance(word, str):
            raise ValueError(f"Not a string was given: {word}")

        if " " in word:
            raise ValueError("Multiple words in guess are not allowed.")

        if len(word) != len(self.answer):
            raise ValueError(
                f"Guess must be {len(self.answer)} letters length.")

        if self.real_words and word not in self.vocabulary:
            raise ValueError(f"{word} is not in vocabulary.")

    def _getpattern(self, guess: str):
        # initialize pattern
        pattern = [None] * len(self.answer)

        # find green letters
        letters_left = []
        for i, a in enumerate(self.answer):
            if a == guess[i]:
                pattern[i] = 'G'
            else:
                letters_left.append(a)
        iscorrect = len(letters_left) == 0
        
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
        return pattern, iscorrect

    def _prepoc(self, word: str):
        return word