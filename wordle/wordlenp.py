import numpy as np


class Wordle:
    def __init__(
        self,
        answer: str = 'hello',
        real_words: bool = True,
        vocabulary_path: str = 'wordle/guesses.txt',
        answers_path: str = 'wordle/answers.txt',
        max_guesses: int = 6,
        need_preproc=True
    ):
        # for fast operations
        self.need_preproc = need_preproc
        self.answer = self._prepoc(answer)

        # load from path:
        self.vocabulary: set = self._load_vocabulary(vocabulary_path)
        self.answers: list = self._load_vocabulary(answers_path, astype=list)
        
        # to generate answers randomly we sample words
        # from `self.answers` in advance and iterate through them
        self.current_answer = -1

        # check `vocabulary` and `answer` consistency
        if real_words:
            if self.vocabulary is None:
                raise ValueError(
                    '`vocabulary` must be provided in case `real_words` is True')
            if answer not in self.vocabulary:
                raise ValueError('`answer` must be in `vocabulary`')
        self.real_words = real_words

        self.guesses = []
        if max_guesses <= 0:
            raise ValueError(f'`max_guesses` must be positive: {max_guesses}')
        self.max_guesses = max_guesses

        # None: game is in progress
        # True: answer is guessed
        # False: ran out of attempts
        self.win: bool = None

    def send_guess(
        self,
        guess: str,
        logging: bool = True
    ):
        # if game is over
        if self.isover():
            raise StopIteration(
                f"You have already {'won' if self.win else 'lost'}.")

        guess = self._prepoc(guess)

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

        return pattern

    def isover(self):
        return self.win is not None

    @staticmethod
    def _tonumpy(word: str):
        return np.array(list(word.lower()), dtype=np.object_)

    @staticmethod
    def _tostr(array: np.ndarray):
        return ''.join(array)

    def _prepoc(self, word):
        if self.need_preproc:
            return self._tonumpy(word)
        return word

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

    def reset(self, replace=True):
        # reset inner data
        self.guesses = []
        self.win = None

        # move to next answer in list of sampled words
        self.current_answer += 1
        self.current_answer %= len(self.answers)
        
        if self.current_answer == 0:
            # if sampled words are over make new sample
            self.answers_sequence = self._sample_answers(replace)
        
        # update answer for new game
        self.asnwer = self.answers_sequence[self.current_answer]
    
    def _sample_answers(self, replace):
        indices = np.random.choice(
            len(self.answers),
            size=len(self.answers),
            replace=replace)
        return [self.answers[i] for i in indices]

    @staticmethod
    def _load_vocabulary(path, sep='\n', astype=set):
        return astype(open(path, mode='r').read().split(sep))
