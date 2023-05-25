# deeply inspired by https://github.com/preritdas/wordle/blob/master/wordle/wordle.py

import numpy as np
from typing import List, Tuple


IN_PROGRESS = 0
LOSE = 1
WIN = 2


class Wordle:
    def __init__(
        self,
        vocabulary: set,
        answers: list,
        max_guesses: int = 6,
        positive_weights=False,
        negative_weights=False,
    ):
        if not isinstance(vocabulary, set):
            raise ValueError(f'`vocabulary` arg must be a set, but given {type(vocabulary)}')
        self.vocabulary = vocabulary
        self.answers = answers
        self.n_letters = len(self.answers[0])
        
        # to generate answers randomly we sample words
        # from `self.answers` in advance and iterate through them
        self.current_answer = -1

        self.guesses_made = 0
        self.max_guesses = max_guesses

        # None, IN_PROGRESS, LOSE, WIN
        self.status = None

        # if some word was guessed [not guesses], then it
        # propability of being sampled on next iterations increases
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights
        self.wins = np.zeros(len(self.answers), dtype=int)
        self.loses = np.zeros(len(self.answers), dtype=int)

    def send_guess(
        self,
        guess: str,
        output=None
    ):
        self.guesses_made += 1

        # comparing with answer
        pattern, iscorrect = self._getpattern(guess)

        # change game state
        if iscorrect:
            self.status = WIN
        elif self.guesses_made == self.max_guesses:
            self.status = LOSE

        # gameplay demonstration
        self._add_to_report(guess, pattern, output)

        return pattern

    def isover(self):
        """Whether the game is ended with result WIN or LOSE"""
        return self.status != IN_PROGRESS
    
    def iswin(self):
        if self.status == IN_PROGRESS:
            raise ValueError('Game is not ended')
        return self.status == WIN

    def _getpattern(self, guess: str):
        # initialize pattern
        pattern = [None for _ in range(self.n_letters)]

        # find green letters
        letters_left = []
        for i, a in enumerate(self.answer):
            if a == guess[i]:
                pattern[i] = 'G'
            else:
                letters_left.append(a)
        iscorrect = (len(letters_left) == 0)

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

    def reset_counter(self):
        self.current_answer = -1

    def reset(self, for_test=None):
        if for_test is not None:
            self.guesses_made = 0
            self.status = IN_PROGRESS
            self.answer = for_test
            self.report = f'Answer: {self.answer}\n'
            return

        # collect stats
        if self.current_answer != -1:
            ind = self.answers_indices[self.current_answer]
            self.wins[ind] += self.iswin()
            self.loses[ind] += not self.iswin()   
        
        # reset inner data
        self.guesses_made = 0
        self.status = IN_PROGRESS

        # move to next answer in list of sampled words
        self.current_answer += 1
        self.current_answer %= len(self.answers)

        if self.current_answer == 0:
            # if sampled words are over make new sample
            self.answers_indices = self._sample_answers()
        
        # update answer for new game
        ind = self.answers_indices[self.current_answer]
        self.answer = self.answers[ind]
        self.report = f'Answer {ind}: {self.answer}\n'
        
    def _sample_answers(self):
        p = Wordle._normalize(self.wins if self.positive_weights else None)
        n = Wordle._normalize(self.loses if self.negative_weights else None)

        return np.random.choice(
            len(self.answers),
            size=len(self.answers),
            p=Wordle._normalize(p+n)
        )
    
    def _normalize(weights, alpha=0.25):
        if weights is None:
            return 0
        if weights is 0:
            return None
        # small constant to prevent zero weight
        w = weights + 1e-5
        return w ** alpha / sum(w ** alpha)

    @staticmethod
    def _load_vocabulary(path, sep='\n', astype=set):
        return astype(open(path, mode='r').read().split(sep))

    def _add_to_report(self, guess, pattern, output):
        if output is None:
            return
        for g, p in zip(guess, pattern):
            letter = g.upper()
            if p == 'B':
                self.report += f'{letter:^7}'
            elif p == 'Y':
                self.report += f'{"*"+letter+"*":^7}'
            elif p == 'G':
                self.report += f"{'**'+letter+'**':^7}"

        self.report += '\n'

        # end of wordle board
        if self.isover():
            self.report += 'WIN\n\n' if self.iswin() else 'LOSE\n\n'
            open(output, mode='a+').write(self.report)


class Dordle:
    def __init__(
        self,
        vocabulary: set,
        boardwise_answers: List[Tuple[str]],
    ):
        self.vocabulary = vocabulary
        self.boardwise_answers = boardwise_answers
        self.n_letters = len(self.boardwise_answers[0][0])

        self.n_boards = len(self.boardwise_answers)
        self.guesses_made = 0
        self.max_guesses = 5 + self.n_boards

        self.boards_list: List[Wordle] = []
        tmp = [list() for _ in boardwise_answers[0]]
        for t in boardwise_answers:
            for lst, e in zip(tmp, t):
                lst.append(e)

        for answers in tmp:
            self.boards_list.append(
                Wordle(vocabulary=self.vocabulary, answers=answers, max_guesses=self.max_guesses)
            )

        self.status = None
    
    def send_guess(self, guess: str, output=None):
        self.guesses_made += 1
        
        # collect patterns from all boards (None for those which are won)
        pattern_list = [None for _ in range(self.n_boards)]
        for i, board in enumerate(self.boards_list):
            if not board.isover():
                pattern_list[i] = board.send_guess(guess)
        
        # which boards are won
        done_list = [board.isover() for board in self.boards_list]
        
        # whether all boards are lost or won
        status_list = [board.status for board in self.boards_list]
        if LOSE in status_list:
            self.status = LOSE
        elif IN_PROGRESS not in status_list:
            self.status = WIN

        # gameplay demonstration
        self._add_to_report(guess, pattern_list, output)

        # done_list is needed by StateYesNoDordle to freeze state instances corresponding to games been won
        return pattern_list, done_list
    
    def isover(self):
        return self.status != IN_PROGRESS
    
    def iswin(self):
        if self.status == IN_PROGRESS:
            raise ValueError('Game is not ended')
        return self.status == WIN

    def reset_counter(self):
        for board in self.boards_list:
            board.reset_counter()

    def reset(self, for_test=None):
        self.guesses_made = 0
        self.status = IN_PROGRESS

        if for_test is not None:
            for answer, board in zip(for_test, self.boards_list):
                board.reset(answer)
        else:
            for board in self.boards_list:
                board.reset()
        
        self.report = 'Answers: ' + ', '.join([board.answer for board in self.boards_list]) + '\n'

    def _add_to_report(self, guess, pattern_list, output):
        if output is None:
            return
        
        for pattern in pattern_list:
            # if board is won
            if pattern is None:
                self.report += ' ' * self.n_letters * 7
                continue
            
            # color each letter
            for g, p in zip(guess, pattern):
                letter = g.upper()
                if p == 'B':
                    self.report += f'{letter:^7}'
                elif p == 'Y':
                    self.report += f'{"*"+letter+"*":^7}'
                elif p == 'G':
                    self.report += f"{'**'+letter+'**':^7}"

        self.report += '\n'

        if self.isover():
            self.report += 'WIN\n\n' if self.iswin() else 'LOSE\n\n'
            open(output, mode='a').write(self.report)