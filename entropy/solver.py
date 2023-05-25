import numpy as np
import itertools as it
import json
from scipy.stats import entropy
from tqdm.contrib.concurrent import process_map
from tqdm.notebook import tqdm

MISS = np.uint8(0)
MISPLACED = np.uint8(1)
EXACT = np.uint8(2)

class PatternMatrix:    
    def __getitem__(self, key):
        return self.pattern_matrix[key]

    def load(self, pattern_matrix_path, answers_path, guesses_path):
        self.pattern_matrix = np.load(pattern_matrix_path)
        self.answer_to_ind = json.load(answers_path)
        self.guess_to_ind = json.load(guesses_path)

    def generate(self, guesses, answers, nickname=None):
        """
        This function computes the pairwise patterns between two lists
        of words, returning the result as a grid of hash values.
        
        Many operations that can be are vectorized. The result
        is saved to file so that this only needs to be evaluated once,
        and all remaining pattern matching is a lookup.

        Params
        ------
            guesses: list of valid guesses
            answers: list of words that can be a final answer
            save_name: path to save matrix as .npy file
        
        Result
        ------
            pattern_matrix: np.ndarray of shape (len(guesses), len(answers)),
            it stores patterns as integers from [0..242] encoding ternary chain,
            it is saved to `self.nickname`.npy
        """
        if hasattr(self, 'pattern_matrix'):
            return
        
        # to have a quick access by native string form
        self.guess_to_ind = dict(zip(guesses, it.count()))
        self.answer_to_ind = dict(zip(answers, it.count()))

        # Number of letters, words
        nl = len(guesses[0])
        nw1 = len(guesses)
        nw2 = len(answers)

        # convert word lists to integer arrays
        guesses = np.array([[ord(c) for c in w] for w in guesses], dtype=np.uint8)
        answers = np.array([[ord(c) for c in w] for w in answers], dtype=np.uint8)

        # equality_grid[a, b, i, j] = guesses[a][i] == answers[b][j]
        equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
        for i, j in it.product(range(nl), range(nl)):
            equality_grid[:, :, i, j] = np.equal.outer(guesses[:, i], answers[:, j])

        # full_pattern_matrix[a, b] should represent the 5-color pattern
        # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
        full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

        # Green pass
        for i in range(nl):
            # matches[a, b] = guesses[a][i] = answers[b][i]
            matches = equality_grid[:, :, i, i].flatten()
            full_pattern_matrix[:, :, i].flat[matches] = EXACT

            for k in range(nl):
                # If it's a match, mark all elements associated with
                # that letter, both from the guess and answer, as covered.
                # This is required by yellow pass.
                equality_grid[:, :, k, i].flat[matches] = False
                equality_grid[:, :, i, k].flat[matches] = False

        # Yellow pass
        for i, j in it.product(range(nl), range(nl)):
            matches = equality_grid[:, :, i, j].flatten()
            full_pattern_matrix[:, :, i].flat[matches] = MISPLACED
            for k in range(nl):
                # Similar to above, we want to mark this letter
                # as taken care of, both for answer and guess
                equality_grid[:, :, k, j].flat[matches] = False
                equality_grid[:, :, i, k].flat[matches] = False

        # Rather than representing a color pattern as a lists of integers,
        # store it as a single integer, whose ternary representations corresponds
        # to that list of integers.
        self.pattern_matrix = np.dot(
            full_pattern_matrix, (3**np.arange(nl)).astype(np.uint8)
        )

        # save result to file system
        if nickname is not None:
            np.save(f'{nickname}-pattern_matrix', self.pattern_matrix)
            json.dump(self.answer_to_ind, f'{nickname}-answer_to_ind')
            json.dump(self.guess_to_ind, f'{nickname}-guess_to_ind')

    def full(self):
        """Get full pattern matrix."""
        return self.pattern_matrix
    
    def sub(self, guesses, answers):
        """Get submatrix of pattern matrix."""
        guesses_ind = [self.guess_to_ind[w] for w in guesses]
        answers_ind = [self.answer_to_ind[w] for w in answers]
        return self.pattern_matrix[np.ix_(guesses_ind, answers_ind)]

    def query(self, guess, answer):
        """
        Get list of 0 (black), 1 (yellow), 2 (green) encoded into ternary integer.
        """
        guess_ind = self.guess_to_ind[guess]
        answer_ind = self.answer_to_ind[answer]
        ternary = self.pattern_matrix[guess_ind, answer_ind]
        return ternary

    def filter(self, guess, pattern, word_list):
        """
        Params
        ------
            guess: str, a word from list of known valid guesses
            pattern: int from [0..242], a ternary number
            word_list: to search possible words from
        """
        all_patterns = self.sub([guess], word_list).flatten()
        word_list = np.array(word_list)
        res = word_list[all_patterns == pattern]
        return list(res)

    def get_word_buckets(self, guess, words):
        """Distribute words to their patterns (buckets) based on the given guess."""
        buckets = [None for _ in range (3**len(guess))]
        hashes = self.sub([guess], words).flatten()
        for hash, word in zip(hashes, words):
            buckets[hash].append(word)
        return buckets


def pattern_distributions(guesses, answers, pattern_matrix):
    """
    For each possible guess in guesses, this finds
    the probability of each of 243 pattern.
    
    Return
    ------
        distributions: np.ndarray of shape (len(guesses), 243), they're not normalized
    """
    n_letters = len(guesses[0])

    res = np.zeros((len(guesses), 3**n_letters))
    tmp = []
    for word in guesses:
        tmp.append(pattern_matrix.guess_to_ind[word])
    
    for i in map(lambda x: pattern_matrix.answer_to_ind[x], answers):
        res[range(len(guesses)), pattern_matrix[tmp, i]] += 1
    
    return res / res.sum(axis=1, keepdims=True)

def entropies(guesses, answers, pattern_matrix):
    """
    This routine will normalize distributions if they don't sum to 1.
    If weights are not specified 1.0 is used.

    Return
    ------
        entropies: np.ndarray of shape (len(guesses),)
    """
    distributions = pattern_distributions(guesses, answers, pattern_matrix)
    axis = len(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)


class EntropySolver:
    def __init__(self, pattern_matrix: PatternMatrix, test_words_list, opening=None):
        self.pattern_matrix = pattern_matrix
        self.test_words_list = test_words_list
        self.opening = opening

    def _play_episode(self, word):
        guesses = list(self.pattern_matrix.guess_to_ind.keys())
        answers = list(self.pattern_matrix.answer_to_ind.keys())
        
        # to store result
        steps = 0
        while steps < 6 and len(answers) != 1:
            if steps == 0:
                guess = self.opening
            else:
                # make a guess that has max entropy
                ents = entropies(guesses, answers, self.pattern_matrix)
                guess = guesses[ents.argmax()]

            steps += 1
        
            # this implements Wordle logic
            pattern = self.pattern_matrix.query(guess, word)

            # narrow down list of possible answers
            answers = self.pattern_matrix.filter(guess, pattern, answers)

        iswin = (len(answers) == 1) and (steps < 6)

        return steps+iswin, iswin
    
    def solve(self, n_workers, chunksize):
        stats = process_map(self._play_episode, self.test_words_list, max_workers=n_workers, chunksize=chunksize)
        all_steps, all_wins = list(zip(*stats))

        # stats
        steps = sum(all_steps) / len(all_steps)
        wins = sum(all_wins) / len(all_wins) * 100
        n_letters = len(self.test_words_list[0])
        test_steps_distribution = np.bincount(all_steps, minlength=n_letters+1)[1:]
        tsd = ', '.join([f'({i}) {val}\t' for i, val in enumerate(test_steps_distribution, start=1)])
        
        # display result
        print(
            f'Mean Steps: {steps:.4f}',
            f'Win Rate: {wins:.2f}%',
            '\n' + tsd,
            sep='\t'
        )


    def play_episode(self, word):
        guesses = list(self.pattern_matrix.guess_to_ind.keys())
        answers = list(self.pattern_matrix.answer_to_ind.keys())
        
        # to store result
        steps = 0

        while steps < 6 and len(answers) != 1:
            if steps == 0:
                guess = self.opening
            else:
                # make a guess that has max entropy
                ents = entropies(guesses, answers, self.pattern_matrix)
                guess = guesses[ents.argmax()]

            steps += 1
        
            # this implements Wordle logic
            pattern = self.pattern_matrix.query(guess, word)
            self._report(guess, pattern)

            # narrow down list of possible answers
            answers = self.pattern_matrix.filter(guess, pattern, answers)

        iswin = (len(answers) == 1) and (steps < 6)
        if iswin:
            self._report(answers[0], 242) 
    
    def _report(self, guess, pattern):
        bitmask = self._decode(pattern)
        res = ''

        for g, p in zip(guess, bitmask):
            letter = g.upper()
            if p == 0:
                res += f'{letter:^7}'
            elif p == 1:
                res += f'{"*"+letter+"*":^7}'
            elif p == 2:
                res += f"{'**'+letter+'**':^7}"

        print(res)
    
    def _decode(self, ternary):
        n = ternary
        nums = []
        for _ in range(5):
            n, r = divmod(n, 3)
            nums.append(r)
        return nums
