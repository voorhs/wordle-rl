from typing import List
from wordle.wordlenp import Wordle
import numpy as np
import torch
import itertools as it
from scipy.special import comb as n_choose_k
import bisect

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseAction:
    @property
    def value(self):
        """
        If currect action instance is made from batch,
        then `value` is torch.LongTensor of shape (batch_size,)
        with indices of words in self.vocabulary corresponding
        to chosen action. This case `value` SHOULD NOT BE CALLED
        out of class definition.

        If the instance is made from iterator,
        then `value` is a single integer.
        """
        return self._value

    @property
    def vocabulary(self) -> List[str]:
        """Get list of all valid guesses"""
        return self._vocabulary

    def get_word(self) -> str:
        return self.vocabulary[self.value]

    def _init_dict(self):
        """
        Dict with extra args for __init__.
        It's the interface for inheritants.
        """
        return {'vocabulary': self.vocabulary}
    
    def __iter__(self):
        self._iter_current = -1
        return self
    
    def __next__(self):
        self._iter_current += 1
        if self._iter_current < len(self.value):
            return type(self)(
                value=self.value[self._iter_current].cpu().item(),
                **self._init_dict()
            )
        raise StopIteration()

    def __call__(self, nn_output):
        return type(self)(
            nn_output=nn_output,
            **self._init_dict()
        )


class ActionVocabulary(BaseAction):
    """Action is an index of word in list of possible answers."""

    def __init__(self, nn_output=None, vocabulary=None, value=None):
        """
        Params
        ------
        nn_output (torch.tensor): shape (batch_size, action_size)
        vocabulary (List[str]): list of all valid guesses
        """
        if nn_output is not None:
            self.qfunc = nn_output
            self._value = torch.argmax(nn_output, dim=1)
        elif value is not None:
            self._value = value
        
        self._vocabulary = vocabulary

    @property
    def size(self):
        return len(self.vocabulary)

class ActionLetters(BaseAction):
    """Action is to choose letter for each position"""

    def __init__(self, vocabulary, nn_output=None, value=None, ohe_matrix=None):
        """
        Params
        ------
        nn_output (torch.tensor): shape (batch_size, action_size)
        vocabulary (List[str]): list of all valid guesses
        ohe_matrix (torch.tensor): shape (130, len(vocabulary)),
            matrix with letter-wise OHEs for each word in vocabulary
        """
        self._vocabulary = vocabulary
        self.ohe_matrix = None
        self.size = 26 * 5

        if ohe_matrix is not None:
            self.ohe_matrix = ohe_matrix
        else:    
            # W[i, j*26+k] = vocabulary[i][j] == 65+k,
            # i.e. indicates that jth letter of ith word is kth letter of alphabet
            self.ohe_matrix = self._ohe()
    
        # self.value: shape (batch_size,)
        if nn_output is not None:
            self.qfunc = nn_output @ self.ohe_matrix
            self._value = self.qfunc.argmax(dim=1)
        elif value is not None:
            self._value = value
   
    def _ohe(self):
        res = torch.zeros((26 * 5, len(self.vocabulary)), device=DEVICE)
            
        for i, word in enumerate(self.vocabulary):
            for j, c in enumerate(word):
                res[j*26+(ord(c)-97), i] = 1
        
        return res

    def _init_dict(self):
        return {
            'ohe_matrix': self.ohe_matrix,
            **super()._init_dict()
        }


class ActionCombLetters(ActionLetters):
    def __init__(self, vocabulary, nn_output=None, value=None, ohe_matrix=None, k=2):
        self.k = 2
        super().__init__(vocabulary, nn_output, value, ohe_matrix)
    
    def _ohe(self):
        # `combs` is a list of k-lengthed tuples with indexes
        # corresponding to combination of letters in word
        combs = list(it.combinations(range(5), self.k))
        n_combs = len(combs)

        # let's find out all unique combinations w.r.t. their positions
        # i.e. all unique pairs of (1st,3rd), (4th,5th) etc.
        unique_combs = [list()] * n_combs
        
        for word in self.vocabulary:
            for i, inds in enumerate(combs):
                comb = ''.join([word[j] for j in inds])
                # keep it sorted to encode quicker then
                bisect.insort_left(unique_combs[i], comb)
        
        lens = [len(combs) for combs in unique_combs]
        
        # in worst case total_length is (5 choose k) * 26^k
        # which is 6760 for k=2
        self.size = sum(lens)

        # to store result
        res = torch.zeros((self.size, len(self.vocabulary)), device=DEVICE)
        
        # these barriers split OHE-vector to `n_combs`` parts
        barriers = [0] + list(it.accumulate(lens))[:-1]

        for i, word in enumerate(self.vocabulary):
            for j, inds in enumerate(combs):
                comb = ''.join([word[k] for k in inds])
                
                # `locate` is index of `comb` in `unique_combs[i]`
                locate = barriers[j] + bisect.bisect_left(unique_combs[j], comb)
                res[locate, i] = 1
                
        return res
