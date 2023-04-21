from typing import List
from wordle.wordlenp import Wordle
import numpy as np
import torch
import itertools as it
import bisect

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAction:
    """
    Accepted sets of __init__ args:
    [vocabulary]                    init first ever action instance (before training)
    [vocabulary, nn_output]         make new action instance using fabric (agent.act)
    [vocabulary, index]             unbatch actions to single ones (__next__)
    [vocabulary, nn_output, index]  select specific actions and qfuncs (agent.learn as target value)
    In all cases except the first extra arguments can be provided to save some attributes
    in order to avoid recalculating it.
    """
    @property
    def vocabulary(self) -> List[str]:
        """Get list of all valid guesses"""
        return self._vocabulary

    @property
    def size(self):
        """Action space size."""
        return self._size

    @property
    def index(self):
        """
        If currect action instance is made from batch,
        then `index` is torch.LongTensor of shape (batch_size,)
        with indices of words in self.vocabulary corresponding
        to chosen action.

        If the instance is made from iterator,
        then `index` is a single integer.
        """
        return self._index

    @property
    def word(self) -> str:
        return self.vocabulary[self.index]

    @property
    def qfunc(self):
        return self._qfunc

    def _init_dict(self):
        """
        Args for __init__ to move important attributes in order to avoid recalculating it.
        """
        return {'vocabulary': self.vocabulary}
    
    def __iter__(self):
        self._iter_current = -1
        return self
    
    def __next__(self):
        """Used to split batched Action object to single ones."""
        self._iter_current += 1
        if self._iter_current < len(self.index):
            return type(self)(
                index=self.index[self._iter_current].cpu().item(),
                **self._init_dict()
            )
        raise StopIteration()

    def __call__(self, nn_output, **kwargs):
        """Fabric."""
        return type(self)(
            nn_output=nn_output,
            **kwargs,
            **self._init_dict()
        )


class ActionVocabulary(BaseAction):
    """Action is an index of word in list of possible answers."""

    def __init__(self, nn_output:torch.Tensor=None, vocabulary=None, index=None):
        """
        Params
        ------
        nn_output (torch.tensor): shape (batch_size, action_size)
        vocabulary (List[str]): list of all valid guesses
        """
        if nn_output is not None:
            if index is not None:
                self._index = index
                self._qfunc = nn_output.gather(1, index)
            else:
                self._qfunc, self._index = nn_output.max(1, keepdim=True)
        elif index is not None:
            self._index = index
        
        self._vocabulary = vocabulary

    @property
    def size(self):
        """Size of action"""
        return len(self.vocabulary)


class ActionLetters(BaseAction):
    """Action is to choose letter for each position"""

    def __init__(self, vocabulary, nn_output:torch.Tensor=None, index=None, ohe_matrix=None):
        """
        Params
        ------
        nn_output (torch.tensor): shape (batch_size, action_size)
        vocabulary (List[str]): list of all valid guesses
        ohe_matrix (torch.tensor): shape (130, len(vocabulary)),
            matrix with letter-wise OHEs for each word in vocabulary
        """
        self._vocabulary = vocabulary
        self._size = 26 * 5

        if ohe_matrix is not None:
            self.ohe_matrix = ohe_matrix
        else:    
            # W[i, j*26+k] = vocabulary[i][j] == 65+k,
            # i.e. indicates that jth letter of ith word is kth letter of alphabet
            self.ohe_matrix = self._ohe()
    
        if nn_output is not None:
            if index is not None:
                self._index = index
                self._qfunc = (nn_output @ self.ohe_matrix).gather(1, index)
            else:
                self._qfunc, self._index = (nn_output @ self.ohe_matrix).max(1, keepdim=True)
        elif index is not None:
            self._index = index
   
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
    def __init__(self, vocabulary, nn_output=None, index=None, ohe_matrix=None, k=1):
        self.k = k
        super().__init__(vocabulary, nn_output, index, ohe_matrix)
    
    def _ohe(self):
        # `combs` is a list of k-lengthed tuples with indexes
        # corresponding to combination of letters in word
        combs = list(it.combinations(range(5), self.k))
        n_combs = len(combs)

        # let's find out all unique combinations w.r.t. their positions
        # i.e. all unique pairs of (1st,3rd), (4th,5th) etc.
        unique_combs = [list() for _ in range(n_combs)]
        
        for word in self.vocabulary:
            for i, inds in enumerate(combs):
                comb = ''.join([word[j] for j in inds])
                # keep it sorted to encode quicker then
                loc = bisect.bisect_left(unique_combs[i], comb)
                if len(unique_combs[i]) <= loc or (unique_combs[i][loc] != comb):
                    unique_combs[i].insert(loc, comb)
        
        lens = [len(combs) for combs in unique_combs]
        
        # in worst case total_length is (5 choose k) * 26^k
        # which is 6760 for k=2
        self._size = sum(lens)

        # to store result
        res = torch.zeros((self.size, len(self.vocabulary)), device=DEVICE)
        
        # these barriers split OHE-vector to `n_combs` parts
        barriers = [0] + list(it.accumulate(lens))[:-1]

        for i, word in enumerate(self.vocabulary):
            for j, inds in enumerate(combs):
                comb = ''.join([word[k] for k in inds])
                
                # `locate` is index of `comb` in `unique_combs[i]`
                locate = barriers[j] + bisect.bisect_left(unique_combs[j], comb)
                res[locate, i] = 1
                
        return res


class ActionEmbedding(BaseAction):
    def __init__(self, vocabulary, emb_size, embeddings_table, nn_output=None, index=None):
        self._vocabulary = vocabulary
        self._size = emb_size
    
        self.embeddings_table = embeddings_table

        if nn_output is not None:
            self._qfunc, self.embedding = nn_output
        elif index is not None:
            self._index = index
    
    def _init_dict(self):
        return {
            'emb_size': self.size,
            'embeddings_table': self.embeddings_table,
            **super()._init_dict()
        }
    
    @property
    def index(self):
        """
        Finds nearest neighbour to current embedding
        among embeddings of words in `self.guesses` using LSH.
        """ + super().__doc__
        if hasattr(self, '_index'):
            return self._index
        # KNN
        pass
