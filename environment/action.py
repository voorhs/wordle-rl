from typing import List
from wordle.wordle import Wordle
import numpy as np
import torch
import itertools as it
import bisect
from math import ceil

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAction:
    """
    Accepted sets of __init__ args:
        [vocabulary]                    init first ever action instance (before training)
        [vocabulary, nn_output]         make new action instance using fabric (agent.act)
        [vocabulary, index]             unbatch actions to single ones (__next__)
        [vocabulary, nn_output, index]  select specific actions and qfuncs (agent.learn as local value)
    
    In all cases (except the first) extra arguments can be provided
    to save some attributes in order to avoid recalculating it.
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

    # for compatibility with max entropy solution
    def __init__(self, word, index=0):
        self._vocabulary = [word]
        self._index = index

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
        """
        Used to split batched Action object to single ones.
        Copies only inheritant-specific attributes and sets the index of word.
        """
        self._iter_current += 1
        if self._iter_current < len(self.index):
            res = type(self)(
                **self._init_dict()
            )
            res.set_index(self.index[self._iter_current].cpu().item())
            return res
        raise StopIteration()
    
    def set_index(self, index):
        """For unbatching via __next__"""
        self._index = index

    def __call__(self, nn_output, **kwargs):
        """Fabric."""
        res = type(self)(
            **kwargs,
            **self._init_dict()
        )
        res.feed_nn_output(nn_output)
        return res
    
    def feed_nn_output(self, nn_output):
        """For fabric via __call__"""
        raise NotImplementedError()

    def qfunc_of_action(self, nn_output, index):
        """For retrieving qfunc for qlocal in agent.learn()"""
        raise NotImplementedError()


class ActionVocabulary(BaseAction):
    """Action is an index of word in list of possible answers."""

    def __init__(self, vocabulary: List[str]):
        if not isinstance(vocabulary, list):
            raise ValueError('`vocabulary` arg must be a list')
        self._vocabulary = vocabulary

    def feed_nn_output(self, nn_output):
        self._qfunc, self._index = nn_output.max(1, keepdim=True)
    
    def qfunc_of_action(self, nn_output, index):
        self._index = index
        self._qfunc = nn_output.gather(1, index)
        return self._qfunc

    @property
    def size(self):
        """Size of action"""
        return len(self.vocabulary)


class ActionLetters(BaseAction):
    """Action is to choose letter for each position"""

    def __init__(self, vocabulary: list, ohe_matrix, wordle_list=None):
        """
        Params
        ------
        vocabulary (List[str]): list of all valid guesses
        ohe_matrix (torch.tensor): shape (130, len(vocabulary)),
            matrix with letter-wise OHEs for each word in vocabulary
        wordle_list (Iterable[str]): full list of Wordle words
        """
        # validate input
        if not isinstance(vocabulary, list):
            raise ValueError(f'`vocabulary` arg must be a list, but given {type(vocabulary)}')
        
        self._vocabulary = vocabulary
        self.ohe_matrix = ohe_matrix
        if wordle_list is not None:
            self.ohe_matrix = ActionLetters._sub_ohe(self._vocabulary, self.ohe_matrix, wordle_list)
        self._size = self.ohe_matrix.shape[0]
   
    def _make_ohe(vocabulary, n_letters=5):
        """
        W[i, j*26+k] = vocabulary[i][j] == 65+k, i.e. indicates that
        jth letter of ith word is kth letter of alphabet
        """
        res = torch.zeros((26 * n_letters, len(vocabulary)), device=DEVICE)
            
        for i, word in enumerate(vocabulary):
            for j, c in enumerate(word):
                res[j*26+(ord(c)-97), i] = 1
        
        return res

    def _init_dict(self):
        return {
            'ohe_matrix': self.ohe_matrix,
            **super()._init_dict()
        }

    def _sub_ohe(vocabulary, ohe_matrix, wordle_list):
        """
        Select specific words from `ohe_matrix`. Used for solving subproblems.

        Params
        ------
        wordle_list, Iterable[str]: full list of Wordle words
        """
        sub_indices = []
        for word in vocabulary:
            sub_indices.append(wordle_list.index(word))
        
        return ohe_matrix[:, torch.LongTensor(sub_indices)]

    def feed_nn_output(self, nn_output):
        self._qfunc, self._index = (nn_output @ self.ohe_matrix).max(1, keepdim=True)

    def qfunc_of_action(self, nn_output, index):
        self._index = index
        self._qfunc = (nn_output @ self.ohe_matrix).gather(1, index)
        return self._qfunc
    

class ActionCombLetters(ActionLetters):
    def __init__(self, vocabulary, ohe_matrix, k, wordle_list=None):
        self.k = k
        super().__init__(vocabulary=vocabulary, ohe_matrix=ohe_matrix, wordle_list=wordle_list)
    
    def _make_ohe(vocabulary, k):
        # list of OHEs for all k
        res = []
        n_letters = len(vocabulary[0])

        for k in range(1, k+1):
            # `combs` is a list of k-lengthed tuples with indexes
            # corresponding to combination of letters in word
            combs = list(it.combinations(range(n_letters), k))
            n_combs = len(combs)

            # let's find out all unique combinations w.r.t. their positions
            # i.e. all unique pairs of (1st,3rd), (4th,5th) etc.
            unique_combs = [list() for _ in range(n_combs)]
            
            for word in vocabulary:
                for i, inds in enumerate(combs):
                    comb = ''.join([word[j] for j in inds])
                    # keep it sorted to encode quicker then
                    loc = bisect.bisect_left(unique_combs[i], comb)
                    if len(unique_combs[i]) <= loc or (unique_combs[i][loc] != comb):
                        unique_combs[i].insert(loc, comb)
            
            lens = [len(combs) for combs in unique_combs]
            
            # in worst case total_length is (5 choose k) * 26^k
            # which is 6760 for k=2
            size = sum(lens)

            # to store result
            tmp_res = torch.zeros((size, len(vocabulary)), device=DEVICE)
            
            # these barriers split OHE-vector to `n_combs` parts
            barriers = [0] + list(it.accumulate(lens))[:-1]

            for i, word in enumerate(vocabulary):
                for j, inds in enumerate(combs):
                    comb = ''.join([word[m] for m in inds])
                    
                    # `locate` is index of `comb` in `unique_combs[i]`
                    locate = barriers[j] + bisect.bisect_left(unique_combs[j], comb)
                    tmp_res[locate, i] = 1
            
            res.append(tmp_res)

        return torch.cat(res, dim=0)

    def _init_dict(self):
        return {
            'k': self.k,
            **super()._init_dict()
        }

class ActionWagons(ActionCombLetters):
    def _make_ohe(vocabulary, k):
        n_letters = len(vocabulary[0])
        n_wagons = ceil(n_letters / k)
        
        unique_wagons = [list() for _ in range(n_wagons)]

        for i, wagons in enumerate(unique_wagons):
            for word in vocabulary:
                wagon = word[i:i+k]
                loc = bisect.bisect_left(wagons, wagon)
                if len(wagons) <= loc or (wagons[loc] != wagon):
                    wagons.insert(loc, wagon)
            
        lens = [len(wagons) for wagons in unique_wagons]
        size = sum(lens)

        res = torch.zeros((size, len(vocabulary)), device=DEVICE)

        barriers = [0] + list(it.accumulate(lens))[:-1]

        for i, word in enumerate(vocabulary):
            for j, wagons in enumerate(unique_wagons):
                wagon = word[j:j+k]
                
                loc = barriers[j] + bisect.bisect_left(wagons, wagon)
                res[loc, i] = 1
        
        return res


# ======= EMBEDDING =======
from annoy import AnnoyIndex
from torch.utils.data import Dataset
import torch.nn as nn
from torch import LongTensor as LT
from torch import FloatTensor as FT
import torch as t
from typing import Literal
from math import perm

import itertools as it
from scipy.special import comb
from tqdm.notebook import tqdm_notebook as tqdm


class WordPairsDataset(Dataset):
    """
    All pairs of Wordle words with shared letters. Each pair
    encounters as many times as many letters it shares.
    """
    def __init__(self, vocabulary, path, generate=False, int_bytes=2):
        """
        Generate (or use pregenerated) byte file with pairs of integer numbers
        representing the indexes of words in `vocabulary`.

        Params
        ------
        vocabulary, Iterable[str]: list of full Wordle words list
        path, str: path to save to or load from the dataset
        generate, bool: if False, use file `path`, if True, generate new dataset to `path`
        int_bytes, int: number of bytes to encode each integer
        """
        self.vocabulary = np.array([[ord(c) for c in word] for word in vocabulary])
        self.path = path
        self.int_bytes = int_bytes
        
        if generate:
            self._generate(self.path)

    def __len__(self):
        # if already computed
        if hasattr(self, 'size'):
            return self.size
        
        # compute len
        size = 0
        for i_letter in range(ord('a'), ord('z') + 1):
            count = np.count_nonzero(np.any(self.vocabulary == i_letter, axis=1))
            size += int(perm(count, 2))
        
        self.size = size
        return self.size

    def __getitem__(self, ind):
        src = open(self.path, 'rb')
        
        # get to ind-th pair
        src.seek(ind * 2 * self.int_bytes)
        i = int.from_bytes(src.read(self.int_bytes), byteorder='big')
        
        src.seek((ind * 2 + 1) * self.int_bytes)
        j = int.from_bytes(src.read(self.int_bytes), byteorder='big')
        
        return i, j
    
    def _generate(self, path):
        """
        Generate all pairs of words with shared letters and write to `path`.
        """
        output = open(path, 'wb')
        for i_letter in tqdm(range(ord('a'), ord('z') + 1), desc='LETTERS'):
            n_with_i_letter = np.nonzero(
                np.any(self.vocabulary == i_letter, axis=1)
            )[0].astype(np.int64)
            for i, j in tqdm(it.permutations(n_with_i_letter, 2), desc='PERMUTS'):
                output.write(int(i).to_bytes(self.int_bytes, byteorder='big'))
                output.write(int(j).to_bytes(self.int_bytes, byteorder='big'))
            

class Embedding(nn.Module):
    def __init__(self, embedding_size, vocab_size, n_negs=5):
        super().__init__()
        self.n_negs = n_negs
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size // 2)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size // 2)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]

        nwords = iword.new_empty(batch_size, self.n_negs, dtype=torch.float).uniform_(0, self.vocab_size-1).long()
        ivectors = self.ivectors(iword).unsqueeze(2)
        ovectors = self.ovectors(owords).unsqueeze(1)
        nvectors = self.ovectors(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log()
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, self.n_negs).sum(1)
        
        return (oloss + nloss).mean().neg()
    
    def train_epoch(self, dataloader, optimizer, device):
        total_loss = 0
        
        self.train()
        for batch in tqdm(dataloader, desc='TRAIN BATCHES'):
        # for batch in dataloader:
            iwords, owords = batch[0].to(device), batch[1].to(device)
            self.zero_grad()
            loss = self.forward(iwords, owords)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        return total_loss / len(dataloader.dataset)
    
    def get_table(self):
        return torch.cat([self.ivectors.weight, self.ovectors.weight], dim=1).detach()


# in develop (critical update missing)
class ActionEmbedding(BaseAction):
    """Compatible only with ConvexQNetwork."""
    def __init__(
            self, vocabulary, emb_size, indexer:AnnoyIndex=None, nn_output=None, index=None,
            metric='euclidean', model_path='environment/embedding_model.pth'
    ):
        self._vocabulary = vocabulary
        self._size = emb_size
        
        if indexer is not None:
            self.indexer = indexer
        else:
            self.indexer = self._build_indexer(metric, model_path)

        if nn_output is not None:
            self._qfunc, self.embedding = nn_output
        if index is not None:
            self._index = index
    
    def _init_dict(self):
        return {
            'emb_size': self.size,
            'indexer': self.indexer,
            **super()._init_dict()
        }
    
    @property
    def index(self):
        """
        Finds nearest neighbour to current embedding
        among embeddings of words in `self.guesses` using AnnoyIndex.
        """ + super().__doc__
        if hasattr(self, '_index'):
            return self._index
        
        _index = []
        for emb in self.embedding.cpu():
            _index.append(
                self.indexer.get_nns_by_vector(emb, n=1)[0]
            )
        
        self._index = torch.LongTensor(_index, device=DEVICE)
        return self._index

    def act(self, index):
        """Fabric. Used in agent.act_batch()"""
        return type(self)(
            index=index,
            **self._init_dict()
        )

    def get_embeddings(self, index):
        res = []
        for i in index:
            res.append(self.indexer.get_item_vector(i))
        return torch.tensor(res)
    
    def _build_indexer(
            self, metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot'],
            model_path
    ):
        # load trained nn.Module and retrieve needed matrix
        model = Embedding(self._size, len(self._vocabulary))
        model.load_state_dict(torch.load(
            model_path,
            map_location=torch.device('cpu')
        ))
        embedding_table = model.get_table()
        
        # full list of Wordle words
        wordle_words = Wordle._load_vocabulary('wordle/guesses.txt', astype=list)
        
        # resulting indexer
        res = AnnoyIndex(self._size, metric)
        
        # for each word in current problem's vocabulary
        for i, word in enumerate(self.vocabulary):
            # find location of this word in the full list
            i_emb = wordle_words.index(word)

            # add to indexer the needed vector
            res.add_item(i, embedding_table[i_emb])
        
        res.build(n_trees=10)

        return res