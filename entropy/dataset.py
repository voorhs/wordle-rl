from torch.utils.data import Dataset
import torch

class EntropyDataset(Dataset):
    def __init__(self, path):
        """
        Params
        ---------
        path : str
            Path to directory with .csv dataset generated by entropy.solver.EntropyData
        """
        with open(path, 'r') as f:
            self.content = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        """
        Returns
        -------
            state (torch.tensor): vector value of envorinment.environment.BaseState inheritant
            action (int): index of word from list of valid guesses
        """
        pieces = self.content[idx].split()
        action = pieces[0]
        state = torch.tensor(pieces[1:])
        return state, action
        
