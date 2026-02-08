import os
import yaml
import numpy as np
from torch.utils.data import Dataset
import re
class PDEDataset(Dataset):
    def __init__(self, data_path, names, indices=None):
        if indices is not None:
            with open(indices, 'r') as f:
                self.samples = [int(line) for line in f]
        else:
            self.samples = [i for i in range(3000)]
        self.names = names
        self.data_path = data_path
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = dict()
        for name in self.names:
            item[name] = np.load(os.path.join(self.data_path, "%d_%s.npy"%(self.samples[idx],name)))
        return item