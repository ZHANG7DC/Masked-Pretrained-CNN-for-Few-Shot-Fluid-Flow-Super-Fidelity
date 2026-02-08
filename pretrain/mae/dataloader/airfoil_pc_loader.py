import os
import yaml
import numpy as np
from torch.utils.data import Dataset
import torch
class AFPCDataset(Dataset):
    def __init__(self, H_path, L_path, fnames, names):
        with open(fnames, 'r') as f:
            self.fn = [line.strip() for line in f if line.strip()]
        self.H_path = H_path
        self.L_path = L_path
        self.names = names
    def __len__(self):
        return len(self.fn)   
    def __getitem__(self, idx):
        item = dict()
        if 'LF' in self.names:
            item['LF'] = [torch.load(os.path.join(self.L_path[i], self.fn[idx])) for i in range(len(self.L_path))]
        if 'HF' in self.names:
            item['HF'] = torch.load(os.path.join(self.H_path, self.fn[idx]))
        return item