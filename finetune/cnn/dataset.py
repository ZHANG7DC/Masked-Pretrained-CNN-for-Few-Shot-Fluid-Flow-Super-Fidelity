import os
import torch
from torch.utils.data import Dataset
import numpy as np

class BurgersDataset(Dataset):
    def __init__(self, L_path, H_path, training_instance_file=None):
        self.L_path = L_path
        self.H_path = H_path
        if training_instance_file is not None:
            with open(training_instance_file, 'r') as f:
                self.fn = [line.strip() for line in f if line.strip()]
        else:
            self.fn = [str(i) for i in range(3000)]
    def __len__(self):
        return len(self.fn)
    def __getitem__(self, idx):
        fn = self.fn[idx]
        return {'LF': torch.from_numpy(np.load(os.path.join(self.L_path, fn+'_LF.npy'))).float(),
                'HF': torch.from_numpy(np.load(os.path.join(self.H_path, fn+'_HF.npy'))).float(),
                'Name': fn}