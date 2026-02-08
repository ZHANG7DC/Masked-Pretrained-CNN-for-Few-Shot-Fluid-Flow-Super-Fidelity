import os
import yaml
import numpy as np
from torch.utils.data import Dataset
import re
import torch
class AFDataset(Dataset):
    def __init__(self, data_path, keys=['Density', 'Momentum', 'Energy'],
             norm=True, norm_file='/home/dczhang/multi-fidelity-pretraining/train-cnn/norm_stats.pt', channels=[0, 1, 2, 4], mask_file='/home/dczhang/multi-fidelity-pretraining/configs/airfoil_mask_256.npy'):
        with open('/home/dczhang/multi-fidelity-pretraining/configs/airfoil-pretrain-instances.txt', 'r') as f:
            self.fn = [line.strip() for line in f if line.strip()]
        self.L_path = data_path
        self.keys = keys
        self.norm = norm
        self.chans = channels

        if mask_file is not None:
            self.mask = torch.from_numpy(np.load(mask_file))  # should be a dict of {fname: [H, W] mask}
        else:
            self.mask = None

        if self.norm:
            if norm_file is not None and os.path.exists(norm_file):
                stats = torch.load(norm_file)
                self.mean = stats['mean']
                self.std = stats['std']
                print(f"Loaded norm stats from {norm_file}")
            else:
                self._compute_norm_stats()
                if norm_file is not None:
                    torch.save({'mean': self.mean, 'std': self.std}, norm_file)
                    print(f"Saved norm stats to {norm_file}")
    def __len__(self):
        return len(self.fn)

    def __getitem__(self, idx):
        return torch.stack([
            self.load_data(os.path.join(path, self.fn[idx]), self.keys)
            for path in self.L_path
        ], dim=0)

    def load_data(self, path, keys):
        data = []
        loaded = torch.load(path)
        for k in keys:
            tensor = loaded[k].permute(2, 0, 1).float()  # [C, H, W]
            data.append(tensor)
        data = torch.cat(data, dim=0)[self.chans]

        if self.norm:
            # Apply normalization only to selected channels
            data = (
                data - self.mean[:, None, None]
            ) / self.std[:, None, None]
            data[:,self.mask] = 0
        return data

    def _compute_norm_stats(self):
        print("Computing mean and std for normalization on selected channels...")
        sum_ = torch.zeros(len(self.chans))
        sum_sq = torch.zeros(len(self.chans))
        n_pixels = 0

        for path in self.L_path:
            for fname in self.fn:
                full_path = os.path.join(path, fname)
                loaded = torch.load(full_path)

                data = torch.cat(
                    [loaded[k].permute(2, 0, 1).float() for k in self.keys], dim=0
                )
                selected = data[self.chans]

                if self.mask is not None:
                    mask = ~self.mask  # [H, W]
                else:
                    mask = torch.ones_like(selected[0], dtype=torch.bool)  # use all

                for i in range(len(self.chans)):
                    values = selected[i][mask]
                    sum_[i] += values.sum()
                    sum_sq[i] += (values ** 2).sum()
                n_pixels += values.numel()

        self.mean = sum_ / n_pixels
        self.std = (sum_sq / n_pixels - self.mean ** 2).sqrt()
        print(f"Selected channel mean: {self.mean}")
        print(f"Selected channel std: {self.std}")