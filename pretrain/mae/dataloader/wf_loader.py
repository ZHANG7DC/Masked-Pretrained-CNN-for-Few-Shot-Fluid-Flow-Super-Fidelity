
import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
import re
def parse_filename(filename):
    """
    Parse the filename to extract yaw angle, wind direction, and wind speed.
    The filename format can be either:
    1. "{wind direction}-{wind speed}.npy"
    2. "{wind speed}{yaw}.npy" (without a dash, yaw can be negative)
    """

    # Remove the ".npy" extension
    base_name = filename[:-4]

    # Pattern 1: "{wind direction}-{wind speed}.npy"
    match = re.match(r"([a-zA-Z\-]+)-(\d+)", base_name)
    if match:
        wind_direction, wind_speed = match.group(1), int(match.group(2))
        yaw = None  # Yaw is not available in this format
        return wind_direction, wind_speed, yaw

    # Pattern 2: "{wind speed}{yaw}.npy" (without a dash, yaw can be negative)
    # Assuming wind speed is always two digits (e.g., "15") and yaw is the rest (e.g., "-30")
    if base_name.startswith('1'):
        wind_speed = int(base_name[:2])  # Two-digit wind speed
        yaw = int(base_name[2:])   # Remaining part is the yaw angle
    else:
        wind_speed = int(base_name[0])   # One-digit wind speed
        yaw = int(base_name[1:])   # Remaining part is the yaw angle
    return None, wind_speed, yaw

    # Raise an error if the filename doesn't match any pattern
    raise ValueError(f"Invalid filename format: {filename}")
class WFDataset(Dataset):
    def __init__(self, data_path):
        self.L_path = data_path
        self.fn = [fn for fn in os.listdir(self.L_path[0]) if not fn.startswith('.')]
    def __len__(self):
        return len(self.fn)
    def __getitem__(self, idx):
        fn = self.fn[idx]
        wd, ws, yaw = parse_filename(fn)
        floris = np.load(os.path.join(self.L_path[0], self.fn[idx]))[:50, :256, 50:306]/ws
        snapshots = np.load(os.path.join(self.L_path[1], self.fn[idx]))[:50, :256, 50:306]/ws
        #print(floris.shape)
        return torch.from_numpy(np.stack([floris, snapshots],axis=0)).float()