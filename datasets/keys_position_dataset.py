from torch.utils.data import Dataset
from .dataset_utils import DEVICE, DATA_NORM_TANH_FULL_FILENAME, DATA_NORM_FULL_FILENAME
import pandas as pd
import torch


class KeysPositionDataset(Dataset):
    def __init__(self, root, smooth_labels=False):
        self.regularize = smooth_labels
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        # generated data: key_front, key_back, key_left, key_right, key_space
        self.x = torch.from_numpy(self.data[:, 31:36]).float()
        # robot_mode, robot_x, robot_y, robot_theta, robot_x_diff, robot_y_diff, robot_theta_diff
        y_columns = [1, 3, 4, 5, 6, 7, 8]
        self.y = torch.from_numpy(self.data[:, y_columns]).float()

    def __getitem__(self, idx):
        y = self.y[idx, :].to(DEVICE)
        x = self.x[idx, :].to(DEVICE)
        if self.regularize:
            x = 0.9 * x
        return x, y

    def __len__(self):
        return len(self.data)


data_tanh = KeysPositionDataset(DATA_NORM_TANH_FULL_FILENAME, smooth_labels=True)
data_sigmoid = KeysPositionDataset(DATA_NORM_FULL_FILENAME, smooth_labels=True)

# t-test
# manova variance
# anova
# posthoc
