import os

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .dataset_utils import DEVICE
import pandas as pd
import torch


class KeysPositionSequencesDataset(Dataset):
    def __init__(self, root, smooth_labels=False):
        self.root = root
        self.smooth_labels = smooth_labels
        self.data_files = os.listdir(root)

    def __getitem__(self, idx):
        df = pd.read_csv(self.root + os.path.sep + self.data_files[idx])
        data = df.to_numpy()
        # generated data: key_front, key_back, key_left, key_right, key_space
        x = torch.from_numpy(data[:, 31:36]).float().to(DEVICE)
        # robot_mode, robot_x, robot_y, robot_theta, robot_x_diff, robot_y_diff, robot_theta_diff
        y_columns = [1, 3, 4, 5, 6, 7, 8]
        y = torch.from_numpy(data[:, y_columns]).float().to(DEVICE)
        if self.smooth_labels:
            x = 0.9 * x
        return x, y

    def __len__(self):
        return len(self.data_files)


dataset = KeysPositionSequencesDataset("processed_data/normalized_sequences", smooth_labels=True)


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, lens
