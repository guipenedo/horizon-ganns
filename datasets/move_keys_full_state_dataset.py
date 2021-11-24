from torch.utils.data import Dataset
import torch.nn.functional as F
from .dataset_utils import DEVICE, DATA_NORM_FULL_FILENAME, DATA_FILTERED_FILENAME, DATA_NO_KEYS_FILENAME
import pandas as pd
import torch

MAX_KEYPRESSES = 10
NOISE_RATE = 9999


class MoveKeysFullStateDataset(Dataset):
    def __init__(self, root, one_hot_encode=False, add_noise_to_encoding=True):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        self.one_hot_encode = one_hot_encode
        self.add_noise_to_encoding = add_noise_to_encoding
        # generated data: key_front, key_back, key_left, key_right, NOT anymore -> key_space
        self.x = torch.from_numpy(self.data[:, 31:35]).float()
        if one_hot_encode:
            self.x = self.x.to(torch.int64)
        # remaining_time,robot_mode,alarm,robot_x,robot_y,robot_theta,robot_x_diff,robot_y_diff,robot_theta_diff,tree_1,tree_2,tree_3,tree_4,tree_5,tree_6,tree_7,tree_8,tree_9,battery_level,temperature,water_robot_tank
        # condition/state: remaining_time, robot_mode, alarm, positions, trees, battery
        # temperature and robot_water
        self.y = torch.from_numpy(self.data[:, :21]).float()

    def __getitem__(self, idx):
        y = self.y[idx, :].to(DEVICE)
        x = self.x[idx, :]
        if self.one_hot_encode:
            x = F.one_hot(x, MAX_KEYPRESSES + 1)
            if self.add_noise_to_encoding:
                noise = torch.distributions.Exponential(
                    torch.full_like(x, NOISE_RATE).float()).sample()
                x = x + noise
                x = x / torch.sum(x, 1, keepdim=True)  # the real value ends up being at around 0.7 prob
                # x = F.softmax(x + noise, dim=1)
        x = x.to(DEVICE)
        return x, y

    def __len__(self):
        return len(self.data)


data = MoveKeysFullStateDataset(DATA_NORM_FULL_FILENAME, one_hot_encode=True)
data_filtered = MoveKeysFullStateDataset(DATA_FILTERED_FILENAME)
data_no_keys = MoveKeysFullStateDataset(DATA_NO_KEYS_FILENAME)

