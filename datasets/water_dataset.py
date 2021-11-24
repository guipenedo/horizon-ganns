from torch.utils.data import Dataset
from .dataset_utils import DEVICE, DATA_NORM_FULL_FILENAME
import pandas as pd
import torch


class WaterDataset(Dataset):
    def __init__(self, root):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        # generated data: click_left click_right click_push	click_wrench click_leak_1-9
        self.x = torch.from_numpy(self.data[:, 36:49]).float()
        # robot_mode,leak_1-9,water_robot_tank_oh,water_ground_tank_oh,close_to_water
        self.y = torch.from_numpy(self.df[["robot_mode", "leak_1", "leak_2", "leak_3", "leak_4",
                                           "leak_5", "leak_6", "leak_7", "leak_8", "leak_9",
                                           "water_robot_tank_oh", "water_ground_tank_oh",
                                           "close_to_water"]].to_numpy()).float()

    def __getitem__(self, idx):
        return self.x[idx, :].to(DEVICE), self.y[idx, :].to(DEVICE)

    def __len__(self):
        return len(self.data)


data = WaterDataset(DATA_NORM_FULL_FILENAME)

