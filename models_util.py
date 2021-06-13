from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import plotter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_NORM_FULL_FILENAME = "processed_data/concat_data_norm.csv"
DATA_FILTERED_FILENAME = "processed_data/concat_data_norm_filtered.csv"
DATA_NO_KEYS_FILENAME = "processed_data/concat_data_norm_filtered_no_key.csv"
TRAIN_LOSSES_FILENAME = "processed_data/cgan_hdcas_losses.csv"

batch_size = 32

game_state_dim = 21
keys_size = 4

water_state_dim = 13
clicks_size = 13


class RFFMKeysDataset(Dataset):
    def __init__(self, root):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        # generated data: key_front, key_back, key_left, key_right, NOT anymore -> key_space
        self.x = torch.from_numpy(self.data[:, 31:35]).float()
        # remaining_time,robot_mode,alarm,robot_x,robot_y,robot_theta,robot_x_diff,robot_y_diff,robot_theta_diff,tree_1,tree_2,tree_3,tree_4,tree_5,tree_6,tree_7,tree_8,tree_9,battery_level,temperature,water_robot_tank
        # condition/state: remaining_time, robot_mode, alarm, positions, trees, battery
        # temperature and robot_water
        self.y = torch.from_numpy(self.data[:, :21]).float()

    def __getitem__(self, idx):
        return self.x[idx, :].to(DEVICE), self.y[idx, :].to(DEVICE)

    def __len__(self):
        return len(self.data)


class SpaceKeyDataset(Dataset):
    def __init__(self, root, filter_key_pressed=False):
        self.df = pd.read_csv(root)
        if filter_key_pressed:
            self.df = self.df[self.df["key_space"] > 0]
        self.data = self.df.to_numpy()
        # generated data: key_space
        self.x = torch.unsqueeze(torch.from_numpy(self.data[:, 35]), 1).float()
        # remaining_time,robot_mode,alarm,robot_x,robot_y,robot_theta,robot_x_diff,robot_y_diff,robot_theta_diff,tree_1,tree_2,tree_3,tree_4,tree_5,tree_6,tree_7,tree_8,tree_9,battery_level,temperature,water_robot_tank
        # condition/state: remaining_time, robot_mode, alarm, positions, trees, battery
        # temperature and robot_water
        self.y = torch.from_numpy(self.data[:, :21]).float()

    def __getitem__(self, idx):
        return self.x[idx].to(DEVICE), self.y[idx, :].to(DEVICE)

    def __len__(self):
        return len(self.data)


class RFFMWaterDataset(Dataset):
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


space_key_dataset = SpaceKeyDataset(DATA_NORM_FULL_FILENAME)
space_key_only_dataset = SpaceKeyDataset(DATA_NORM_FULL_FILENAME, True)
data_loader_space_key = DataLoader(space_key_dataset, batch_size=batch_size, shuffle=True)
data_loader_space_key_2 = DataLoader(space_key_dataset, batch_size=batch_size, shuffle=True)
data_loader_space_key_only = DataLoader(space_key_only_dataset, batch_size=batch_size, shuffle=True)

rffm_data_full = RFFMKeysDataset(DATA_NORM_FULL_FILENAME)
data_loader_full = DataLoader(rffm_data_full, batch_size=batch_size, shuffle=True)

rffm_data_filtered = RFFMKeysDataset(DATA_FILTERED_FILENAME)
data_loader_keys = DataLoader(rffm_data_filtered, batch_size=batch_size, shuffle=True)

rffm_data_no_keys = RFFMKeysDataset(DATA_NO_KEYS_FILENAME)
data_loader_no_keys = DataLoader(rffm_data_no_keys, batch_size=batch_size, shuffle=True)

water_data = RFFMWaterDataset(DATA_NORM_FULL_FILENAME)
data_loader_water = DataLoader(water_data, batch_size=batch_size, shuffle=True)

no_keys_iter = iter(data_loader_no_keys)


# loss discriminator: relu(1 + d_fake) + relu(1 - d_true)
# no sigmoid discriminator


def get_fake_game_state():
    global no_keys_iter
    try:
        _, fake_game_state = next(no_keys_iter)
        return fake_game_state
    except Exception:
        no_keys_iter = iter(data_loader_no_keys)
        return get_fake_game_state()


def get_fake_water_state():
    return (torch.rand((batch_size, water_state_dim)) < 0.5).int().to(DEVICE)


class GANModel:
    def __init__(self, D, G, model_data_filename, loss=None):
        self.D = D
        self.G = G
        self.model_data_filename = model_data_filename
        self.loss = loss
        self.g_losses = []
        self.d_losses = []

    def save_model(self):
        data = {
            'discriminator': self.D.state_dict(),
            'generator': self.G.state_dict()
        }
        torch.save(data, self.model_data_filename)

    def load_model(self):
        data = torch.load(self.model_data_filename)
        self.D.load_state_dict(data['discriminator'])
        self.G.load_state_dict(data['generator'])

    def save_losses(self):
        losses = np.array([self.g_losses, self.d_losses]).transpose()
        df = pd.DataFrame(losses, columns=["generator_loss", "discriminator_loss"])
        df.to_csv(TRAIN_LOSSES_FILENAME, index=False)

    def sample_generator(self, plot_all=False):
        self.G.eval()
        x, y = next(iter(data_loader_keys))
        x_fake = self.G.generate(y)
        for i in range(len(x)):
            plotter.plot_sample(x[i].cpu(), y[i].cpu())
            plotter.plot_sample(x_fake[i].cpu(), y[i].cpu())
            if not plot_all:
                break

    def water_sample_generator(self, plot_all=False):
        self.G.eval()
        x, y = next(iter(data_loader_water))
        x_fake = self.G.generate(y)
        for i in range(len(x)):
            plotter.plot_water_sample(x[i].cpu(), y[i].cpu())
            plotter.plot_water_sample(x_fake[i].cpu(), y[i].cpu())
            if not plot_all:
                break

    def real_loss(self, disc_labels):
        real = torch.ones_like(disc_labels, requires_grad=False).to(DEVICE)
        return self.loss(disc_labels, real)

    def fake_loss(self, disc_labels):
        fake = torch.zeros_like(disc_labels, requires_grad=False).to(DEVICE)
        return self.loss(disc_labels, fake)
