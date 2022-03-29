from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from os.path import sep
# from datasets import move_keys_full_state_no_keys_dataset

MODEL_DATA_DIR = "../model_data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_LOSSES_FILENAME = "../processed_data/cgan_hdcas_losses.csv"

batch_size = 32

game_state_dim = 21
keys_size = 4

water_state_dim = 13
clicks_size = 13

MAX_KEYPRESSES = 10

NOISE_RATE = 9999

# data_loader_no_keys = DataLoader(move_keys_full_state_no_keys_dataset, batch_size=batch_size, shuffle=True)
# no_keys_iter = iter(data_loader_no_keys)


# loss discriminator: relu(1 + d_fake) + relu(1 - d_true)
# no sigmoid discriminator

#
# def get_fake_game_state():
#     global no_keys_iter
#     try:
#         _, fake_game_state = next(no_keys_iter)
#         return fake_game_state
#     except Exception:
#         no_keys_iter = iter(data_loader_no_keys)
#         return get_fake_game_state()


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

    def save_model(self, file_name=None):
        data = {
            'discriminator': self.D.state_dict(),
            'generator': self.G.state_dict()
        }
        if not file_name:
            file_name = self.model_data_filename
        torch.save(data, MODEL_DATA_DIR + sep + file_name)

    def load_model(self, file_name=None):
        if not file_name:
            file_name = self.model_data_filename
        data = torch.load(MODEL_DATA_DIR + sep + file_name, map_location=DEVICE)
        self.D.load_state_dict(data['discriminator'])
        self.G.load_state_dict(data['generator'])

    def save_losses(self):
        losses = np.array([self.g_losses, self.d_losses]).transpose()
        df = pd.DataFrame(losses, columns=["generator_loss", "discriminator_loss"])
        df.to_csv(TRAIN_LOSSES_FILENAME, index=False)

    # def sample_generator(self, plot_all=False):
    #     self.G.eval()
    #     x, y = next(iter(data_loader_keys))
    #     x_fake = self.G.generate(y)
    #     for i in range(len(x)):
    #         plotter.plot_sample(x[i].cpu(), y[i].cpu())
    #         plotter.plot_sample(x_fake[i].cpu(), y[i].cpu())
    #         if not plot_all:
    #             break
    #
    # def water_sample_generator(self, plot_all=False):
    #     self.G.eval()
    #     x, y = next(iter(data_loader_water))
    #     x_fake = self.G.generate(y)
    #     for i in range(len(x)):
    #         plotter.plot_water_sample(x[i].cpu(), y[i].cpu())
    #         plotter.plot_water_sample(x_fake[i].cpu(), y[i].cpu())
    #         if not plot_all:
    #             break

    def real_loss(self, disc_labels):
        real = torch.ones_like(disc_labels, requires_grad=False).to(DEVICE)
        return self.loss(disc_labels, real)

    def fake_loss(self, disc_labels):
        fake = torch.zeros_like(disc_labels, requires_grad=False).to(DEVICE)
        return self.loss(disc_labels, fake)
