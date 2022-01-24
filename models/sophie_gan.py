import random
from os.path import sep

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from datasets import SplitSequencesDataset
from models_util import DEVICE, MODEL_DATA_DIR
from itertools import chain

# Observations parameters
obs_encoding_size = 128
nr_observations = 5
timesteps_per_sample = 10
nr_future_actions = timesteps_per_sample - nr_observations

# Generator parameters
g_hidden_size = 512
g_latent_dim = 100

# Discriminator parameters
d_hidden_size = 256

# Optimization parameters
lr_g = 0.0001
lr_d = 0.0004
beta1 = 0.5
beta2 = 0.999

# Data config
game_state_columns = ["robot_mode", "robot_x", "robot_y", "robot_theta",
                      "tree_1", "tree_2", "tree_3", "tree_4", "tree_5", "tree_6", "tree_7", "tree_8", "tree_9"]
player_output_columns = ["key_front", "key_back", "key_left", "key_right", "key_space"]
game_state_dim = len(game_state_columns)
player_output_dim = len(player_output_columns)
batch_size = 32

dataset10s = SplitSequencesDataset("processed_data/normalized_sequences_10s",
                                   x_columns=player_output_columns,
                                   y_columns=game_state_columns,
                                   smooth_labels=True)
data_loader = DataLoader(dataset10s, batch_size=32, shuffle=True)


class Encoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=2):
        super(Encoder, self).__init__()
        # Linear embedding obsv_size x h
        self.embed = nn.Linear(input_size, encoding_size)
        # The LSTM cell.
        # Input dimension (observations mapped through embedding) is the same as the output
        self.lstm = nn.LSTM(encoding_size, encoding_size, num_layers=n_layers, batch_first=True)

    def forward(self, obsv):
        # Linear embedding
        obsv = self.embed(obsv)
        # Reshape and applies LSTM over a whole sequence or over one single step
        y, _ = self.lstm(obsv)
        return y


class Generator(nn.Module):
    def __init__(self, encoding_size, hidden_size, latent_dim, output_size, num_heads=4):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Q is prev state, K,V come from the condition
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, kdim=encoding_size, vdim=encoding_size,
                                          num_heads=num_heads, dropout=0.3, batch_first=True)
        # layers to be applied iteratively for each future actions step
        self.fc_layer1 = nn.Sequential(
            nn.Linear(hidden_size + latent_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.lstm_cell1 = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.lstm_cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()
        )

    def forward(self, encoded, z):
        bs = encoded.shape[0]
        h_c_1, h_c_2 = None, None
        att_input = torch.zeros((bs, 1, self.hidden_size), device=DEVICE)
        future_actions = []
        for ni in torch.split(z, 1, dim=1):  # split into time chunks
            # attention
            att_out, _ = self.attn(att_input, encoded, encoded)  # N x 1 x hidden_size
            # add noise
            out = torch.concat((att_out, ni), dim=2)  # Nx1x(E+Z)
            out = out.squeeze(dim=1)
            # FC layer resize to hidden_size
            out = self.fc_layer1(out)  # Nx1xE
            # pass through LSTM -> DROPOUT -> LSTM
            h_c_1 = self.lstm_cell1(out, h_c_1)
            out = self.dropout(h_c_1[0])
            h_c_2 = self.lstm_cell2(out, h_c_2)
            att_input = h_c_2[0].unsqueeze(dim=1)
            # FC, final size = output_size
            out = self.fc_layer2(att_input)
            future_actions.append(out)
        return torch.cat(future_actions, dim=1)  # final output NxLxO, with L prediction_steps and O output_size


class Discriminator(nn.Module):
    def __init__(self, encoding_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(output_size + encoding_size, hidden_size)
        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, encoded, future_actions):
        prediction_steps = future_actions.shape[1]
        final_encoding = encoded[:, -1:, :].repeat(1, prediction_steps, 1)
        prediction_with_obs = torch.cat((final_encoding, future_actions), dim=2)

        encoded_prediction = self.encoder(prediction_with_obs)
        return self.fc_layer(encoded_prediction)


# Observations encoder
OE = Encoder(game_state_dim + player_output_dim, obs_encoding_size).to(DEVICE)

# Generator and Discriminator
G = Generator(obs_encoding_size, g_hidden_size, g_latent_dim, player_output_dim).to(DEVICE)
D = Discriminator(obs_encoding_size, d_hidden_size, player_output_dim).to(DEVICE)

OE.float()
D.float()
G.float()

d_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
g_optimizer = optim.Adam(chain(G.parameters(), OE.parameters()), lr=lr_g, betas=(beta1, beta2))


# def compute_gradient_penalty(real_samples, fake_samples, game_state):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = torch.rand_like(real_samples, device=DEVICE)
#     # Get random interpolation between real and fake samples
#     # print("alpha size:", alpha.size(), "real_Samples size:", real_samples.size(), "fake samples:", fake_samples.size())
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
#     d_interpolates = D(interpolates, game_state)
#     fake = torch.ones((real_samples.shape[0], real_samples.shape[1], 1), requires_grad=False, device=DEVICE)
#     # Get gradient w.r.t. interpolates
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(), -1)
#     gradient_penalty = ((gradients.norm(2, dim=2) - 1) ** 2).mean()
#     return gradient_penalty

bce_loss = nn.BCELoss().to(DEVICE)


def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake).to(DEVICE)
    return bce_loss(scores_fake, y_fake).mean()


def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real).to(DEVICE)
    y_fake = torch.zeros_like(scores_fake).to(DEVICE)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return (loss_real + loss_fake).mean()


def train_model(num_epochs=10):
    D.train()
    G.train()
    d_losses = []
    g_losses = []
    pbar = tqdm(total=num_epochs * len(data_loader))

    fixed_real_keys, fixed_game_state = next(iter(data_loader))
    fixed_z = torch.randn((1, nr_future_actions, g_latent_dim), device=DEVICE)
    print(f"\nReal_keys for fixed test input: {fixed_real_keys[:1, nr_observations:, :]}")

    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state) in enumerate(data_loader):
            observed_keys = real_keys[:, :nr_observations, :]
            real_future_keys = real_keys[:, nr_observations:, :]

            observed_game_state = game_state[:, :nr_observations, :]

            """
                Discriminator
            """
            d_optimizer.zero_grad()

            with torch.no_grad():
                game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))
                z = torch.randn((game_state_encoding.shape[0], nr_future_actions, g_latent_dim), device=DEVICE)
                fake_future_keys = G(game_state_encoding, z)

            real_validity = D(game_state_encoding.detach(), real_future_keys)
            fake_validity = D(game_state_encoding.detach(), fake_future_keys.detach())

            #with torch.backends.cudnn.flags(enabled=False):
            #    gradient_penalty = compute_gradient_penalty(real_keys, fake_keys, robot_mode)
            #d_loss = - real_validity.mean() + fake_validity.mean() + lambda_gp * gradient_penalty

            d_loss = gan_d_loss(real_validity, fake_validity)

            d_loss.backward()
            d_optimizer.step()

            # Print some loss stats
            #if batch_i % n_critic == 0:
            """
                Generator
            """
            g_optimizer.zero_grad()

            game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

            z = torch.randn((game_state_encoding.shape[0], nr_future_actions, g_latent_dim), device=DEVICE)
            fake_future_keys = G(game_state_encoding, z)
            fake_validity = D(game_state_encoding.detach(), fake_future_keys.detach())

            g_loss = gan_g_loss(fake_validity)

            g_loss.backward()
            g_optimizer.step()

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            # print discriminator and generator loss
            if batch_i % 100 == 0:
                print('\nEpoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, batch_i, len(data_loader), d_loss, g_loss))
                game_state_encoding = OE(torch.cat((fixed_game_state[:1, :nr_observations, :],
                                                    fixed_real_keys[:1, :nr_observations, :]), dim=2))
                print(f'\nGenerated data for fixed input: {G(game_state_encoding, fixed_z)}')
            pbar.update()
    pbar.close()
    return d_losses, g_losses


MODEL_SAVE_FILENAME = "sophie_gan"
SAVEFILE_PATH = MODEL_DATA_DIR + sep + MODEL_SAVE_FILENAME + ".pt"


def save_model():
    data = {
        'obs_encoder': OE.state_dict(),
        'discriminator': D.state_dict(),
        'generator': G.state_dict()
    }
    torch.save(data, SAVEFILE_PATH)


def load_model():
    data = torch.load(SAVEFILE_PATH, map_location=DEVICE)
    D.load_state_dict(data['discriminator'])
    G.load_state_dict(data['generator'])
    OE.load_state_dict(data['obs_encoder'])


def save_losses(g_losses, d_losses):
    losses = np.array([g_losses, d_losses]).transpose()
    df = pd.DataFrame(losses, columns=["generator_loss", "discriminator_loss"])
    df.to_csv(MODEL_DATA_DIR + sep + MODEL_SAVE_FILENAME + ".csv", index=False)
