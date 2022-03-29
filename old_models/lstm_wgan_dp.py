import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from datasets import keys_position_sequences_dataset_30s
from old_models.models_util import DEVICE, GANModel

# Discriminator hyperparameters
lr_d = 0.0004
d_dim = 512

# Generator hyperparameters
lr_g = 0.0001
z_size = 100
g_dim = 256

# Other hyperparameters
beta1 = 0.5
beta2 = 0.999

# state and data dimensions
keys_size = 5  # includes space
game_state_dim = 1  # robot_mode

# Loss weight for gradient penalty
lambda_gp = 10
n_critic = 5

hidden_units = 400  # size of lstm hidden layer

batch_size = 32

data_loader = DataLoader(keys_position_sequences_dataset_30s, batch_size=32, shuffle=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.lstm = nn.LSTM(input_size=keys_size + game_state_dim, hidden_size=hidden_units, dropout=0.3, num_layers=2,
                            batch_first=True)
        self.fc_layer = nn.Sequential(
            # nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(in_features=hidden_units, out_features=1),
        )

    def forward(self, seq, game_state):
        real_bs, seq_len = game_state.shape[0], game_state.shape[1]
        seq = torch.cat((seq, game_state), dim=2)
        out, _ = self.lstm(seq)
        out = self.fc_layer(out.contiguous().view(real_bs * seq_len, hidden_units))

        return out.view(real_bs, seq_len, 1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc_layer1 = nn.Sequential(
            nn.Linear(in_features=(z_size + game_state_dim), out_features=hidden_units),
            # nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=0.3)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Sequential(
            # nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.Dropout(0.3),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=hidden_units, out_features=keys_size),
            nn.Tanh()
        )

    def forward(self, z, game_state):
        inp = torch.cat((z, game_state), dim=2)
        # split to (seq_len, batch_size, num_feats)
        inps = torch.split(inp, 1, dim=1)

        # manually process each timestep
        state1, state2 = None, None  # (h1, c1), (h2, c2)

        gen_out = []

        for in_step in inps:
            in_step = in_step.squeeze(dim=1)  # (batch_size, z_size + game_state_dim)
            out = self.fc_layer1(in_step)  # (batch_size, hidden_units)

            h1, c1 = self.lstm_cell1(out, state1)  # h1 -> (batch_size, hidden_units)

            h1 = self.dropout(h1)  # feature dropout only (no recurrent dropout)

            h2, c2 = self.lstm_cell2(h1, state2)  # h1 -> (batch_size, hidden_units)

            out = self.fc_layer2(h2)  # (batch_size, keys_size)

            gen_out.append(out)

            state1 = (h1, c1)
            state2 = (h2, c2)

            # we would calculate the condition for next step and append it to the next z here

        # (seq_len, batch_size, keys_size) -> (batch_size, seq_len, keys_size)
        gen_feats = torch.stack(gen_out, dim=1)

        return gen_feats

    def generate(self, game_state):
        z = torch.empty([game_state.shape[0], game_state.shape[1], z_size]).uniform_().to(DEVICE)
        return self(z, game_state)


D = Discriminator()
G = Generator()

D.to(DEVICE)
G.to(DEVICE)

D.float()
G.float()

d_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))

gan_model = GANModel(D, G, "lstm_wgan_dp.pt")


def compute_gradient_penalty(real_samples, fake_samples, game_state):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand_like(real_samples, device=DEVICE)
    # Get random interpolation between real and fake samples
    # print("alpha size:", alpha.size(), "real_Samples size:", real_samples.size(), "fake samples:", fake_samples.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
    d_interpolates = D(interpolates, game_state)
    fake = torch.ones((real_samples.shape[0], real_samples.shape[1], 1), requires_grad=False, device=DEVICE)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(), -1)
    gradient_penalty = ((gradients.norm(2, dim=2) - 1) ** 2).mean()
    return gradient_penalty


def train_model(num_epochs=10):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state) in enumerate(data_loader):
            robot_mode = game_state[:, :, 0:1]

            """
                Discriminator
            """
            d_optimizer.zero_grad()

            with torch.no_grad():
                fake_keys = G.generate(robot_mode)

            real_validity = D(real_keys, robot_mode)
            fake_validity = D(fake_keys.detach(), robot_mode)

            with torch.backends.cudnn.flags(enabled=False):
                gradient_penalty = compute_gradient_penalty(real_keys, fake_keys, robot_mode)
            d_loss = - real_validity.mean() + fake_validity.mean() + lambda_gp * gradient_penalty

            d_loss.backward()
            d_optimizer.step()

            # Print some loss stats
            if batch_i % n_critic == 0:
                """
                    Generator
                """
                g_optimizer.zero_grad()

                fake_keys = G.generate(robot_mode)
                fake_validity = D(fake_keys, robot_mode)

                g_loss = -fake_validity.mean()

                g_loss.backward()
                g_optimizer.step()

                gan_model.d_losses.append(d_loss.item())
                gan_model.g_losses.append(g_loss.item())

                # print discriminator and generator loss
                print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, batch_i, len(data_loader), d_loss, g_loss))


def sample_generator():
    G.eval()
    real_keys, game_state = next(iter(data_loader))
    robot_mode = game_state[:, :, 0:1]
    x_fake = G.generate(robot_mode).cpu().detach().numpy()
    df = pd.DataFrame(x_fake[0, :, :], columns=["key_front", "key_back", "key_left", "key_right", "key_space"])
    df["robot_mode"] = robot_mode[0, :, :].cpu()
    df.to_csv("sequence_sample.csv")
