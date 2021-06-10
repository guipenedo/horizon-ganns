import torch.nn as nn
import torch.optim as optim
from torch import autograd

from models_util import *

# Discriminator hyperparameters
lr_d = 0.002
d_dim = 512

# Generator hyperparameters
lr_g = 0.002
z_size = 100
g_dim = 256

# Loss weight for gradient penalty
lambda_gp = 10

# train generator every N iterations
n_critic = 5


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.weight.data = m.weight.data.float()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        m.weight.data = m.weight.data.float()
        m.bias.data = m.bias.data.float()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Linear(in_feat, out_feat)]
            if bn:
                block.append(nn.BatchNorm1d(out_feat))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout(0.25))
            return block

        self.layers = nn.Sequential(
            # input layer
            *discriminator_block(keys_size + game_state_dim, 4 * d_dim, bn=False),
            # hidden layers
            *discriminator_block(4 * d_dim, 2 * d_dim),
            *discriminator_block(2 * d_dim, d_dim),
            # output layer
            nn.Linear(d_dim, 1)
        )

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_feat, out_feat, bn=True):
            block = [nn.Linear(in_feat, out_feat)]
            if bn:
                block.append(nn.BatchNorm1d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout(0.3))
            return block

        self.layers = nn.Sequential(
            # input layer
            *generator_block(z_size + game_state_dim, g_dim, bn=False),
            # hidden layers
            *generator_block(g_dim, 2 * g_dim),
            *generator_block(2 * g_dim, 4 * g_dim),
            # output layer
            nn.Linear(4 * g_dim, keys_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.layers(x)

    def generate(self, game_state):
        z = np.random.uniform(-1, 1, size=(game_state.shape[0], z_size))
        z = torch.from_numpy(z).to(DEVICE).float()
        return self(z, game_state)


D = Discriminator()
G = Generator()

D.to(DEVICE)
G.to(DEVICE)

D.float()
G.float()

D.apply(weights_init)
G.apply(weights_init)

d_optimizer = optim.Adam(D.parameters(), lr=lr_d)
g_optimizer = optim.Adam(G.parameters(), lr=lr_g)

gan_model = GANModel(D, G, "wgan_dp_hdcas_data.pt")


# adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
def compute_gradient_penalty(real_samples, fake_samples, game_state):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1))).to(DEVICE)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
    d_interpolates = D(interpolates, game_state)
    fake = torch.ones((real_samples.shape[0], 1), requires_grad=False).to(DEVICE)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_d(real_keys, game_state):
    d_optimizer.zero_grad()
    D_real = D(real_keys, game_state)

    with torch.no_grad():
        fake_keys = G.generate(game_state)

    # Compute the discriminator losses on fake keys
    D_fake = D(fake_keys, game_state)

    gradient_penalty = compute_gradient_penalty(real_keys.data, fake_keys.data, game_state)
    # Adversarial loss
    d_loss = -torch.mean(D_real) + torch.mean(D_fake) + lambda_gp * gradient_penalty

    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_g(game_state):
    g_optimizer.zero_grad()

    fake_keys = G.generate(game_state)
    D_fake = D(fake_keys, game_state)

    g_loss = -torch.mean(D_fake)

    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def train_model(num_epochs=40):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state) in enumerate(data_loader_keys):
            d_loss = train_d(real_keys, game_state)

            # Print some loss stats
            if batch_i % n_critic == 0:
                g_loss = train_g(get_fake_game_state())

                gan_model.d_losses.append(d_loss)
                gan_model.g_losses.append(g_loss)

                if batch_i % (10 * n_critic) == 0:
                    # print discriminator and generator loss
                    print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch + 1, num_epochs, batch_i, len(data_loader_keys), d_loss, g_loss))
