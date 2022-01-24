import torch.nn as nn
import torch.optim as optim
from models_util import *

# Discriminator hyperparameters
lr_d = 0.001
d_dim = 64

# Generator hyperparameters
lr_g = 0.001
z_size = 100
g_dim = 256


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

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Linear(in_filters, out_filters)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout(0.25))
            return block

        self.layers = nn.Sequential(
            # input layer
            *discriminator_block(clicks_size + water_state_dim, d_dim, bn=False),
            # hidden layers
            *discriminator_block(d_dim, 2 * d_dim),
            *discriminator_block(2 * d_dim, 4 * d_dim),
            # output layer
            nn.Linear(4 * d_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters, bn=True):
            block = [nn.Linear(in_filters, out_filters)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout(0.25))
            return block

        self.layers = nn.Sequential(
            # input layer
            *generator_block(z_size + water_state_dim, g_dim, bn=False),
            # hidden layers
            *generator_block(g_dim, 2 * g_dim),
            *generator_block(2 * g_dim, 4 * g_dim),
            # output layer
            nn.Linear(4 * g_dim, clicks_size),
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


gan_model = GANModel(D, G, "cgan_water_hdcas_data.pt", torch.nn.BCELoss())


def train_d(real_clicks, water_state):
    d_optimizer.zero_grad()
    D_output = D(real_clicks, water_state)
    # loss on real clicks
    r_loss = gan_model.real_loss(D_output)

    fake_water_state = get_fake_water_state()
    with torch.no_grad():
        fake_clicks = G.generate(fake_water_state)

    # Compute the discriminator losses on fake keys
    D_output = D(fake_clicks, fake_water_state)
    f_loss = gan_model.fake_loss(D_output)
    # add up real and fake losses and perform backprop
    d_loss = r_loss + f_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_g():
    g_optimizer.zero_grad()

    water_state = get_fake_water_state()

    fake_clicks = G.generate(water_state)
    D_fake = D(fake_clicks, water_state)

    g_loss = gan_model.real_loss(D_fake)

    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def train_model(num_epochs=40):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_clicks, water_state) in enumerate(data_loader_water):
            d_loss = train_d(real_clicks, water_state)
            gan_model.d_losses.append(d_loss)

            g_loss = train_g()
            gan_model.g_losses.append(g_loss)

            # Print some loss stats
            if batch_i % 400 == 0:
                # print discriminator and generator loss
                print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, batch_i, len(data_loader_water), d_loss, g_loss))
