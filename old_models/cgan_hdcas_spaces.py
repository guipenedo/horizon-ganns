import torch.nn as nn
import torch.optim as optim
from old_models.models_util import *

# Discriminator hyperparameters
lr_d = 0.001
d_dim = 64

# Generator hyperparameters
lr_g = 0.001
z_size = 100
g_dim = 256

# keys_size is now 1 : space


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
            *discriminator_block(1 + game_state_dim, d_dim, bn=False),
            # hidden layers
            *discriminator_block(d_dim, 2 * d_dim),
            *discriminator_block(2 * d_dim, 4 * d_dim),
            # output layer
            nn.Linear(4 * d_dim, 1)
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
            *generator_block(z_size + game_state_dim, g_dim, bn=False),
            # hidden layers
            *generator_block(g_dim, 2 * g_dim),
            *generator_block(2 * g_dim, 4 * g_dim),
            # output layer
            nn.Linear(4 * g_dim, 1),
            nn.Sigmoid()
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


def loss_function(output, target):
    # - (y ln(x) + (1-y) ln(1-x))
    ones = torch.ones_like(output)
    return ((target - ones) * torch.relu(ones - output) - target * torch.relu(output)).mean()


gan_model = GANModel(D, G, "cgan_hdcas_spaces_data.pt", loss_function)


def train_d(real_keys, game_state, fake_game_state):
    d_optimizer.zero_grad()
    D_output_real = D(real_keys, game_state)
    # loss on real keys
    r_loss = torch.relu(torch.ones_like(D_output_real) - D_output_real).mean()

    with torch.no_grad():
        fake_keys = G.generate(fake_game_state)

    # Compute the discriminator losses on fake keys
    D_output_fake = D(fake_keys, fake_game_state)
    f_loss = torch.relu(torch.ones_like(D_output_fake) + D_output_fake).mean()
    # add up real and fake losses and perform backprop
    # relu(1 - d_true) + relu(1 + d_fake)
    d_loss = r_loss + f_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_g(game_state):
    g_optimizer.zero_grad()

    fake_keys = G.generate(game_state)
    D_fake = D(fake_keys, game_state)

    g_loss = torch.relu(torch.ones_like(D_fake) - D_fake).mean()

    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def train_model(num_epochs=40):
    D.train()
    G.train()

    state_iter = iter(data_loader_space_key_2)

    def next_state():
        global state_iter
        try:
            _, fake_game_state = next(state_iter)
            return fake_game_state
        except Exception:
            state_iter = iter(data_loader_space_key_2)
            return next_state()

    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state) in enumerate(data_loader_space_key):
            d_loss = train_d(real_keys, game_state, next_state())
            gan_model.d_losses.append(d_loss)
            g_loss = train_g(next_state())
            gan_model.g_losses.append(g_loss)

            # Print some loss stats
            if batch_i % 400 == 0:
                # print discriminator and generator loss
                print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, batch_i, len(data_loader_space_key), d_loss, g_loss))


def sample_generator():
    G.eval()
    x, y = next(iter(data_loader_space_key_only))
    x_fake = G.generate(y)
    empty = torch.tensor([0] * 4).float()
    for i in range(len(x)):
        plotter.plot_sample(torch.cat((empty, x[i].cpu())), y[i].cpu(), False)
        plotter.plot_sample(torch.cat((empty, x_fake[i].cpu())), y[i].cpu(), False)

