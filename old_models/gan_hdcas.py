import torch.nn as nn
import torch.optim as optim
from old_models.models_util import *

# Discriminator hyperparameters
lr_d = 0.001
d_dim = 128

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

        def discriminator_block(in_filters, out_filters):
            return [nn.Linear(in_filters, out_filters), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]

        self.layers = nn.Sequential(
            nn.Flatten(),
            # input layer
            *discriminator_block(keys_size * (MAX_KEYPRESSES + 1), d_dim),
            # hidden layers
            *discriminator_block(d_dim, 2 * d_dim),
            *discriminator_block(2 * d_dim, 4 * d_dim),
            # output layer
            nn.Linear(4 * d_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters, bn=True):
            block = [nn.Linear(in_filters, out_filters)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.layers = nn.Sequential(
            # input layer
            *generator_block(z_size, g_dim, bn=False),
            # hidden layers
            *generator_block(g_dim, 2 * g_dim),
            #*generator_block(2 * g_dim, 4 * g_dim),
            # output layer
            #nn.Linear(4 * g_dim, keys_size * (MAX_KEYPRESSES + 1)),  # includes 0
            nn.Linear(2 * g_dim, keys_size * (MAX_KEYPRESSES + 1)),  # includes 0
            nn.Unflatten(1, (keys_size, MAX_KEYPRESSES + 1)),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        return self.layers(x)

    def generate(self):
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).to(DEVICE).float()
        return self(z)


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

gan_model = GANModel(D, G, "gan_hdcas_data.pt", nn.MSELoss())


def train_d(real_keys):
    d_optimizer.zero_grad()
    D_output_real = D(real_keys)
    # loss on real keys
    r_loss = gan_model.real_loss(D_output_real)
    #r_loss = torch.relu(torch.ones_like(D_output_real).to(DEVICE) - D_output_real).mean()

    with torch.no_grad():
        fake_keys = G.generate()

    # Compute the discriminator losses on fake keys
    D_output_fake = D(fake_keys)
    f_loss = gan_model.fake_loss(D_output_fake)
    #f_loss = torch.relu(torch.ones_like(D_output_fake).to(DEVICE) + D_output_fake).mean()
    # add up real and fake losses and perform backprop
    d_loss = r_loss + f_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_g():
    g_optimizer.zero_grad()

    fake_keys = G.generate()
    D_fake = D(fake_keys)

    #g_loss = torch.relu(torch.ones_like(D_fake).to(DEVICE) - D_fake).mean()
    g_loss = gan_model.real_loss(D_fake)

    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def train_model(num_epochs=40):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state) in enumerate(data_loader_full):
            d_loss = train_d(real_keys)
            gan_model.d_losses.append(d_loss)
            g_loss = train_g()
            gan_model.g_losses.append(g_loss)

            # Print some loss stats
            if batch_i % 400 == 0:
                # print discriminator and generator loss
                print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, batch_i, len(data_loader_full), d_loss, g_loss))


def sample_generator():
    G.eval()
    x, y = next(iter(data_loader_full))
    x = x.argmax(dim=2)
    x_fake = G.generate().argmax(dim=2)
    empty = torch.tensor([0]).float()
    for i in range(len(x)):
        plotter.plot_sample(torch.cat((x[i].cpu(), empty)), y[i].cpu(), False)
        plotter.plot_sample(torch.cat((x_fake[i].cpu(), empty)), y[i].cpu(), False)
