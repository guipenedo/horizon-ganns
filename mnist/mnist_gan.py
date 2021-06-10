import numpy as np
import torch
from matplotlib import pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
train_data = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)


def real_loss(d_out, smooth=False):
    labels = torch.ones_like(d_out).to(DEVICE)
    if smooth:
        labels = labels * 0.9
    loss = F.binary_cross_entropy_with_logits(d_out, labels)
    return loss


def fake_loss(d_out):
    labels = torch.zeros_like(d_out).to(DEVICE)
    loss = F.binary_cross_entropy_with_logits(d_out, labels)
    return loss


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_size, hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        ])
        self.output_layer = nn.Linear(hidden_dim, output_size)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # flatten image
        x = x.view(x.shape[0], -1)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


class Generator(Discriminator):
    def forward(self, x):
        x = super(Generator, self).forward(x)
        x = torch.tanh(x)
        return x

    def generate(self):
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).to(DEVICE).float()
        return self(z)


# Discriminator hyperparameters
lr_d = 0.002
input_size = 28 * 28
d_output_size = 1
d_hidden_size = 32

# Generator hyperparameters
lr_g = 0.002
z_size = 100
g_output_size = 28 * 28
g_hidden_size = 32

D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

D.to(DEVICE)
G.to(DEVICE)

d_optimizer = optim.Adam(D.parameters(), lr=lr_d)
g_optimizer = optim.Adam(G.parameters(), lr=lr_g)


def train_d(real_images):
    d_optimizer.zero_grad()
    D_output = D(real_images)
    # loss on real images
    r_loss = real_loss(D_output, True)

    with torch.no_grad():
        fake_images = G.generate()

    # Compute the discriminator losses on fake images
    D_output = D(fake_images)
    f_loss = fake_loss(D_output)
    # add up real and fake losses and perform backprop
    d_loss = r_loss + f_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_g():
    g_optimizer.zero_grad()

    fake_images = G.generate()
    D_fake = D(fake_images)

    g_loss = real_loss(D_fake)  # switched labels

    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def train_model(num_epochs=40):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            real_images = real_images * 2 - 1  # rescale input images from [0,1) to [-1, 1)
            real_images = real_images.to(DEVICE)

            d_loss = train_d(real_images)
            g_loss = train_g()

            # Print some loss stats
            if batch_i % 400 == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, d_loss, g_loss))


DATA_SAVE_PATH = 'mnist_gan_data.pt'


def save_model():
    data = {
        'discriminator': D.state_dict(),
        'generator': G.state_dict()
    }
    torch.save(data, DATA_SAVE_PATH)


def load_model():
    data = torch.load(DATA_SAVE_PATH)
    D.load_state_dict(data['discriminator'])
    G.load_state_dict(data['generator'])


def view_samples(samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach().cpu()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')


def sample_generator():
    G.eval()
    view_samples(G.generate())
