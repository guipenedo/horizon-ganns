import torch.nn as nn
import torch.optim as optim
from models_util import *
import plotter
from datasets import move_keys_full_state_dataset

# Discriminator hyperparameters
lr_d = 0.0001
d_dim = 512

# Generator hyperparameters
lr_g = 0.001
z_size = 100
g_dim = 256


# Other hyperparameters
beta1 = 0.5
beta2 = 0.999

# Loss weight for gradient penalty
lambda_gp = 10
n_critic = 5

data_loader = DataLoader(move_keys_full_state_dataset, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            # input layer
            nn.Linear(keys_size * (MAX_KEYPRESSES + 1), d_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_dim, d_dim),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_dim, 1)
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_feat, out_feat, bn=True):
            block = [nn.Linear(in_feat, out_feat)]
            if bn:
                block.append(nn.BatchNorm1d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.layers = nn.Sequential(
            # input layer
            *generator_block(z_size, g_dim, bn=False),
            # hidden layers
            *generator_block(g_dim, 2 * g_dim),
            *generator_block(2 * g_dim, 4 * g_dim),
            # output layer
            nn.Linear(4 * g_dim, keys_size * (MAX_KEYPRESSES + 1)),  # includes 0
            nn.Unflatten(1, (keys_size, MAX_KEYPRESSES + 1)),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        return self.layers(x)

    def generate(self):
        z = torch.randn((batch_size, z_size), device=DEVICE)
        return self(z)


D = Discriminator()
G = Generator()

D.to(DEVICE)
G.to(DEVICE)

D.float()
G.float()

d_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))

gan_model = GANModel(D, G, "wgan_dp_no_cond_hdcas.pt")


# adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L119
def compute_gradient_penalty(real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand_like(real_samples, device=DEVICE)
    # Get random interpolation between real and fake samples
    #print("alpha size:", alpha.size(), "real_Samples size:", real_samples.size(), "fake samples:", fake_samples.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), requires_grad=False, device=DEVICE)
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_d(real_keys):
    d_optimizer.zero_grad()

    # critic on real_keys
    real_validity = D(real_keys)

    # critic on fake_keys
    with torch.no_grad():
        z = torch.randn((real_keys.size(0), z_size), device=DEVICE)
        fake_keys = G(z)
    fake_validity = D(fake_keys)

    gradient_penalty = compute_gradient_penalty(real_keys, fake_keys)

    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_g():
    g_optimizer.zero_grad()

    fake_keys = G.generate()
    fake_validity = D(fake_keys)

    g_loss = -torch.mean(fake_validity)

    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def train_model(num_epochs=40):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state) in enumerate(data_loader):
            d_loss = train_d(real_keys)

            if batch_i % n_critic == 0:
                g_loss = train_g()
                gan_model.d_losses.append(d_loss)
                gan_model.g_losses.append(g_loss)

                # Print some loss stats
                if batch_i % 400 == 0:
                    # print discriminator and generator loss
                    print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch + 1, num_epochs, batch_i, len(data_loader), d_loss, g_loss))


def sample_generator():
    G.eval()
    x, y = next(iter(data_loader))
    x = x.argmax(dim=2)
    x_fake = G.generate().argmax(dim=2)
    empty = torch.tensor([0]).float()
    for i in range(len(x)):
        plotter.plot_sample(torch.cat((x[i].cpu(), empty)), y[i].cpu(), False)
        plotter.plot_sample(torch.cat((x_fake[i].cpu(), empty)), y[i].cpu(), False)
