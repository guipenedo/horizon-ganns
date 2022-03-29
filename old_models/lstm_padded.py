import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from old_models import plotter
from old_models.models_util import DEVICE, GANModel
from datasets import keys_position_sequences_dataset, pad_collate

# partly adapted from
# https://github.com/cjbayron/c-rnn-gan.pytorch/blob/master/c_rnn_gan.py

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

# Loss weight for gradient penalty
lambda_gp = 10
n_critic = 5

# state and data dimensions
flat_labels_size = 5  # includes space
hidden_units = 256  # size of lstm hidden layer

batch_size = 32
MAX_SEQ_LEN = 601

data_loader = DataLoader(keys_position_sequences_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(input_size=flat_labels_size, hidden_size=hidden_units, num_layers=2, batch_first=True,
                            dropout=0.4, bidirectional=True)
        self.fc_layer = nn.Linear(in_features=2 * hidden_units, out_features=1)

    def forward(self, seq, lens):
        state = self.init_hidden(seq.shape[0])

        drop_in = self.dropout(seq)

        # (batch_size, seq_len, num_directions*hidden_size)
        packed_seq = pack_padded_sequence(drop_in, lens, batch_first=True, enforce_sorted=False)

        lstm_out, state = self.lstm(packed_seq, state)

        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True, padding_value=0)

        # (batch_size, seq_len, 1)
        out = self.fc_layer(output_padded)

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        out = torch.mean(out, dim=reduction_dims)

        return out

    def init_hidden(self, batch_size):
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 2  # for being bidirectional

        hidden = (weight.new(2 * layer_mult, batch_size,
                             hidden_units).zero_().to(DEVICE),
                  weight.new(2 * layer_mult, batch_size,
                             hidden_units).zero_().to(DEVICE))

        return hidden


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         def generator_block(in_feat, out_feat, bn=True):
#             block = [nn.Linear(in_feat, out_feat)]
#             if bn:
#                 block.append(nn.BatchNorm1d(out_feat, 0.8))
#             block.append(nn.Dropout(0.4))
#             block.append(nn.LeakyReLU(0.2, inplace=True))
#             return block
#
#         self.layers = nn.Sequential(
#             # input layer
#             *generator_block(z_size, g_dim, bn=False),
#             # hidden layers
#             *generator_block(g_dim, 2 * g_dim),
#             *generator_block(2 * g_dim, 4 * g_dim),
#             # output layer
#             nn.Linear(4 * g_dim, flat_labels_size),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#     def generate(self):
#         z = torch.randn((batch_size, z_size), device=DEVICE)
#         return self(z)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc_layer1 = nn.Linear(in_features=(z_size + flat_labels_size), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=0.4)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=flat_labels_size)

    def forward(self, z):
        # z: (batch_size, seq_len, num_feats)
        # z here is the uniformly random vector
        batch_size, seq_len, num_feats = z.shape

        states = self.init_hidden(batch_size)

        # split to seq_len * (batch_size * num_feats)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze(dim=1) for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, flat_labels_size]).uniform_(-1).to(DEVICE)

        # manually process each timestep
        state1, state2 = states  # (h1, c1), (h2, c2)
        gen_feats = []
        for z_step in z:
            # concatenate current input features and previous timestep output features
            concat_in = torch.cat((z_step, prev_gen), dim=-1)
            out = F.relu(self.fc_layer1(concat_in))
            h1, c1 = self.lstm_cell1(out, state1)
            h1 = self.dropout(h1)  # feature dropout only (no recurrent dropout)
            h2, c2 = self.lstm_cell2(h1, state2)
            prev_gen = torch.tanh(self.fc_layer2(h2))
            gen_feats.append(prev_gen)

            state1 = (h1, c1)
            state2 = (h2, c2)

        # seq_len * (batch_size * flat_labels_size) -> (batch_size * seq_len * flat_labels_size)
        gen_feats = torch.stack(gen_feats, dim=1)

        return gen_feats

    def generate(self):
        z = torch.empty([batch_size, MAX_SEQ_LEN, z_size]).uniform_().to(DEVICE)
        return self(z)

    def init_hidden(self, batch_size):
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        hidden = ((weight.new(batch_size, hidden_units).zero_().to(DEVICE),
                   weight.new(batch_size, hidden_units).zero_().to(DEVICE)),
                  (weight.new(batch_size, hidden_units).zero_().to(DEVICE),
                   weight.new(batch_size, hidden_units).zero_().to(DEVICE)))

        return hidden


D = Discriminator()
G = Generator()

D.to(DEVICE)
G.to(DEVICE)

D.float()
G.float()

d_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))

gan_model = GANModel(D, G, "lstm_wgan_dp.pt")


# adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L119
def compute_gradient_penalty(real_samples, fake_samples, lens):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand_like(real_samples, device=DEVICE)
    # Get random interpolation between real and fake samples
    # print("alpha size:", alpha.size(), "real_Samples size:", real_samples.size(), "fake samples:", fake_samples.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
    d_interpolates = D(interpolates, lens)
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


# def train_d(real_keys, lens):
#     d_optimizer.zero_grad()
#
#     # critic on real_keys
#     real_validity = D(real_keys, lens)
#
#     # critic on fake_keys
#     with torch.no_grad():
#         z = torch.randn((real_keys.size(0), z_size), device=DEVICE)
#         fake_keys = G(z)
#         fake_keys_lens = [fake_keys.shape[1]] * fake_keys.shape[0]
#     fake_validity = D(fake_keys, fake_keys_lens)
#
#     gradient_penalty = compute_gradient_penalty(real_keys, fake_keys)
#
#     d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
#
#     d_loss.backward()
#     d_optimizer.step()
#     return d_loss.item()
#
#
# def train_g():
#     g_optimizer.zero_grad()
#
#     fake_keys = G.generate()
#     fake_validity = D(fake_keys)
#
#     g_loss = -torch.mean(fake_validity)
#
#     g_loss.backward()
#     g_optimizer.step()
#     return g_loss.item()


def train_model(num_epochs=10):
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_keys, game_state, lens) in enumerate(data_loader):
            real_batch_size = real_keys.shape[0]
            fake_lens = [MAX_SEQ_LEN] * real_batch_size

            """
                Generator
            """
            g_optimizer.zero_grad()

            fake_keys = G.generate()
            fake_validity = D(fake_keys, fake_lens)

            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            g_optimizer.step()
            g_loss = g_loss.item()

            """
                Discriminator
            """
            d_optimizer.zero_grad()

            real_validity = D(real_keys, lens)
            fake_validity = D(fake_keys.detach(), fake_lens)
            # gradient_penalty = compute_gradient_penalty(real_keys, fake_keys, lens, d_state)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)  # + lambda_gp * gradient_penalty

            d_loss.backward()
            d_optimizer.step()

            d_loss = d_loss.item()

            gan_model.d_losses.append(d_loss)
            gan_model.g_losses.append(g_loss)

            # Print some loss stats
            if batch_i % 5 == 0:
                # print discriminator and generator loss
                print('Epoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, batch_i, len(data_loader), d_loss, g_loss))


def sample_generator():
    G.eval()
    x, y = next(iter(data_loader))
    x_fake = G.generate()
    empty = torch.tensor([0]).float()
    for i in range(len(x)):
        plotter.plot_sample(torch.cat((x[i].cpu(), empty)), y[i].cpu(), False)
        plotter.plot_sample(torch.cat((x_fake[i].cpu(), empty)), y[i].cpu(), False)
