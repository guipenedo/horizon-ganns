import os
from os.path import sep

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SplitSequencesDataset
from old_models.models_util import DEVICE, MODEL_DATA_DIR
from itertools import chain
from sklearn.model_selection import KFold

# Observations parameters
obs_encoding_size = 128
nr_observations = 5

# Predictor parameters
g_hidden_size = 512
g_latent_dim = 100

# Optimization parameters
lr_g = 0.0001
beta1 = 0.9
beta2 = 0.999

# Data config
game_state_columns = ["robot_mode", "robot_x", "robot_y", "robot_theta"]
player_output_columns = ["key_front", "key_back", "key_left", "key_right", "key_space"]
game_state_dim = len(game_state_columns)
player_output_dim = len(player_output_columns)
batch_size = 32

dataset_path = os.getenv("OBSERVATIONS_DIR", "processed_data/observations_5s")

dataset = SplitSequencesDataset(dataset_path,
                                x_columns=player_output_columns,
                                y_columns=game_state_columns,
                                smooth_labels=False)


class Encoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=2):
        super(Encoder, self).__init__()
        # dropout
        self.dropout = nn.Dropout(0.3)
        # Linear embedding obsv_size x h
        self.embed = nn.Linear(input_size, encoding_size)
        # The LSTM cell.
        # Input dimension (observations mapped through embedding) is the same as the output
        self.lstm = nn.LSTM(encoding_size, encoding_size, num_layers=n_layers, batch_first=True)

    def forward(self, obsv):
        # dropout
        obsv = self.dropout(obsv)
        # Linear embedding
        obsv = self.embed(obsv)
        # Reshape and applies LSTM over a whole sequence or over one single step
        y, _ = self.lstm(obsv)
        return y


class Predictor(nn.Module):
    def __init__(self, encoding_size, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Q is prev state, K,V are the entire encoded sequence
        self.attn = nn.MultiheadAttention(embed_dim=encoding_size, kdim=encoding_size, vdim=encoding_size,
                                          num_heads=1, dropout=0.3, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(encoding_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()
        )

    def forward(self, encoded):
        last_encoded_state = encoded[:, -1:, :]
        # attention
        out, _ = self.attn(last_encoded_state, encoded, encoded)  # N x 1 x encoding_size
        # FC, final size = output_size
        out = self.fc_layers(out)  # N x 1 x output_size
        return out

#
# def train_model(num_epochs=10):
#     P.train()
#     OE.train()
#     losses = []
#     pbar = tqdm(total=num_epochs * len(data_loader))
#
#     fixed_real_keys, fixed_game_state = next(iter(data_loader))
#     fixed_z = torch.randn((1, nr_future_actions, g_latent_dim), device=DEVICE)
#     print(f"\nReal_keys for fixed test input: {fixed_real_keys[:1, nr_observations:, :]}")
#
#     for epoch in range(num_epochs):
#         for batch_i, (real_keys, game_state) in enumerate(data_loader):
#             observed_keys = real_keys[:, :nr_observations, :]
#             real_future_keys = real_keys[:, nr_observations:, :]
#
#             observed_game_state = game_state[:, :nr_observations, :]
#
#             """
#                 Discriminator
#             """
#             d_optimizer.zero_grad()
#
#             with torch.no_grad():
#                 game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))
#                 z = torch.randn((game_state_encoding.shape[0], nr_future_actions, g_latent_dim), device=DEVICE)
#                 fake_future_keys = G(game_state_encoding, z)
#
#             real_validity = D(game_state_encoding.detach(), real_future_keys)
#             fake_validity = D(game_state_encoding.detach(), fake_future_keys.detach())
#
#             # with torch.backends.cudnn.flags(enabled=False):
#             #    gradient_penalty = compute_gradient_penalty(real_keys, fake_keys, robot_mode)
#             # d_loss = - real_validity.mean() + fake_validity.mean() + lambda_gp * gradient_penalty
#
#             d_loss = gan_d_loss(real_validity, fake_validity)
#
#             d_loss.backward()
#             d_optimizer.step()
#
#             # Print some loss stats
#             # if batch_i % n_critic == 0:
#             """
#                 Generator
#             """
#             g_optimizer.zero_grad()
#
#             game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))
#
#             z = torch.randn((game_state_encoding.shape[0], nr_future_actions, g_latent_dim), device=DEVICE)
#             fake_future_keys = G(game_state_encoding, z)
#             fake_validity = D(game_state_encoding.detach(), fake_future_keys.detach())
#
#             g_loss = gan_g_loss(fake_validity)
#
#             g_loss.backward()
#             g_optimizer.step()
#
#             d_losses.append(d_loss)
#             g_losses.append(g_loss)
#
#             # print discriminator and generator loss
#             if batch_i % 100 == 0:
#                 print('\nEpoch [{:d}/{:d}] | Batch [{:d}/{:d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
#                     epoch + 1, num_epochs, batch_i, len(data_loader), d_loss, g_loss))
#                 game_state_encoding = OE(torch.cat((fixed_game_state[:1, :nr_observations, :],
#                                                     fixed_real_keys[:1, :nr_observations, :]), dim=2))
#                 print(f'\nGenerated data for fixed input: {G(game_state_encoding, fixed_z)}')
#             pbar.update()
#     pbar.close()
#     return d_losses, g_losses


MODEL_SAVE_FILENAME = "predictor_model_att"
SAVEDIR = MODEL_DATA_DIR + sep
SAVEFILE_PATH = SAVEDIR + MODEL_SAVE_FILENAME + ".pt"


def save_model(filename=SAVEFILE_PATH):
    data = {
        'obs_encoder': OE.state_dict(),
        'predictor': P.state_dict()
    }
    torch.save(data, filename)


def load_model(filename=SAVEFILE_PATH):
    data = torch.load(filename, map_location=DEVICE)
    P.load_state_dict(data['predictor'])
    OE.load_state_dict(data['obs_encoder'])


def save_losses(g_losses, d_losses):
    losses = np.array([g_losses, d_losses]).transpose()
    df = pd.DataFrame(losses, columns=["generator_loss", "discriminator_loss"])
    df.to_csv(MODEL_DATA_DIR + sep + MODEL_SAVE_FILENAME + ".csv", index=False)


def extract_data(data):
    real_keys, game_state = data
    # split observation from current value
    observed_keys = real_keys[:, :nr_observations, :]
    real_future_keys = real_keys[:, nr_observations:, :]

    observed_game_state = game_state[:, :nr_observations, :]

    return observed_keys, real_future_keys, observed_game_state


if __name__ == '__main__':

    # Configuration options
    k_folds = 4
    num_epochs = 5

    # For fold results
    results = torch.zeros((k_folds, num_epochs, 3))

    # Set fixed random number seed
    torch.manual_seed(42)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    folds = kfold.split(dataset)
    folds = list(folds)
    pbar = tqdm(total=num_epochs * (k_folds) * (len(folds[0][0]) // batch_size))
    for fold, (train_ids, test_ids) in enumerate(folds):
        # Print
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Observations encoder
        OE = Encoder(game_state_dim + player_output_dim, obs_encoding_size).to(DEVICE)

        # Predictor
        P = Predictor(obs_encoding_size, g_hidden_size, player_output_dim).to(DEVICE)

        OE.float()
        P.float()

        # optimizer
        p_optimizer = optim.Adam(chain(P.parameters(), OE.parameters()), lr=lr_g, betas=(beta1, beta2))

        # loss
        mseloss = nn.MSELoss().to(DEVICE)

        OE.train()
        P.train()
        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            OE.train()
            P.train()
            for batch_i, data in enumerate(trainloader):
                # extract data
                observed_keys, real_future_keys, observed_game_state = extract_data(data)

                # Zero the gradients
                p_optimizer.zero_grad()

                # forward pass
                # encode
                game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

                # sample latent vector
                # z = torch.randn((game_state_encoding.shape[0], 1, g_latent_dim), device=DEVICE)

                # predict
                predicted_future_keys = P(game_state_encoding)

                # loss
                p_loss = mseloss(predicted_future_keys, real_future_keys)

                # backward pass
                p_loss.backward()

                # optimization
                p_optimizer.step()

                # Print statistics
                current_loss += p_loss.item()
                if batch_i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (batch_i + 1, current_loss / 500))
                    current_loss = 0.0
                pbar.update()

            # Saving the model
            save_path = f'{SAVEDIR}{MODEL_SAVE_FILENAME}-{fold}-epoch-{epoch}.pth'
            save_model(save_path)

            # Evaluation for this fold
            # Print about testing
            print('Starting testing')
            OE.eval()
            P.eval()
            matches, total, correct_presses, total_presses, actual_total_presses = 0, 0, 0, 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for ba_i, data in enumerate(testloader, 0):
                    # extract data
                    observed_keys, real_future_keys, observed_game_state = extract_data(data)

                    # encode
                    game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

                    # sample latent vector
                    # z = torch.randn((game_state_encoding.shape[0], 1, g_latent_dim), device=DEVICE)

                    # predict
                    predicted_future_keys = P(game_state_encoding)

                    # Set total and correct
                    binary_prediction = (predicted_future_keys > 0).float()
                    binary_target = (real_future_keys > 0).float()

                    total += torch.numel(binary_target)
                    total_presses += torch.sum(binary_prediction).item()  # count positives
                    correct_presses += torch.sum(torch.mul(binary_prediction,
                                                           binary_target)).item()  # multiply so only correct presses remain
                    actual_total_presses += torch.sum(binary_target).item()  # count real positives
                    matches += torch.numel(binary_target) - torch.sum(
                        torch.abs(binary_prediction - binary_target)).item()

                # Print accuracy
                accuracy = 100.0 * matches / total
                precision = 100.0 * correct_presses / total_presses
                recall = 100.0 * correct_presses / actual_total_presses
                print('Accuracy for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, accuracy))
                print('Precision for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, precision))
                print('Recall for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, recall))
                print('--------------------------------')
                results[fold][epoch][0] = accuracy
                results[fold][epoch][1] = precision
                results[fold][epoch][2] = recall

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    fold_avg = torch.mean(results, dim=1)  # k_folds x 3
    for fold in range(k_folds):
        print(f'Fold {fold + 1}: A={fold_avg[fold][0]}% P={fold_avg[fold][1]}% R={fold_avg[fold][2]}%')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_epochs} EPOCHS')
    print('--------------------------------')
    epoch_avg = torch.mean(results, dim=0)  # num_epochs x 3
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}: A={epoch_avg[epoch][0]}% P={epoch_avg[epoch][1]}% R={epoch_avg[epoch][2]}%')
    print(f'K-FOLD CROSS VALIDATION GLOBAL RESULTS')
    print('--------------------------------')
    global_avg = torch.mean(results, dim=[0, 1])  # 3
    print(f'A={global_avg[0]}% P={global_avg[1]}% R={global_avg[2]}%')
    pbar.close()
