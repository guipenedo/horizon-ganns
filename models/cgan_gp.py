import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from datasets import SplitSequencesDataset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_NAME = "cgan_gp"

parser = argparse.ArgumentParser()

# Discriminator parameters
parser.add_argument("--d_hidden_size", type=int, default=256, help="biggest size of hidden layers in discriminator")

# Generator parameters
parser.add_argument("--g_hidden_size", type=int, default=512, help="biggest size of hidden layers in generator")
parser.add_argument("--g_latent_dim", type=int, default=100, help="dimension of latent space")

# Optimization parameter
parser.add_argument("--lr_g", type=float, default=0.0001, help="generator learning rate") # 0.00001
parser.add_argument("--lr_d", type=float, default=0.0001, help="discriminator learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="discriminator learning rate")
parser.add_argument("--beta2", type=float, default=0.999, help="discriminator learning rate")
parser.add_argument("--lambda_gp", type=float, default=0.01, help="loss weight for gradient penalty") # 0.1
parser.add_argument("--n_critic", type=int, default=5, help="number of times to train discriminator per "
                                                            "generator training iteration")

# training and validation parameters
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--k_folds", type=int, default=4, help="number of folds for cross validation")
parser.add_argument("--samples_to_validate", type=int, default=5, help="amount of samples for validation")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--cache_dataset", default=True, action=argparse.BooleanOptionalAction,
                    help="cache dataset for faster training time on supercomputer")

opt = parser.parse_args()

# Data config
game_state_columns = ["robot_mode", "robot_x", "robot_y", "robot_theta"]
player_output_columns = ["key_front", "key_back", "key_left", "key_right", "key_space"]
game_state_dim = len(game_state_columns)
player_output_dim = len(player_output_columns)

dataset = SplitSequencesDataset(DATASET_DIR,
                                x_columns=player_output_columns,
                                y_columns=game_state_columns,
                                smooth_labels=False, cache=opt.cache_dataset, cache_filename="observations_cache")


class Generator(nn.Module):
    def __init__(self, condition_size, hidden_size, latent_dim, output_size):
        super(Generator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(condition_size + latent_dim, hidden_size),
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

    def forward(self, last_observation, z):  # N x 1 x encoding_size
        encoding_with_z = torch.cat((last_observation, z), dim=2)
        # FC, final size = output_size
        out = self.fc_layers(encoding_with_z)  # N x 1 x output_size
        return out


class Discriminator(nn.Module):
    def __init__(self, condition_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(output_size + condition_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, last_observation, future_action):
        prediction_with_encoding = torch.cat((last_observation, future_action), dim=2)
        return self.fc_layer(prediction_with_encoding)


# adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L119
def compute_gradient_penalty(D, real_samples, fake_samples, game_state):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand_like(real_samples, device=DEVICE)
    # Get random interpolation between real and fake samples
    # print("alpha size:", alpha.size(), "real_Samples size:", real_samples.size(), "fake samples:", fake_samples.size(), "game state:", game_state.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
    d_interpolates = D(game_state, interpolates)
    fake = torch.ones_like(d_interpolates, requires_grad=False, device=DEVICE)
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


def train(save=True):
    # loss

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=opt.k_folds, shuffle=True)

    folds = kfold.split(dataset)
    folds = list(folds)

    pbar_total_per_objective = opt.num_epochs * opt.k_folds * (len(folds[0][0]) // opt.batch_size)

    results = torch.zeros((opt.k_folds, opt.num_epochs, 3))

    # Set fixed random number seed
    torch.manual_seed(42)

    pbar = tqdm(total=pbar_total_per_objective)
    for fold, (train_ids, test_ids) in enumerate(folds):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=test_subsampler)

        # Generator
        G = Generator(game_state_dim + player_output_dim, opt.g_hidden_size, opt.g_latent_dim, player_output_dim).to(DEVICE)

        # Discriminator
        D = Discriminator(game_state_dim + player_output_dim, opt.d_hidden_size, player_output_dim).to(DEVICE)

        G.float()
        D.float()

        # optimizer
        g_optimizer = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
        d_optimizer = optim.Adam(D.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

        # save losses
        g_losses = []
        d_losses = []

        # save only best parameters
        best_D, best_G = None, None
        best_f1 = None

        # Run the training loop for defined number of epochs
        for epoch in range(opt.num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch}')

            # Set current loss value
            current_d_loss = 0.0
            current_g_loss = 0.0

            # Iterate over the DataLoader for training data
            G.train()
            D.train()
            for batch_i, data in enumerate(trainloader):
                # extract data
                observed_keys, real_future_keys, observed_game_state = extract_data(data)
                last_observed_keys = observed_keys[:, -1:, :]
                last_observed_game_state = observed_game_state[:, -1:, :]

                last_observation = torch.cat((last_observed_keys, last_observed_game_state), dim=2)

                """
                    Discriminator
                """
                # zero gradients
                d_optimizer.zero_grad()

                # critic on fake keys
                with torch.no_grad():
                    # sample latent vector
                    z = torch.randn((last_observation.shape[0], 1, opt.g_latent_dim), device=DEVICE)

                    # predict
                    predicted_future_keys = G(last_observation, z)

                # validity
                validity_gt = D(last_observation, real_future_keys)
                validity_predicted = D(last_observation, predicted_future_keys)

                # loss
                gradient_penalty = compute_gradient_penalty(D, real_future_keys, predicted_future_keys, last_observation)

                d_loss = -torch.mean(validity_gt) + torch.mean(validity_predicted) + opt.lambda_gp * gradient_penalty

                # save loss
                d_losses.append(d_loss.item())

                # backward pass
                d_loss.backward()

                # optimization
                d_optimizer.step()

                # save loss
                current_d_loss += d_loss.item()

                if batch_i % opt.n_critic == 0:
                    """
                        Generator
                    """
                    # zero gradients
                    g_optimizer.zero_grad()

                    # forward pass
                    # sample latent vector
                    z = torch.randn((last_observation.shape[0], 1, opt.g_latent_dim), device=DEVICE)

                    # predict
                    predicted_future_keys = G(last_observation, z)

                    # validity
                    validity_predicted = D(last_observation, predicted_future_keys)

                    # loss
                    g_loss = -torch.mean(validity_predicted)

                    # save losses
                    g_losses.append(g_loss.item())

                    # backward pass
                    g_loss.backward()

                    # optimization
                    g_optimizer.step()

                    current_g_loss += g_loss.item()
                # Print statistics
                if batch_i % 500 == 499:
                    print(f'Loss after mini-batch {batch_i + 1}: g={current_g_loss / 500 * opt.n_critic} d={current_d_loss / 500}')
                    current_g_loss = 0.0
                    current_d_loss = 0.0
                pbar.update()

            # Evaluation for this fold
            # Print about testing
            G.eval()
            D.eval()
            with torch.no_grad():

                validation_scores = []
                # Iterate over the test data and generate predictions
                for ba_i, data in enumerate(testloader, 0):
                    # extract data
                    observed_keys, real_future_keys, observed_game_state = extract_data(data)
                    last_observed_keys = observed_keys[:, -1:, :]
                    last_observed_game_state = observed_game_state[:, -1:, :]
                    last_observation = torch.cat((last_observed_keys, last_observed_game_state), dim=2)

                    # define target for this batch
                    binary_target = (real_future_keys > 0).view(-1).cpu()

                    # we use the best out of "samples_to_validate" results for this particular observation
                    best_scores = None

                    for sample_i in range(opt.samples_to_validate):
                        # sample latent vector
                        z = torch.randn((last_observation.shape[0], 1, opt.g_latent_dim), device=DEVICE)

                        # predict
                        predicted_future_keys = G(last_observation, z)

                        # Set total and correct
                        binary_prediction = (predicted_future_keys > 0).view(-1).cpu()

                        f1 = f1_score(binary_target, binary_prediction)

                        if best_scores is None or f1 > best_scores[0]:
                            precision = precision_score(binary_target, binary_prediction)
                            recall = recall_score(binary_target, binary_prediction)
                            best_scores = torch.tensor([f1, precision, recall])

                    validation_scores.append(best_scores)
                av_stats = torch.mean(torch.stack(validation_scores), dim=0)
                # Print stats
                print(f"precision={av_stats[1]}, recall={av_stats[2]}, f1={av_stats[0]}")
                print('--------------------------------')

                results[fold][epoch] = av_stats

                if best_f1 is None or av_stats[0] > best_f1:
                    best_f1 = av_stats[0]
                    best_D = D.state_dict()
                    best_G = G.state_dict()

        data = {
            'losses': {
                'g_losses': g_losses,
                'd_losses': d_losses
            },
            'validation': results[fold].cpu().detach().numpy(),
            'generator': best_G,
            'discriminator': best_D,
        }
        if save:
            torch.save(data,
                       SAVE_DIR + f"{MODEL_NAME}_V{opt.samples_to_validate}-F{fold}-BS{opt.batch_size}.pt")

    pbar.close()

    return get_final_validation_stats(results)


if __name__ == "__main__":
    train()
