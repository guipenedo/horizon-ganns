import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pando_utils import *
from split_sequences_dataset import SplitSequencesDataset
from itertools import chain
from sklearn.model_selection import KFold

# Observations parameters
# obs_encoding_size = 128

# Discriminator parameters
# d_hidden_size = 256

# Generator parameters
# g_hidden_size = 512
g_latent_dim = 100

# Optimization parameters
# l2_loss_weight = 1
# variety_loss_k = 5
# lr_g = 0.0001
# lr_d = 0.0004
beta1 = 0.9
beta2 = 0.999

# Data config
game_state_columns = ["robot_mode", "robot_x", "robot_y", "robot_theta"]
player_output_columns = ["key_front", "key_back", "key_left", "key_right", "key_space"]
game_state_dim = len(game_state_columns)
player_output_dim = len(player_output_columns)
batch_size = 256

dataset = SplitSequencesDataset(DATASET_DIR,
                                x_columns=player_output_columns,
                                y_columns=game_state_columns,
                                smooth_labels=False, cache=True)


class Encoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=2):
        super(Encoder, self).__init__()
        # Linear embedding obsv_size x h
        self.embed = nn.Sequential(nn.Dropout(0.3),
                                   nn.Linear(input_size, encoding_size),
                                   nn.LeakyReLU(0.2, inplace=True))
        # The LSTM cell.
        # Input dimension (observations mapped through embedding) is the same as the output
        self.lstm = nn.LSTM(encoding_size, encoding_size, num_layers=n_layers, batch_first=True)

    def forward(self, obsv):
        # embedding
        obsv = self.embed(obsv)
        # Reshape and applies LSTM over a whole sequence or over one single step
        y, _ = self.lstm(obsv)
        return y[:, -1:, :]  # N x 1 x encoding_size


class Generator(nn.Module):
    def __init__(self, encoding_size, hidden_size, latent_dim, output_size):
        super(Generator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(encoding_size + latent_dim, hidden_size),
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

    def forward(self, last_encoded_state, z):  # N x 1 x encoding_size
        encoding_with_z = torch.cat((last_encoded_state, z), dim=2)
        # FC, final size = output_size
        out = self.fc_layers(encoding_with_z)  # N x 1 x output_size
        return out


class Discriminator(nn.Module):
    def __init__(self, encoding_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_size + encoding_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, encoded, future_action):
        prediction_with_encoding = torch.cat((encoded, future_action), dim=2)
        return self.fc_layer(prediction_with_encoding)


MODEL_SAVE_FILENAME = "predictor_gan_v_loss_m_validation_optuna"

trial_count = 0


def save_model(fold, epoch, losses, validation, hyperparams, OE, OE_disc, G, D):
    data = {
        'losses': losses,
        'validation': validation,
        'hyperparams': hyperparams,
        'obs_encoder': OE.state_dict(),
        'obs_encoder_disc': OE_disc.state_dict() if OE_disc else None,
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
    }
    torch.save(data, SAVE_DIR + MODEL_SAVE_FILENAME + f"-{trial_count}-f{fold}-e{epoch}.pt")


# loss
mseloss = nn.MSELoss().to(DEVICE)
adversarial_loss = nn.BCELoss().to(DEVICE)

# Configuration options
k_folds = 4
num_epochs = 10

# amount for validation
samples_to_validate = 5

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

folds = kfold.split(dataset)
folds = list(folds)

pbar_total_per_objective = num_epochs * k_folds * (len(folds[0][0]) // batch_size)


def objective(trial):
    global trial_count
    trial_count += 1
    obs_encoding_size = trial.suggest_int('obs_encoding_size', 16, 128 * 2, step=16)
    d_hidden_size = trial.suggest_int('d_hidden_size', 128, 256 * 2, step=32)
    g_hidden_size = trial.suggest_int('g_hidden_size', 128, 512 * 2, step=64)
    l2_loss_weight = trial.suggest_loguniform('l2_loss_weight', 1e-5, 1)
    variety_loss_k = trial.suggest_int('variety_loss_k', 1, 5)
    lr_g = trial.suggest_loguniform('lr_g', 1e-5, 1e-1)
    lr_d = trial.suggest_loguniform('lr_d', 1e-5, 1e-1)
    separate_encoder = trial.suggest_categorical("separate_encoder", [True, False])

    last_for_each_fold = torch.zeros(k_folds, 4)

    hyperparams = {
        'obs_encoding_size': obs_encoding_size,
        'd_hidden_size': d_hidden_size,
        'g_hidden_size': g_hidden_size,
        'l2_loss_weight': l2_loss_weight,
        'variety_loss_k': variety_loss_k,
        'lr_g': lr_g,
        'lr_d': lr_d,
        'separate_encoder': separate_encoder
    }

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
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Observations encoder
        OE = Encoder(game_state_dim + player_output_dim, obs_encoding_size).to(DEVICE)
        OE_disc = None
        if separate_encoder:
            OE_disc = Encoder(game_state_dim + player_output_dim, obs_encoding_size).to(DEVICE)

        # Generator
        G = Generator(obs_encoding_size, g_hidden_size, g_latent_dim, player_output_dim).to(DEVICE)

        # Discriminator
        D = Discriminator(obs_encoding_size, d_hidden_size, player_output_dim).to(DEVICE)

        OE.float()
        if separate_encoder:
            OE_disc.float()
        G.float()
        D.float()

        # optimizer
        g_optimizer = optim.Adam(chain(G.parameters(), OE.parameters()), lr=lr_g, betas=(beta1, beta2))
        if not separate_encoder:
            d_optimizer = optim.Adam(chain(D.parameters(), OE.parameters()), lr=lr_d, betas=(beta1, beta2))
        else:
            d_optimizer = optim.Adam(chain(D.parameters(), OE_disc.parameters()), lr=lr_d, betas=(beta1, beta2))

        # early stopping
        iters_no_improv = 0

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch}')

            # Set current loss value
            current_d_loss = 0.0
            current_g_loss = 0.0

            # save losses
            g_adversarial_losses = []
            g_variety_losses = []
            d_losses = []

            # Iterate over the DataLoader for training data
            OE.train()
            if separate_encoder:
                OE_disc.train()
            G.train()
            D.train()
            for batch_i, data in enumerate(trainloader):
                # extract data
                observed_keys, real_future_keys, observed_game_state = extract_data(data)

                """
                    Generator
                """
                # zero gradients
                g_optimizer.zero_grad()

                # forward pass
                # encode
                game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

                g_loss = 0
                variety_loss = None
                for k in range(variety_loss_k):
                    # sample latent vector
                    z = torch.randn((game_state_encoding.shape[0], 1, g_latent_dim), device=DEVICE)

                    # predict
                    predicted_future_keys = G(game_state_encoding, z)

                    # validity
                    validity_predicted = D(game_state_encoding, predicted_future_keys)

                    # loss
                    g_loss += adversarial_loss(validity_predicted, torch.ones_like(validity_predicted, device=DEVICE))

                    # computation for variety loss
                    l2_loss = mseloss(predicted_future_keys, real_future_keys)
                    if variety_loss is None or l2_loss.item() < variety_loss.item():
                        variety_loss = l2_loss

                # compute loss
                g_adv_loss = g_loss / variety_loss_k
                g_variety_loss = l2_loss_weight * variety_loss
                g_loss = g_adv_loss + g_variety_loss

                # save losses
                g_adversarial_losses.append(g_adv_loss.item())
                g_variety_losses.append(g_variety_loss.item())

                # backward pass
                g_loss.backward()

                # optimization
                g_optimizer.step()

                """
                    Discriminator
                """
                # zero gradients
                d_optimizer.zero_grad()

                # encode
                game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

                # sample
                with torch.no_grad():
                    # sample latent vector
                    z = torch.randn((game_state_encoding.shape[0], 1, g_latent_dim), device=DEVICE)

                    # predict
                    predicted_future_keys = G(game_state_encoding, z)

                # discriminator encoding
                disc_game_state_encoding = OE_disc(torch.cat((observed_game_state, observed_keys), dim=2)) \
                    if separate_encoder else game_state_encoding

                # validity
                validity_gt = D(disc_game_state_encoding, real_future_keys)
                validity_predicted = D(disc_game_state_encoding, predicted_future_keys)

                # loss
                d_loss = 0.5 * adversarial_loss(validity_predicted,
                                                torch.zeros_like(validity_predicted, device=DEVICE))
                d_loss += 0.5 * adversarial_loss(validity_gt,
                                                 torch.ones_like(validity_gt, device=DEVICE))

                # save loss
                d_losses.append(d_loss.item())

                # backward pass
                d_loss.backward()

                # optimization
                d_optimizer.step()

                # Print statistics
                current_g_loss += g_loss.item()
                current_d_loss += d_loss.item()
                # if batch_i % 500 == 499:
                #     print(f'Loss after mini-batch {batch_i + 1}: g={current_g_loss / 500} d={current_d_loss / 500}')
                #     current_g_loss = 0.0
                #     current_d_loss = 0.0
                pbar.update()

            # Evaluation for this fold
            # Print about testing
            OE.eval()
            G.eval()
            D.eval()
            validation_stats_0 = {
                'correct_presses': 0,
                'total_presses': 0,
                'actual_total_presses': 0
            }
            validation_stats_1 = {
                'correct_presses': 0,
                'total_presses': 0,
                'actual_total_presses': 0
            }
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for ba_i, data in enumerate(testloader, 0):
                    # extract data
                    observed_keys, real_future_keys, observed_game_state = extract_data(data)

                    # encode
                    game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

                    # we use the best out of "samples_to_validate" results for this particular observation
                    best = None
                    best_binary_prediction, best_binary_target = None, None

                    for sample_i in range(samples_to_validate):
                        # sample latent vector
                        z = torch.randn((game_state_encoding.shape[0], 1, g_latent_dim), device=DEVICE)

                        # predict
                        predicted_future_keys = G(game_state_encoding, z)

                        # Set total and correct
                        binary_prediction = (predicted_future_keys > 0).float()
                        binary_target = (real_future_keys > 0).float()

                        prec = torch.sum(torch.mul(binary_prediction, binary_target)).item() / torch.sum(
                            binary_prediction).item() if torch.sum(
                            binary_prediction).item() > 0 else 0

                        if best is None or prec > best:
                            best = prec
                            best_binary_prediction = binary_prediction
                            best_binary_target = binary_target

                    update_validation_stats(validation_stats_1, best_binary_prediction, best_binary_target, 1)
                    update_validation_stats(validation_stats_0, best_binary_prediction, best_binary_target, 0)

                # Print accuracy
                accuracy, precision, recall_1, f1 = compute_stats(validation_stats_0, validation_stats_1)
                print(f"accuracy={accuracy}, precision={precision}, recall={recall_1}, f1={f1}")

                print('--------------------------------')

            save_model(fold, epoch, {
                'g_adv_losses': g_adversarial_losses,
                'g_variety_losses': g_variety_losses,
                'd_losses': d_losses
            }, {
                           'accuracy': accuracy,
                           'precision': precision,
                           'recall': recall_1,
                           'f1': f1
                       }, hyperparams, OE, OE_disc, G, D)
            if last_for_each_fold[fold][3] < f1:
                last_for_each_fold[fold][0] = accuracy
                last_for_each_fold[fold][1] = precision
                last_for_each_fold[fold][2] = recall_1
                last_for_each_fold[fold][3] = f1
                iters_no_improv = 0
                continue
            # f1 is lower - early stopping after 2 iterations worse
            if iters_no_improv >= 2:
                print(f"Early stopping at epoch {epoch}")
                break
            iters_no_improv += 1
    pbar.close()

    fold_avg = torch.mean(last_for_each_fold, dim=0)  # x 4
    f1 = fold_avg[3]
    return f1


study = optuna.create_study(study_name="optimize_gan_variety_loss_m_validation", direction='maximize',
                            load_if_exists=True, storage=f'sqlite:///{OPTUNA_PATH}')
study.optimize(objective, n_trials=200, show_progress_bar=True)

print(study.best_params)
