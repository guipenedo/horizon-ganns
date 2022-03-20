import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pando_utils import *
from split_sequences_dataset import SplitSequencesDataset
from itertools import chain
from sklearn.model_selection import KFold

# Observations parameters
obs_encoding_size = 128

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
batch_size = 256

dataset = SplitSequencesDataset(DATASET_DIR,
                                x_columns=player_output_columns,
                                y_columns=game_state_columns,
                                smooth_labels=False, cache=True)


class Encoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_layers=2):
        super(Encoder, self).__init__()
        # dropout
        self.dropout = nn.Sequential(nn.Dropout(0.3),
                                     nn.Linear(input_size, encoding_size),
                                     nn.LeakyReLU(0.2, inplace=True))
        # Linear embedding obsv_size x h
        self.embed = nn.Linear(input_size, encoding_size)
        # The LSTM cell.
        # Input dimension (observations mapped through embedding) is the same as the output
        self.lstm = nn.LSTM(encoding_size, encoding_size, num_layers=n_layers, batch_first=True)

    def forward(self, obsv):
        # Linear embedding
        obsv = self.embed(obsv)
        # Reshape and applies LSTM over a whole sequence or over one single step
        y, _ = self.lstm(obsv)
        return y[:, -1:, :]


class Predictor(nn.Module):
    def __init__(self, encoding_size, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

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
        # FC, final size = output_size
        out = self.fc_layers(encoded)  # N x 1 x output_size
        return out


MODEL_SAVE_FILENAME = "predictor_model"


def save_model(fold, epoch, losses, validation, OE, P):
    data = {
        'losses': losses,
        'validation': validation,
        'obs_encoder': OE.state_dict(),
        'predictor': P.state_dict(),
    }
    torch.save(data, SAVE_DIR + MODEL_SAVE_FILENAME + f"-f{fold}-e{epoch}.pt")


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
    num_epochs = 100

    # loss
    mseloss = nn.MSELoss().to(DEVICE)

    # For fold results
    results = torch.zeros((k_folds, num_epochs, 4))

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

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0
            losses = []

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

                # save losses
                losses.append(p_loss.item())

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

            # Evaluation for this fold
            # Print about testing
            print('Starting testing')
            OE.eval()
            P.eval()
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

                    # sample latent vector
                    # z = torch.randn((game_state_encoding.shape[0], 1, g_latent_dim), device=DEVICE)

                    # predict
                    predicted_future_keys = P(game_state_encoding)

                    # Set total and correct
                    binary_prediction = (predicted_future_keys > 0).float()
                    binary_target = (real_future_keys > 0).float()

                    update_validation_stats(validation_stats_1, binary_prediction, binary_target, 1)
                    update_validation_stats(validation_stats_0, binary_prediction, binary_target, 0)

                # Print accuracy
                accuracy, precision, recall_1, f1 = compute_stats(validation_stats_0, validation_stats_1)
                print('Accuracy for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, accuracy))
                print('Precision for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, precision))
                print('Recall for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, recall_1))
                print('F1 for fold %d, epoch %d: %.3f %%' % (fold + 1, epoch + 1, recall_1))
                print('--------------------------------')
                results[fold][epoch][0] = accuracy
                results[fold][epoch][1] = precision
                results[fold][epoch][2] = recall_1
                results[fold][epoch][3] = f1

                # Saving the model
                save_model(fold, epoch, losses, {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall_1,
                    'f1': f1
                }, OE, P)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    fold_avg = torch.mean(results, dim=1)  # k_folds x 4
    for fold in range(k_folds):
        print(f'Fold {fold + 1}: A={fold_avg[fold][0]}% P={fold_avg[fold][1]}% R={fold_avg[fold][2]}% F1={fold_avg[fold][3]}%')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_epochs} EPOCHS')
    print('--------------------------------')
    epoch_avg = torch.mean(results, dim=0)  # num_epochs x 4
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}: A={epoch_avg[epoch][0]}% P={epoch_avg[epoch][1]}% R={epoch_avg[epoch][2]}% F1={epoch_avg[epoch][3]}%')
    print(f'K-FOLD CROSS VALIDATION GLOBAL RESULTS')
    print('--------------------------------')
    global_avg = torch.mean(results, dim=[0, 1])  # 4
    print(f'A={global_avg[0]}% P={global_avg[1]}% R={global_avg[2]}% F1={global_avg[3]}%')
    pbar.close()
