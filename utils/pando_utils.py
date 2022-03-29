import torch

# DATASET_DIR = "/fasttmp/g.cabral/observations_5s"
# SAVE_DIR = "/scratch/students/g.cabral/saves/"
# OPTUNA_PATH = "/scratch/students/g.cabral/optuna.db"
DATASET_DIR = "processed_data/observations_5s"
SAVE_DIR = "models/saves/"
OPTUNA_PATH = "optuna.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

nr_observations = 5


def extract_data(data):
    real_keys, game_state = data
    # split observation from current value
    observed_keys = real_keys[:, :nr_observations, :]
    real_future_keys = real_keys[:, nr_observations:, :]

    observed_game_state = game_state[:, :nr_observations, :]

    return observed_keys, real_future_keys, observed_game_state


def get_final_validation_stats(results):
    k_folds = results.shape[0]
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    fold_avg = torch.mean(results, dim=1)  # k_folds x 3
    for fold in range(k_folds):
        print(
            f'Fold {fold + 1}: F1={fold_avg[fold][0]} P={fold_avg[fold][1]} R={fold_avg[fold][2]}')
    num_epochs = results.shape[1]
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_epochs} EPOCHS')
    print('--------------------------------')
    epoch_avg = torch.mean(results, dim=0)  # num_epochs x 3
    for epoch in range(num_epochs):
        print(
            f'Epoch {epoch + 1}: F1={epoch_avg[epoch][0]} P={epoch_avg[epoch][1]} R={epoch_avg[epoch][2]}')
    max_v, max_i = torch.max(epoch_avg, dim=0)
    print(f"Highest f1_score={max_v[0]} at iteration {max_i[0] + 1}")
    return max_v[0]
