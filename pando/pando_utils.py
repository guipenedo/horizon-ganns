import torch

DATASET_DIR = "/fasttmp/g.cabral/observations_5s"
SAVE_DIR = "/scratch/students/g.cabral/saves/"
OPTUNA_PATH = "/scratch/students/g.cabral/optuna.db"
# DATASET_DIR = "../processed_data/observations_5s"
# SAVE_DIR = "../test/saves/"
# OPTUNA_PATH = "../test/optuna.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

nr_observations = 5


def extract_data(data):
    real_keys, game_state = data
    # split observation from current value
    observed_keys = real_keys[:, :nr_observations, :]
    real_future_keys = real_keys[:, nr_observations:, :]

    observed_game_state = game_state[:, :nr_observations, :]

    return observed_keys, real_future_keys, observed_game_state


def update_validation_stats(stats, binary_prediction, binary_target, bin_class=1):
    if bin_class == 0:
        binary_prediction = (binary_prediction == 0).float()
        binary_target = (binary_target == 0).float()
    stats['total_presses'] += torch.sum(binary_prediction).item()  # count positives
    stats['correct_presses'] += torch.sum(torch.mul(binary_prediction,
                                                    binary_target)).item()  # multiply so only correct presses remain
    stats['actual_total_presses'] += torch.sum(binary_target).item()  # count real positives


def compute_stats(validation_stats_0, validation_stats_1):
    precision = 100.0 * validation_stats_1['correct_presses'] / validation_stats_1['total_presses'] if \
        validation_stats_1['total_presses'] > 0 else 0

    recall_1 = 100.0 * validation_stats_1['correct_presses'] / validation_stats_1['actual_total_presses'] if \
        validation_stats_1['actual_total_presses'] > 0 else 0

    recall_0 = 100.0 * validation_stats_0['correct_presses'] / validation_stats_0['actual_total_presses'] if \
        validation_stats_0['actual_total_presses'] > 0 else 0

    accuracy = 0.5 * (recall_1 + recall_0)
    f1 = 2 * precision * recall_1 / (precision + recall_1) if precision + recall_1 > 0 else 0
    return accuracy, precision, recall_1, f1
