import os

import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .dataset_utils import get_df_column_names


class SplitSequencesDataset(Dataset):
    def __init__(self, root, x_columns=None, y_columns=None, smooth_labels=False):
        self.x_columns = x_columns
        self.y_columns = y_columns
        if not x_columns:
            self.x_columns = ["key_front", "key_back", "key_left", "key_right", "key_space"]
        if not y_columns:
            self.y_columns = ["robot_mode", "robot_x", "robot_y", "robot_theta"]
        self.root = root
        self.smooth_labels = smooth_labels
        self.data_files = os.listdir(root)

    def __getitem__(self, idx):
        df = pd.read_csv(self.root + os.path.sep + self.data_files[idx])
        # generated data
        x = get_df_column_names(df, self.x_columns)
        # game_state
        y = get_df_column_names(df, self.y_columns)
        if self.smooth_labels:
            x = 0.9 * x
        return x, y

    def __len__(self):
        return len(self.data_files)


# dataset = SplitSequencesDataset("processed_data/normalized_sequences", smooth_labels=True)
# dataset10s = SplitSequencesDataset("processed_data/normalized_sequences_10s", smooth_labels=True)
# dataset30s = SplitSequencesDataset("processed_data/normalized_sequences_30s", smooth_labels=True)


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, lens
