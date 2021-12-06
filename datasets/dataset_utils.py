import torch

DATA_NORM_FULL_FILENAME = "processed_data/concat_data_norm.csv"
DATA_FILTERED_FILENAME = "processed_data/concat_data_norm_filtered.csv"
DATA_NO_KEYS_FILENAME = "processed_data/concat_data_norm_filtered_no_key.csv"
DATA_NORM_TANH_FULL_FILENAME = "processed_data/concat_data_norm_tanh.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_df_column_names(df, columns):
    return torch.from_numpy(df[columns].to_numpy()).float().to(DEVICE)

