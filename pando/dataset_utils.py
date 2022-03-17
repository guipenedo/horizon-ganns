import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_df_column_names(df, columns):
    return torch.from_numpy(df[columns].to_numpy()).float().to(DEVICE)

