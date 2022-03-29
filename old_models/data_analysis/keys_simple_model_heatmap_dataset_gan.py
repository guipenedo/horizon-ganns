import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

import torch
from torch.utils.data import DataLoader
from datasets import keys_position_tanh_dataset

from models.cgan_dp_keys_tanh import gan_model, G
from old_models.models_util import DEVICE

gan_model.load_model()
G.eval()

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

data_loader = DataLoader(keys_position_tanh_dataset, batch_size=32, shuffle=True)

df = pd.DataFrame()

board_counts = np.zeros((20, 20))
r_board_counts = np.zeros((20, 20))
total_board_counts = np.zeros((20, 20))

row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    game_state[:, 1:3] += torch.from_numpy(np.random.uniform(low=-0.1, high=0.1, size=game_state[:, 1:3].shape)).to(DEVICE)
    game_state[:, 3:7] = 0

    res = G.generate(game_state)

    for i in range(len(res)):
        state = game_state[i]
        keys = res[i]
        r_keys = real_keys[i]

        if state[0] != 0:  # only robot_mode == 0
            continue

        x = min(math.ceil(10 * (state[1] + 0.95)), 19)
        y = min(math.ceil(10 * (state[2] + 0.95)), 19)
        if keys[0].item() > 0:
            board_counts[x][19 - y] += 1
        if r_keys[0].item() > 0:
            r_board_counts[x][19 - y] += 1
        total_board_counts[x][19 - y] += 1

board_counts = board_counts

sns.heatmap(board_counts, vmax=1200)
plt.title("Generated keys")
plt.show()


r_board_counts = r_board_counts

sns.heatmap(r_board_counts, vmax=1200)
plt.title("Dataset keys")
plt.show()

sns.heatmap(total_board_counts, vmax=5000)
plt.title("Total samples")
plt.show()
