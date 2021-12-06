import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from torch.utils.data import DataLoader
from datasets import keys_position_tanh_dataset

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

data_loader = DataLoader(keys_position_tanh_dataset, batch_size=32, shuffle=True)

df = pd.DataFrame()

board_counts = np.zeros((20, 20))

row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    for i in range(len(real_keys)):
        state = game_state[i]
        keys = real_keys[i]

        if state[0] != 0:  # only robot_mode == 0
            continue

        x = min(math.ceil(10 * (state[1] + 0.95)), 19)
        y = min(math.ceil(10 * (state[2] + 0.95)), 19)
        if keys[0].item() > 0:
            board_counts[x][19 - y] += 1

board_counts = board_counts / board_counts.max()

ax = sns.heatmap(board_counts)
plt.show()
