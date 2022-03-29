"""
    This file generates geographical frequency distributions for each key (see S3 report)
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from models.rcgan import *

# config
fold = 0
generations_per_sample = 10
save_file = f"{SAVE_DIR}positional_keys.pt"

print("Loading save")

save = torch.load(save_file)

print("Loaded save")

totals = save["totals"]
keypresses = save["keypresses"]
freqs = torch.nan_to_num(keypresses / totals.unsqueeze(dim=1))
freqs = freqs.cpu()

board_counts = np.zeros((10, 10, 5))
totals_map = np.zeros((10, 10))
for y in range(10):
    yy = 9 - y
    for x in range(10):
        board_counts[y][x] = freqs[x * 10 + yy]

for y in range(10):
    yy = 9 - y
    for x in range(10):
        totals_map[y][x] = totals[x * 10 + yy]

keynames = ["front", "back", "left", "right", "space"]
for key in range(5):
    ax = sns.heatmap(100 * board_counts[:, :, key], cbar_kws={'label': 'percentage of predictions with keypress'})
    ax.set(xlabel='X discretized region', ylabel='Y discretized region')
    plt.title(f"Generated \"{keynames[key]}\" keypress frequencies")
    plt.show()
