import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from datasets.dataset_utils import DEVICE
from normalize_utils import normalize, data_ranges

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

from models.cgan_dp_keys_tanh import gan_model, G, data_loader

df = pd.DataFrame()

gan_model.load_model()
G.eval()

board_counts = np.zeros((20, 20))

game_state = torch.zeros((50, 7)).to(DEVICE)
game_state[:, 0] = 1
# theta: up (pi/2)
game_state[:, 3] = normalize(-np.pi/2, data_ranges["robot_theta"])
# going up
game_state[:, 5] = normalize(-2, data_ranges["robot_x_diff"])

for i in range(100):
    print("Iter i=", i + 1, "/100")
    for x in range(20):
        game_state[:, 1] = 0.1 * x - 1 + 0.5
        for y in range(20):
            game_state[:, 2] = 0.1 * y - 1 + 0.5
            res = G.generate(game_state)
            for row in res:
                if row[0].item() > 0:  # 1 front
                    board_counts[x][19-y] += 1

ax = sns.heatmap(board_counts)
plt.show()
