import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from datasets.dataset_utils import DEVICE

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

from models.cgan_dp_keys_tanh import gan_model, G

df = pd.DataFrame()

gan_model.load_model()
G.eval()

board_counts = np.zeros((20, 20))

game_state = torch.zeros((50, 7)).to(DEVICE)
game_state[:, 0] = 0
# # theta: up (pi/2)
# game_state[:, 3] = normalize(-np.pi/2, data_ranges["robot_theta"])
# # going up
# game_state[:, 5] = normalize(-2, data_ranges["robot_y_diff"])

for i in range(1000):
    print("Iter i=", i + 1, "/1000")
    for x in range(20):
        game_state[:, 1] = 0.1 * x - 1 + 0.05
        for y in range(20):
            game_state[:, 2] = 0.1 * y - 1 + 0.05

            res = G.generate(game_state)
            for row in res:
                if row[0].item() > 0:  # 1 front
                    board_counts[x][19-y] += 1

board_counts = board_counts / board_counts.max()

ax = sns.heatmap(board_counts)
plt.title("Generated keys")
plt.show()
