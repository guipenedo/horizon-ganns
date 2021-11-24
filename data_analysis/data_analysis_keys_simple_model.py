import pandas as pd
import numpy as np

keys = ["front", "back", "left", "right", "space"]
K = len(keys)
#df = pd.read_csv("key_counts.csv")

from models.cgan_dp_keys_tanh import gan_model, G, data_loader


key_counts = np.zeros([2] * K)
key_counts_dataset = np.zeros([2] * K)

gan_model.load_model()
G.eval()

row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    counts = G.generate(game_state)
    ri = 0
    for row in counts:
        row_count += 1
        arr = key_counts
        arr2 = key_counts_dataset
        for i in range(K):
            y = 1 if row[i].item() > 0 else 0
            y2 = 1 if real_keys[ri][i].item() > 0 else 0
            if i == K - 1:
                arr[y] += 1
                arr2[y2] += 1
            arr = arr[y]
            arr2 = arr2[y2]
        ri += 1

pct_diffs = np.abs(key_counts - key_counts_dataset) * 100 / row_count
print(f"Max: {np.amax(pct_diffs)}; Mean: {np.mean(pct_diffs)}")

"""
cgan_dp_keys:
10 training epochs: Max: 9.328401719991254; Mean: 0.5887644850958385
20 training epochs: Max: 9.471248451279061; Mean: 0.6611216383645506

cgan_dp_keys_tanh (with lr tuning):
10 training epochs: Max: 2.3387508199110854; Mean: 0.28881367976095035
20 training epochs: Max: 12.746519932949493; Mean: 0.8012808833175424
"""
