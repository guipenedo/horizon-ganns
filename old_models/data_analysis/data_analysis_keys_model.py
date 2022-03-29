import pandas as pd

keys = ["front", "back", "left", "right", "space"]

#df = pd.read_csv("key_counts.csv")

from models.cgan_dp_keys import *

keys_counts = {k: 11 * [0] for k in keys}
front_back_combo = [[0 for i in range(11)] for j in range(11)]

gan_model.load_model()
G.eval()


for i in range(100000):
    counts = G.generate().argmax(dim=2)#.cpu()
    for row in counts:
        for x in range(4):
            keys_counts[keys[x]][row[x].item()] += 1
        front_back_combo[row[0].item()][row[1].item()] += 1

df = pd.DataFrame(keys_counts)
df.to_csv("key_counts_model.csv", index=False)

df = pd.DataFrame(front_back_combo)
df.to_csv("front_back_combo_model.csv", index=False)
