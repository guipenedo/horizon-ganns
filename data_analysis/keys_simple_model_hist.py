import pandas as pd
from statsmodels.multivariate.manova import MANOVA

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

from models.cgan_dp_keys_tanh import gan_model, G, data_loader

df = pd.DataFrame()

gan_model.load_model()
G.eval()

data_rows = []
def add_rows(keys_d, state, generator=1):
    ri = 0
    for row in keys_d:
        dict = {'generator': generator, 'robot_mode': state[ri][0].item()}
        for i in range(K):
            y = 1 if row[i].item() > 0 else 0
            dict[keys[i]] = y
        data_rows.append(dict)
        ri += 1


row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    counts = G.generate(game_state)

    add_rows(counts, game_state, generator=1)

    add_rows(real_keys, game_state, generator=0)

    if batch_i % 10 == 0:
        print(batch_i)

df = pd.DataFrame(data_rows)
df.to_csv("full_results.csv", index=False)
