import pandas as pd

keys = ["front", "back", "left", "right", "space"]
K = len(keys)

from models.lstm_30s import gan_model, G, data_loader

df = pd.DataFrame()

gan_model.load_model()
G.eval()

data_rows = []


def add_rows(keys_d, state, generator=1):
    for i in range(len(keys_d)):
        exp = keys_d[i]
        for j in range(len(exp)):
            dict = {'generator': generator, 'robot_mode': state[i][j][0].item()}
            for k in range(K):
                y = 1 if exp[j][k].item() > 0 else 0
                dict[keys[k]] = y
            data_rows.append(dict)


row_count = 0
for batch_i, (real_keys, game_state) in enumerate(data_loader):
    robot_mode = game_state[:, :, 0:1]

    counts = G.generate(robot_mode)

    add_rows(counts, robot_mode, generator=1)

    add_rows(real_keys, robot_mode, generator=0)

    if batch_i % 10 == 0:
        print(batch_i)

df = pd.DataFrame(data_rows)
df.to_csv("full_results_lstm.csv", index=False)

suc = df.groupby(["generator", "robot_mode"]).sum()
counts = df.groupby(["generator", "robot_mode"]).count()

freqs = df.groupby(["generator", "robot_mode"]).mean()

print(suc)
print(counts)
print(freqs)
