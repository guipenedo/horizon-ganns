import pandas as pd
from scipy.stats import wasserstein_distance

keys = ["front", "back", "left", "right"]

df_dataset = pd.read_csv("key_counts.csv")
df_generated = pd.read_csv("key_counts_model.csv")


def norm(dataset):
    dt_np = dataset.to_numpy()[:, :4]
    sums = dt_np.sum(axis=0)
    norms = dt_np / sums
    return {keys[i]: norms[:, i] for i in range(len(keys))}


keys_dataset = norm(df_dataset)
keys_generated = norm(df_generated)

for key in keys:
    print(f"Key: {key} | Distance: {wasserstein_distance(keys_dataset[key], keys_generated[key])}")

for key in keys:
    for i in range(min(len(keys_dataset[key]), len(keys_generated[key]))):
        print("Key:", key, "i:", i, "Dataset:", keys_dataset[key][i], "Generated:", keys_generated[key][i])

"""
gan
Key: front | Distance: 0.04833164837186235
Key: back | Distance: 0.03722339141586545
Key: left | Distance: 0.05901513286283793
Key: right | Distance: 0.10258043313368939
wgan_dp_new
Key: front | Distance: 0.006742809700347541
Key: back | Distance: 0.0020699340506820725
Key: left | Distance: 0.0037143731594944207
Key: right | Distance: 0.002987400913947196
"""
