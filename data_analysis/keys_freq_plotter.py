import pandas as pd
from matplotlib import pyplot as plt

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

dist_dataset = keys_generated

fig = plt.figure()
plt.title("Generated data")
for i, key in enumerate(keys):
    fig = plt.subplot(2, 2, i + 1)
    plt.bar(list(range(11)), dist_dataset[key])
    plt.title(f"Key: {key}")
    plt.xlabel(f"# keypresses")
    plt.ylabel("# samples")
plt.tight_layout()
plt.show()
