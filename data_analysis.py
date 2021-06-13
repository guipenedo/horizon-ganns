import pandas as pd
import os
from matplotlib import pyplot as plt

import matplotlib.pyplot as pyplot

raw_data_folder = "recorded_csv_data2"

keys = ["front", "back", "left", "right", "space"]

keys_counts = {k: 40*[0] for k in keys}
no_clicks_files = 0
file_counter = 0
for filename in os.listdir(raw_data_folder):
    with open(os.path.join(raw_data_folder, filename), "r") as data_file:
        df = pd.read_csv(data_file, dtype=object)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        has_clicks = False
        keys_counts_file = {k: 40 * [0] for k in keys}
        for index, row in df.iterrows():
            c_s = str(row["keys"])
            for c in keys:
                x = c_s.count(c)
                keys_counts_file[c][x] += 1
                if x > 0:
                    has_clicks = True
        if file_counter % 10 == 0:
            print(file_counter, "files processed")
        file_counter += 1
        if not has_clicks:
            no_clicks_files += 1
        else:
            for c in keys:
                for i in range(40):
                    keys_counts[c][i] += keys_counts_file[c][i]

for key in keys:
    fig = pyplot.figure()
    plt.bar(list(range(40)), keys_counts[key])
    plt.title(f"Key: {key}")
    plt.xlabel(f"# keypresses")
    plt.ylabel("# samples")
    plt.show()

# 20% of the files have no keypresses whatsoever
