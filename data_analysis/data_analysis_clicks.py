import pandas as pd
import os
from matplotlib import pyplot as plt

import matplotlib.pyplot as pyplot

raw_data_folder = "recorded_csv_data2"


def get_name_list(base, count):
    return [base + "_" + str(i) for i in range(1, count + 1)]


clicks = ["left", "right", "push", "wrench", *get_name_list("leak", 9), "rm_alarm"]

MAX_COUNT = 40

keys_counts = {k: MAX_COUNT*[0] for k in clicks}
no_clicks_files = 0
file_counter = 0
for filename in os.listdir(raw_data_folder):
    with open(os.path.join(raw_data_folder, filename), "r") as data_file:
        df = pd.read_csv(data_file, dtype=object)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        has_clicks = False
        keys_counts_file = {k: MAX_COUNT * [0] for k in clicks}
        for index, row in df.iterrows():
            c_s = str(row["clicks"])
            for c in clicks:
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
            for c in clicks:
                for i in range(MAX_COUNT):
                    keys_counts[c][i] += keys_counts_file[c][i]

for key in clicks:
    fig = pyplot.figure()
    plt.bar(list(range(MAX_COUNT)), keys_counts[key])
    plt.title(f"Key: {key}")
    plt.xlabel(f"# keypresses")
    plt.ylabel("# samples")
    plt.show()

# 20% of the files have no keypresses whatsoever
