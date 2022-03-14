from normalize_utils import normalize_csv_row
import pandas as pd
import os
from os.path import sep

OBS_SECS = 5

output_dir = os.getenv("OUTPUT_DIR", "processed_data/observations_5s")
os.makedirs(output_dir, exist_ok=True)

raw_data_folder = os.getenv("INPUT_DIR", "recorded_csv_data2")


def get_name_list(base, count):
    return [base + "_" + str(i) for i in range(1, count + 1)]


def join_word_list(base, word_list):
    return [base + "_" + w for w in word_list]


def bitmask_to_one_hot(bitmask, size):
    bitmask = int(bitmask)
    if bitmask < 0:
        bitmask = 0
    return [int(i) if int(i) > 0 else -1 for i in str(bitmask).rjust(size, "0")]


def get_columns_as_list(df_input, columns):
    return [df_input[column] for column in columns]


def get_input_counts_as_list(input_row, name, values, max=9999):
    return [min(str(input_row[name]).count(value), max) for value in values]


def get_has_input_as_list(input_row, name, values):
    return [1 if str(input_row[name]).count(value) > 0 else -1 for value in values]


def get_diff_pos(input_row, input_prev_row):
    return [float(input_row[p]) - float(input_prev_row[p]) for p in ["robot_x", "robot_y", "robot_theta"]]


keys = ["front", "back", "left", "right", "space"]
clicks = ["left", "right", "push", "wrench", *get_name_list("leak", 9), "rm_alarm"]
errors = ["down", "up", "click"]

data_columns = [
    "remaining_time",
    "robot_mode",
    "alarm",
    "robot_x",
    "robot_y",
    "robot_theta",
    *get_name_list("tree", 9),
    "battery_level",
    "temperature",
    "water_robot_tank",
    "water_ground_tank",
    *get_name_list("leak", 9),
    *join_word_list("key", keys),
    *join_word_list("click", clicks),
    *join_word_list("error", errors),
    "shortcuts",
    "water_robot_tank_oh",
    "water_ground_tank_oh",
    "close_to_water"
]

file_counter = 0
for filename in os.listdir(raw_data_folder):
    with open(os.path.join(raw_data_folder, filename), "r") as data_file:
        df = pd.read_csv(data_file, dtype=object)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        file_data_norm = []
        has_keys = False
        for index, row in df.iterrows():
            keypresses = get_has_input_as_list(row, "keys", keys)
            if keypresses.count(-1) != len(keypresses):
                has_keys = True
            norm_data = normalize_csv_row(data_columns, [
                *get_columns_as_list(row, [
                    "remaining_time",
                    "robot_mode",
                    "alarm",
                    "robot_x",
                    "robot_y",
                    "robot_theta"
                ]),
                *bitmask_to_one_hot(row["forest_state"], 9),
                *get_columns_as_list(row, [
                    "battery_level",
                    "temperature",
                    "water_robot_tank",
                    "water_ground_tank"
                ]),
                *bitmask_to_one_hot(row["leaks_state"], 9),
                *keypresses,
                *get_input_counts_as_list(row, "clicks", clicks, 6),
                *get_input_counts_as_list(row, "errors", errors),
                row["shortcuts"],
                1 if float(row["water_robot_tank"]) > 40 else 0,
                1 if float(row["water_ground_tank"]) > 40 else 0,
                1 if float(row["robot_x"]) ** 2 + (float(row["robot_y"]) + 10) ** 2 < 4 else 0
            ])
            file_data_norm.append(norm_data)
            prev_row = row
        if file_counter % 10 == 0:
            print(file_counter, "files processed")
        file_counter += 1
        if has_keys:
            i = 0
            while i + OBS_SECS < len(file_data_norm):
                df = pd.DataFrame(file_data_norm[i: OBS_SECS + i + 1], columns=data_columns)
                df.to_csv(output_dir + sep + filename.split(".")[0] + "_" + str(i) + ".csv", index=False)
                i += 1
