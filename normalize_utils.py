import math
import sys

KEYS_MAX_VALUE = 10  # actual one is 37
CLICK_LEAK_MAX_VALUE = 6
KEYS_COLUMNS = range(31, 35)  # don't normalize space key, which is already 0 or 1
CLICK_LEAK_COLUMNS = range(40, 49)

data_ranges = {
    "remaining_time": [0, 600],  # remaining_time
    "alarm": [-1, 6],  # alarm
    "robot_x": [-20, 20],  # robot_x
    "robot_y": [-20, 20],  # robot_y
    "robot_theta": [-math.pi, math.pi],  # robot_theta
    "robot_x_diff": [-40, 40],  # robot_x_diff
    "robot_y_diff": [-40, 40],  # robot_y_diff
    "robot_theta_diff": [-2 * math.pi, 2 * math.pi],  # robot_theta_diff
    "battery_level": [0, 100],  # battery_level,
    "temperature": [20, 240],  # temperature
    "water_robot_tank": [0, 100],  # water_robot_tank
    "water_ground_tank": [0, 100],  # water_ground_tank
    # "click_left": [0, 7],  # click_left
    # "click_right": [0, 7],  # click_right
    # "click_push": [0, 32],  # click_push
    # "click_wrench": [0, 32]  # click_wrench
}


# normalize value in [range, range] to [-1, 1]
def normalize(value, range):
    return 2 * (float(value) - range[0]) / (range[1] - range[0]) - 1


def undo_normalization(value, range):
    return 0.5 * (1 + float(value)) * (range[1] - range[0]) + range[0]


def undo_state_normalization(state):
    for i in data_ranges:
        if i >= len(state):
            continue
        state[i] = undo_normalization(state[i], data_ranges[i])


def undo_keys_normalization(keys):
    for i in range(len(keys)):
        keys[i] = int(undo_normalization(keys[i], [0, KEYS_MAX_VALUE]))


def undo_water_normalization(clicks):
    for i in range(4):
        clicks[i] = undo_normalization(clicks[i], data_ranges[i + 36])
    for i in range(4, 13):
        clicks[i] = undo_normalization(clicks[i], [0, CLICK_LEAK_MAX_VALUE])


def normalize_csv_row(data_columns, row1):
    row = row1.copy()
    for i in range(len(data_columns)):
        column = data_columns[i]
        if column in data_ranges:
            row[i] = normalize(row[i], data_ranges[column])
            # print(f"Normalizing {column}. {row1[i]}->{row[i]} range {data_ranges[column]}")
    #sys.exit(0)
    # for i in KEYS_COLUMNS:
    #    row[i] = normalize(min(row[i], KEYS_MAX_VALUE), [0, KEYS_MAX_VALUE])
    #for i in CLICK_LEAK_COLUMNS:
    #    row[i] = normalize(row[i], [0, CLICK_LEAK_MAX_VALUE])
    return row
