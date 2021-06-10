import math

KEYS_MAX_VALUE = 40  # actual one is 37
CLICK_LEAK_MAX_VALUE = 6
KEYS_COLUMNS = range(31, 36)
CLICK_LEAK_COLUMNS = range(40, 49)

data_ranges = {
    0: [0, 600],  # remaining_time
    2: [-1, 6],  # alarm
    3: [-20, 20],  # robot_x
    4: [-20, 20],  # robot_y
    5: [-math.pi, math.pi],  # robot_theta
    6: [-40, 40],  # robot_x_diff
    7: [-40, 40],  # robot_y_diff
    8: [-2 * math.pi, 2 * math.pi],  # robot_theta_diff
    18: [0, 100],  # battery_level,
    19: [20, 240],  # temperature
    20: [0, 100],  # water_robot_tank
    21: [0, 100],  # water_ground_tank
    36: [0, 7],  # click_left
    37: [0, 7],  # click_right
    38: [0, 32],  # click_push
    39: [0, 32]  # click_wrench
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


def normalize_csv_row(row):
    row = row.copy()
    for i in data_ranges:
        row[i] = normalize(row[i], data_ranges[i])
    for i in KEYS_COLUMNS:
        row[i] = normalize(row[i], [0, KEYS_MAX_VALUE])
    for i in CLICK_LEAK_COLUMNS:
        row[i] = normalize(row[i], [0, CLICK_LEAK_MAX_VALUE])
    return row
