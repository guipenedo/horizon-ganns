from normalize_utils import undo_state_normalization, undo_keys_normalization, undo_water_normalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_losses():
    df = pd.read_csv("processed_data/cgan_hdcas_losses.csv").to_numpy().transpose()
    plt.figure()
    xs = list(range(len(df[0])))
    plt.plot(xs, df[0], label="G loss")
    plt.plot(xs, df[1], label="D loss")
    plt.legend()
    plt.show()


def plot_sample(gen_data, state, isNormalized=True):
    gen_data = gen_data.float()
    state = state.detach().clone()
    if isNormalized:
        undo_state_normalization(state)
        undo_keys_normalization(gen_data)

    time = int(state[0])
    auto_mode = state[1] == 1.0
    battery = int(state[18])
    water = int(state[19])

    robot_x = state[3]
    robot_y = state[4]
    robot_theta = state[5]
    robot_x_diff = state[6]
    robot_y_diff = state[7]
    robot_x_prev = robot_x - robot_x_diff
    robot_y_prev = robot_y - robot_y_diff

    plt.figure()

    # trees
    treesX = [4.55076, -0.7353, -15.10146, -2.6019, -1.33158, 16.58292, 16.87086, 0.6078, -16.65378]
    treesY = [14.66826, 14.75052, 15.76476, 6.7425, 10.02042, -12.5847, -16.01952, -16.23906, -16.23906]
    plt.plot(treesX, treesY, 'g^', markersize=10)
    # fires
    for i in range(9):
        if state[9 + i] == 1.0:
            plt.plot(treesX[i], treesY[i], 'r^', markersize=5)

    # water square
    plt.plot(0, -10, 'bs')
    # battery square
    plt.plot(16, 16.29, 'rp')

    # finally, the robot
    plt.plot(robot_x, robot_y, marker=(3, 0, -90 + np.degrees(robot_theta)), color="black", markersize=15)
    plt.plot([robot_x_prev, robot_x], [robot_y_prev, robot_y], 'k-')

    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figtext(0.2, 0.96, f"Time: {time}/600", ha="center", fontsize=12)
    plt.figtext(0.4, 0.96, f"Battery: {battery}%", ha="center", fontsize=12,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 3})
    plt.figtext(0.6, 0.96, f"Water: {water}%", ha="center", fontsize=12,
                bbox={"facecolor": "blue", "alpha": 0.5, "pad": 3})
    plt.figtext(0.75, 0.96, "Automatic" if auto_mode else "Manual", ha="left", fontsize=12,
                bbox={"facecolor": "red" if auto_mode else "grey", "alpha": 0.5, "pad": 3})
    # key_front, key_back, key_left, key_right, key_space
    gen_data = gen_data.int()
    plt.figtext(0.5, 0.9,
                f"Front: {gen_data[0]} | Back: {gen_data[1]} | Left: {gen_data[2]} | Right: {gen_data[3]} | Space: {gen_data[4]}",
                ha="center", fontsize=12)
    plt.show()


def plot_water_sample(gen_data, state, isNormalized=True):
    gen_data = gen_data.float()
    state = state.float()
    if isNormalized:
        undo_water_normalization(gen_data)

    auto_mode = state[0] == 1.0
    water_robot = state[10] == 1.0
    water_ground = state[11] == 1.0
    close_water = state[12] == 1.0

    plt.figure()
    # leaks
    for i in range(9):
        if state[1 + i] == 1.0:
            plt.plot(1 + 4*(i % 3), 9 - 4*(i // 3), 'bo', markersize=5)
        plt.text(1 + 4*(i % 3), 8.5 - 4*(i // 3), str(int(gen_data[4 + i])))

    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.text(1, 9.5, f"Water robot: {'Nominal' if water_robot else 'Low'}", ha="left", fontsize=12,
                bbox={"facecolor": "blue", "alpha": 0.5, "pad": 3})
    plt.text(9, 9.5, f"Water ground: {'Nominal' if water_ground else 'Low'}", ha="right", fontsize=12,
             bbox={"facecolor": "blue", "alpha": 0.5, "pad": 3})

    plt.figtext(0.25, 0.96, f"Close to water: {'Yes' if close_water else 'No'}", ha="left", fontsize=12,
                bbox={"facecolor": "blue", "alpha": 0.5, "pad": 3})
    plt.figtext(0.75, 0.96, "Automatic" if auto_mode else "Manual", ha="right", fontsize=12,
                bbox={"facecolor": "red" if auto_mode else "grey", "alpha": 0.5, "pad": 3})
    # click_left click_right click_push	click_wrench
    gen_data = gen_data.int()
    plt.figtext(0.5, 0.9,
                f"Left: {gen_data[0]} | Right: {gen_data[1]} | Push: {gen_data[2]} | Wrench: {gen_data[3]}",
                ha="center", fontsize=12)
    plt.show()
