"""
    This file extracts and displays validation data from models' savefiles
"""

import torch
import numpy as np

# saves to load
saves = [
    "rcgan_5V20-F{fold}-BS256",
    "rcgan_5V5-F{fold}-BS256",
    "rcgan_20V20-F{fold}-BS256",
    "rcgan_1V5-F{fold}-BS256",
    "gan_gp_V5-F{fold}-BS256",
]


def get_savename(save, fold):
    return "saves/" + save.replace("{fold}", str(fold)) + ".pt"


def load_save(save, fold):
    return torch.load(get_savename(save, fold))


if __name__ == '__main__':
    for save in saves:
        print(save)
        folds = []
        for i in range(4):
            # try:
            fold_save = load_save(save, i)["validation"]
            folds.append(fold_save)
            best_f1 = np.argmax(fold_save, axis=0)[0]
            print(f"best for fold {i}: {fold_save[best_f1]}")
            # except Exception:
            #     continue
        average = np.stack(folds, axis=1)
        mean_folds = np.mean(average, axis=1)
        best_f1 = np.argmax(mean_folds, axis=0)[0]
        print(mean_folds[best_f1])
        p, r = mean_folds[best_f1][1], mean_folds[best_f1][2]
        print("Averaged F1:", 2 * p * r / (p + r), "vs:", mean_folds[best_f1][0])
