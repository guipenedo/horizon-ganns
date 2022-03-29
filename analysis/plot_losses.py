import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # g_adv_losses = []
    g_variety_losses = []
    g_adv_losses = []
    d_losses = []
    f1_scores = []
    for i in range(4):
        try:
            save_filename = f"saves/rcgan_5V5-F{i}-BS256.pt"
            save = torch.load(save_filename, map_location='cpu')
            losses = save["losses"]
            g_adv_losses.append(np.array(losses["g_adv_losses"]))
            g_variety_losses.append(np.array(losses["g_variety_losses"]))
            d_losses.append(np.array(losses["d_losses"]))
            # g_losses.append(np.array(losses["g_losses"]))
            f1_scores.append(save["validation"][:, 0])
        except Exception:
            continue
    g_adv_losses = np.mean(np.stack(g_adv_losses, axis=1), axis=1)
    g_variety_losses = np.mean(np.stack(g_variety_losses, axis=1), axis=1)
    d_losses = np.mean(np.stack(d_losses, axis=1), axis=1)
    f1_scores = np.mean(np.stack(f1_scores, axis=1), axis=1)
    avg_g_adv_losses = np.average(g_adv_losses.reshape(-1, g_adv_losses.size // 100), axis=1)
    avg_g_variety_losses = np.average(g_variety_losses.reshape(-1, g_variety_losses.size // 100), axis=1)
    avg_d_losses = np.average(d_losses.reshape(-1, d_losses.size // 100), axis=1)
    epochs = np.arange(1, 101)
    plt.plot(epochs, avg_g_adv_losses, 'g', label="G adv losses")
    plt.plot(epochs, avg_g_variety_losses, 'r', label="G v losses")
    plt.plot(epochs, avg_d_losses, 'b', label="D adv losses")
    plt.plot(epochs, avg_g_adv_losses + avg_g_variety_losses, 'black', label="G losses")
    plt.plot(epochs, f1_scores, 'orange', label="F1-score")
    plt.title('Training losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

