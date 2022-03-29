"""
    This file generates plots with samples and the corresponding GAN keys (see S3 report)
    Change the filter below (line has TODO) to select the type of sample
"""
import numpy as np
from matplotlib import pyplot as plt

from models.rcgan import *

# config
fold = 0
generations_per_sample = 100
save_file = f"{SAVE_DIR}rcgan_5V5-F0-BS256_1.pt"


print("Loading save")

save = torch.load(save_file)

print("Loaded save")

# Observations encoder
OE = Encoder(game_state_dim + player_output_dim, opt.obs_encoding_size).to(DEVICE)

# Generator
G = Generator(opt.obs_encoding_size, opt.g_hidden_size, opt.g_latent_dim, player_output_dim).to(DEVICE)

# Discriminator
D = Discriminator(opt.obs_encoding_size, opt.d_hidden_size, player_output_dim).to(DEVICE)

OE.float()
G.float()
D.float()

OE.load_state_dict(save["obs_encoder"])
D.load_state_dict(save["discriminator"])
G.load_state_dict(save["generator"])

OE.eval()
G.eval()
D.eval()

# Define the K-fold Cross Validator
kfold = KFold(n_splits=opt.k_folds, shuffle=True)

folds = kfold.split(dataset)
folds = list(folds)

train_ids, test_ids = folds[fold]
test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=test_subsampler)

totals = torch.zeros((100,)).cuda()
keypresses = torch.zeros((100, 5)).cuda()

# Iterate over the test data and generate predictions
for data in testloader:
    with torch.no_grad():
        # extract data
        observed_keys, real_future_keys, observed_game_state = extract_data(data)

        # encode
        game_state_encoding = OE(torch.cat((observed_game_state, observed_keys), dim=2))

        for sample_i in range(generations_per_sample):
            # sample latent vector
            z = torch.randn((game_state_encoding.shape[0], 1, opt.g_latent_dim), device=DEVICE)

            # predict
            predicted_future_keys = G(game_state_encoding, z)

            # Set total and correct
            binary_prediction = (predicted_future_keys > 0).float()


    game_state = observed_game_state.detach()
    game_state[:, :, 1:3] = 0.5 * (1 + game_state[:, :, 1:3]) * (40) + -20
    game_state[:, :, 3] = 0.5 * (1 + game_state[:, :, 3]) * (2 * np.pi) + -np.pi
    game_state = game_state.cpu()

    for i in range(game_state.shape[0]):
        auto_mode = game_state[i, -1, 0] == 1.0
        if auto_mode:
            continue

        # TODO: change desired filters for prediction here
        # this example will skip samples without "left" and "right" keys together
        if binary_prediction[i, 0, 3] == 0 or binary_prediction[i, 0, 2] == 0:
            continue

        plt.figure()
        treesX = [4.55076, -0.7353, -15.10146, -2.6019, -1.33158, 16.58292, 16.87086, 0.6078, -16.65378]
        treesY = [14.66826, 14.75052, 15.76476, 6.7425, 10.02042, -12.5847, -16.01952, -16.23906, -16.23906]
        plt.plot(treesX, treesY, 'g^', markersize=10)

        # water square
        plt.plot(0, -10, 'bs')
        # battery square
        plt.plot(16, 16.29, 'rp')

        # finally, the robot
        for j in range(game_state.shape[1]):
            plt.plot(game_state[i, j, 1], game_state[i, j, 2], marker=(3, 0, -90 + np.degrees(game_state[i, j, 3])), color="black", markersize=15)
            if j > 0:
                plt.plot([game_state[i, j - 1, 1], game_state[i, j, 1]], [game_state[i, j - 1, 2], game_state[i, j, 2]], 'k-')

        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.xlabel('x')
        plt.ylabel('y')

        plt.figtext(0.75, 0.96, "Automatic" if auto_mode else "Manual", ha="left", fontsize=12,
                    bbox={"facecolor": "red" if auto_mode else "grey", "alpha": 0.5, "pad": 3})
        # key_front, key_back, key_left, key_right, key_space
        plt.figtext(0.5, 0.9,
                    f"Front: {binary_prediction[i,0,0]} | Back: {binary_prediction[i, 0, 1]} | Left: {binary_prediction[i, 0, 2]} | Right: {binary_prediction[i, 0, 3]} | Space: {binary_prediction[i, 0, 4]}",
                    ha="center", fontsize=12)
        plt.show()
        input("next...")
