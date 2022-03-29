import pandas as pd
import statsmodels.stats.proportion
import scipy.stats

df = pd.read_csv("full_results.csv")

keys = ["front", "back", "left", "right", "space"]

suc = df.groupby(["generator", "robot_mode"]).sum()
counts = df.groupby(["generator", "robot_mode"]).count()

freqs = df.groupby(["generator", "robot_mode"]).mean()

print(freqs)

print(suc)
print(counts)

print("ZTEST:")
for key in keys:
    for i in range(2):
        x = suc[key]
        y = counts[key]
        print(f"{key=}, robot_mode={i}")
        print(statsmodels.stats.proportion.proportions_ztest([x[0][i], x[1][i]], [y[0][i], y[1][i]], alternative="two-sided"))

print("ZTEST for mode:")
for key in keys:
    for i in range(2):
        x = suc[key]
        y = counts[key]
        print(f"{key=}, generator={i}")
        print(statsmodels.stats.proportion.proportions_ztest([x[i][0], x[i][1]], [y[i][0], y[i][1]],
                                                             alternative="two-sided"))

print("ChiSquare:")
for key in keys:
    for i in range(2):
        x = suc[key]
        y = counts[key]
        print(f"{key=}, robot_mode={i}")
        # df -> [key][generated][robot_mode]
        print(statsmodels.stats.proportion.proportions_chisquare([x[0][i], x[1][i]], y[0][i]))

print()
print()
print("ChiSquare SciPy:")
for i in range(2):
    f_obs = [[freqs[key][1][i], freqs[key][0][i]] for key in keys]
    print(f"robot_mode={i}")
    print(scipy.stats.chisquare(f_obs, axis=1))
