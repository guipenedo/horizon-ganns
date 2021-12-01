import numpy as np
import pandas as pd
import scipy.stats as stats

df = pd.read_csv("full_results.csv")

keys = ["front", "back", "left", "right", "space"]

suc = df.groupby(["generator", "robot_mode"]).sum()
counts = df.groupby(["generator", "robot_mode"]).count()

freqs = df.groupby(["generator", "robot_mode"]).mean()

print(freqs)


def compute_chi_square_gof(expected, observed):
    """
    Runs a chi-square goodness-of-fit test and returns the p-value.
    Inputs:
    - expected: numpy array of expected values.
    - observed: numpy array of observed values.
    Returns: p-value
    """
    expected_scaled = expected / float(sum(expected)) * sum(observed)
    result = stats.chisquare(f_obs=observed, f_exp=expected_scaled)
    return result[1]


def make_decision(p_value):
    """
    Makes a goodness-of-fit decision on an input p-value.
    Input: p_value: the p-value from a goodness-of-fit test.
    Returns: "different" if the p-value is below 0.05, "same" otherwise
    """
    return "different" if p_value < 0.05 else "same"


def get_keys_distribution(key, generated, robot_mode):
    """
    Returns a size 5 np array with either the frequencies for all 5 keys or the number
    of successes for the specified generator/sample mode and robot_mode
    """
    return np.array([freqs[key][generated][robot_mode], 1 - freqs[key][generated][
        robot_mode]])


if __name__ == "__main__":
    for key in keys:
        print("Key: " + key)
        dataset_manual = get_keys_distribution(key, 0, 0)
        dataset_automatic = get_keys_distribution(key, 0, 1)
        generated_manual = get_keys_distribution(key, 1, 0)
        generated_automatic = get_keys_distribution(key, 1, 1)

        p_value = compute_chi_square_gof(dataset_manual, generated_manual)
        print(f"Comparing frequencies of manual dataset {dataset_manual} vs manual generated {generated_manual}: {p_value=}; {make_decision(p_value)}")

        p_value = compute_chi_square_gof(dataset_automatic, generated_automatic)
        print(f"Comparing frequencies of automatic dataset {dataset_automatic} vs automatic generated {generated_automatic}: {p_value=}; {make_decision(p_value)}")
