import pandas as pd
import numpy as np
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
    result = stats.chisquare(f_obs=observed, f_exp=expected)
    return result[1]


def make_decision(p_value):
    """
    Makes a goodness-of-fit decision on an input p-value.
    Input: p_value: the p-value from a goodness-of-fit test.
    Returns: "different" if the p-value is below 0.05, "same" otherwise
    """
    return "different" if p_value < 0.05 else "same"


def get_keys_distribution(generated, robot_mode, freq=True):
    """
    Returns a size 5 np array with either the frequencies for all 5 keys or the number
    of successes for the specified generator/sample mode and robot_mode
    """
    x = freqs if freq else suc
    return np.array([x[key][generated][robot_mode] for key in keys])


if __name__ == "__main__":
    dataset_manual = get_keys_distribution(0, 0)
    dataset_automatic = get_keys_distribution(0, 1)
    generated_manual = get_keys_distribution(1, 0)
    generated_automatic = get_keys_distribution(1, 1)

    p_value = compute_chi_square_gof(dataset_manual, generated_manual)
    print(f"Comparing frequencies of manual dataset vs manual generated: {p_value=}; {make_decision(p_value)}")

    p_value = compute_chi_square_gof(dataset_automatic, generated_automatic)
    print(f"Comparing frequencies of automatic dataset vs automatic generated: {p_value=}; {make_decision(p_value)}")

    dataset_manual = get_keys_distribution(0, 0, False)
    dataset_automatic = get_keys_distribution(0, 1, False)
    generated_manual = get_keys_distribution(1, 0, False)
    generated_automatic = get_keys_distribution(1, 1, False)

    p_value = compute_chi_square_gof(dataset_manual, generated_manual)
    print(f"Comparing counts of manual dataset vs manual generated: {p_value=}; {make_decision(p_value)}")

    p_value = compute_chi_square_gof(dataset_automatic, generated_automatic)
    print(f"Comparing counts of automatic dataset vs automatic generated: {p_value=}; {make_decision(p_value)}")
