import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def moment_variance(signal):
    return np.var(signal)


def moment_skew(signal):
    return skew(signal)


def moment_skew(signal):
    return kurtosis(signal)


def evaluate_stationarity(moments, std_multiplier=1.5):
    """
    Evaluate the stationarity of a signal based on statistical moments.

    Parameters:
        moments (list of tuples): List containing tuples of statistical moments (mean, variance, skewness, kurtosis) for each window.
        std_multiplier (float): Multiplier for standard deviation to determine the threshold.

    Returns:
        dict: Dictionary indicating whether the signal is stationary based on each moment and overall.
    """
    moments_array = np.array(moments)
    mean_values = moments_array[:, 0]
    variance_values = moments_array[:, 1]
    skewness_values = moments_array[:, 2]
    kurtosis_values = moments_array[:, 3]

    # Calculate mean and standard deviation of the moments
    mean_mean = np.mean(mean_values)
    std_mean = np.std(mean_values)

    mean_variance = np.mean(variance_values)
    std_variance = np.std(variance_values)

    mean_skewness = np.mean(skewness_values)
    std_skewness = np.std(skewness_values)

    mean_kurtosis = np.mean(kurtosis_values)
    std_kurtosis = np.std(kurtosis_values)

    # Determine thresholds based on standard deviation
    mean_threshold = std_multiplier * std_mean
    variance_threshold = std_multiplier * std_variance
    skewness_threshold = std_multiplier * std_skewness
    kurtosis_threshold = std_multiplier * std_kurtosis

    # Initialize variables to track consistency
    is_mean_stationary = True
    is_variance_stationary = True
    is_skewness_stationary = True
    is_kurtosis_stationary = True

    # Check consistency across windows
    for moment in moments:
        if abs(moment[0] - mean_mean) > mean_threshold:
            is_mean_stationary = False
        if abs(moment[1] - mean_variance) > variance_threshold:
            is_variance_stationary = False
        if abs(moment[2] - mean_skewness) > skewness_threshold:
            is_skewness_stationary = False
        if abs(moment[3] - mean_kurtosis) > kurtosis_threshold:
            is_kurtosis_stationary = False

    # Determine overall stationarity
    is_stationary = (
        is_mean_stationary
        and is_variance_stationary
        and is_skewness_stationary
        and is_kurtosis_stationary
    )

    return {
        "Mean Stationary": is_mean_stationary,
        "Variance Stationary": is_variance_stationary,
        "Skewness Stationary": is_skewness_stationary,
        "Kurtosis Stationary": is_kurtosis_stationary,
        "Overall Stationary": is_stationary,
    }


def moments_observation(moments, num_windows, experiment_name="images/temp"):
    experiment_dir = experiment_name
    os.makedirs(experiment_dir, exist_ok=True)
    means = [moment[0] for moment in moments]
    variances = [moment[1] for moment in moments]
    skewnesses = [moment[2] for moment in moments]
    kurtoses = [moment[3] for moment in moments]

    # Window data
    # Plotting the statistical moments over time
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    axs[0].plot(range(1, num_windows + 1), means, marker="o")
    axs[0].set_title("Mean Over Time")
    axs[0].set_xlabel("Window Index")
    axs[0].set_ylabel("Mean")
    axs[0].grid(True)

    axs[1].plot(range(1, num_windows + 1), variances, marker="o", color="orange")
    axs[1].set_title("Variance Over Time")
    axs[1].set_xlabel("Window Index")
    axs[1].set_ylabel("Variance")
    axs[1].grid(True)

    axs[2].plot(range(1, num_windows + 1), skewnesses, marker="o", color="green")
    axs[2].set_title("Skewness Over Time")
    axs[2].set_xlabel("Window Index")
    axs[2].set_ylabel("Skewness")
    axs[2].grid(True)

    axs[3].plot(range(1, num_windows + 1), kurtoses, marker="o", color="red")
    axs[3].set_title("Kurtosis Over Time")
    axs[3].set_xlabel("Window Index")
    axs[3].set_ylabel("Kurtosis")
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "statistical_moments_observations.png"))
    plt.show()
