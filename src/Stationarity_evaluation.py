import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox


def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]


def kpss_test(series):
    result = kpss(series, regression="c")
    return result[0], result[1]


def jarque_bera_test(series):
    jb_stat, jb_p_value = jarque_bera(series)
    return jb_stat, jb_p_value


def ljung_box_test(series, lags=10):
    result = acorr_ljungbox(series, lags=lags)
    return result["lb_stat"], result["lb_pvalue"]


def analyze_stationarity(window_signal):
    adf_stat, adf_p = adf_test(window_signal)
    kpss_stat, kpss_p = kpss_test(window_signal)

    return {"adf_p": adf_p, "kpss_p": kpss_p}


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
