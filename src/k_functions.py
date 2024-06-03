import numpy as np
import matplotlib.pyplot as plt


def k_linear(t):
    """
    Linear function of time for scaling.

    Parameters:
        t (numpy.ndarray): Time array.

    Returns:
        numpy.ndarray: Array of scaling factors.
    """
    return t


def k_sin(t):
    """
    Sine function of time for scaling.

    Parameters:
        t (numpy.ndarray): Time array.

    Returns:
        numpy.ndarray: Array of scaling factors.
    """
    return np.sin(2 * np.pi * t)


def k_piecewise(t, interval, variances):
    """
    Piecewise constant function to control the amplitude over time based on specified variances.

    Parameters:
        t (numpy.ndarray): Time array.
        interval (int): Number of samples to keep the same amplitude.
        variances (list): List of variances for each interval.

    Returns:
        numpy.ndarray: Array of scaling factors.
    """
    total_samples = len(t)
    k_values = np.zeros_like(t)
    num_intervals = (total_samples // interval) + 1

    for i in range(num_intervals):
        start = i * interval
        end = min(start + interval, total_samples)
        variance = variances[i % len(variances)]
        amplitude = np.sqrt(2 * variance)
        k_values[start:end] = amplitude

    return k_values
