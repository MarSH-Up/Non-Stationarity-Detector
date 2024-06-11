import numpy as np
from k_functions import k_linear, k_piecewise


def calculate_amplitude(t, interval, variances, k_func, k_args=None):
    """
    Piecewise constant function to control the amplitude over time based on specified variances and scaling function.

    Parameters:
        t (numpy.ndarray): Time array.
        interval (int): Number of samples to keep the same amplitude.
        variances (list): List of variances for each interval.
        k_func (function): Scaling function of time.
        k_args (dict, optional): Additional arguments for the k function.

    Returns:
        numpy.ndarray: Array of amplitudes.
    """
    if k_args is None:
        k_args = {}

    total_samples = len(t)

    # Special handling for k_piecewise function
    if k_func == k_piecewise:
        k_values = k_func(t, interval, variances)
    else:
        k_values = k_func(t, **k_args)

    amplitudes = np.zeros_like(t)
    num_intervals = (total_samples // interval) + 1

    for i in range(num_intervals):
        start = i * interval
        end = min(start + interval, total_samples)
        variance = variances[i % len(variances)]
        amplitude = np.sqrt(variance) * k_values[start:end]
        amplitudes[start:end] = amplitude

    return amplitudes


def generate_nonstationary_sine(
    frequency, variances, length, sampling_rate=1000, interval=5000, k_func=k_linear
):
    """
    Generates a non-stationary sine signal.

    Parameters:
        frequency (float): Frequency of the sine wave.
        variances (list): List of variances affecting the amplitude of the sine wave.
        length (int): Length of the signal in seconds.
        sampling_rate (int): Number of samples per second.
        interval (int): Number of samples to keep the same amplitude.
        k_func (function): Function to scale the amplitude over time.

    Returns:
        t (numpy.ndarray): Time array.
        signal (numpy.ndarray): Generated sine signal.
    """
    t = np.linspace(0, length, length * sampling_rate)
    amplitude = calculate_amplitude(
        t, interval, variances, k_func
    )  # Time-varying amplitude
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    return t, signal, amplitude


def generate_classical_sine(frequency, length, sampling_rate):
    """
    Generates a classical sine signal.

    Parameters:
        frequency (float): Frequency of the sine wave.
        length (int): Length of the signal in seconds.
        sampling_rate (int): Number of samples per second.

    Returns:
        t (numpy.ndarray): Time array.
        signal (numpy.ndarray): Generated sine signal.
    """
    t = np.linspace(0, length, length * sampling_rate)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal
