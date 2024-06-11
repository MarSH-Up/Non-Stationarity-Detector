import os
import numpy as np
import matplotlib.pyplot as plt
from Incremental_Average import incremental_average
from scipy.stats import norm
from Signal_miscellaneous import *
from Statistical_Moments import *
from k_functions import k_sin


def Linear_Signal_Experiment(
    frequency,
    variances,
    length,
    sampling_rate,
    interval,
    window_lenght,
    window_overlaping,
):
    experiment_dir = "images/Linear_Signal_Overlaping_Window"
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate the signal using k_piecewise
    t, signal_linear, amplitude = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_linear
    )

    # Calculate moving averages
    moving_average_signal = incremental_average(signal_linear)
    moving_average_amplitude = incremental_average(amplitude)

    # Plot the signal with k_piecewise
    plt.plot(t, signal_linear)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Piecewise Constant Amplitude")
    plt.savefig(os.path.join(experiment_dir, "signal_plot.png"))
    plt.show()

    # Plot the Moving Average with k_piecewise
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_linear, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude, label="Moving Average Amplitude")
    plt.title("Non-stationary Signal and its Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "moving_average_plot.png"))
    plt.show()

    # Fragment the signal into overlapping windows and calculate statistical moments
    window_size = window_lenght
    overlap = window_overlaping
    num_windows = (len(signal_linear) - window_size) // (window_size - overlap) + 1
    moments = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_linear[start:end]
        mean_value = np.mean(window_signal)
        variance_value = moment_variance(window_signal)
        skewness_value = moment_skew(window_signal)
        kurtosis_value = moment_skew(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))
        # Plot the histogram for the current window
        plt.figure(figsize=(10, 6))
        count, bins, ignored = plt.hist(
            window_signal,
            bins=30,
            density=True,
            edgecolor="k",
            alpha=0.7,
            label="Signal Distribution",
        )

        # Fit a normal distribution
        normal_dist = norm.pdf(bins, mean_value, np.sqrt(variance_value))

        # Print the statistical moments and ADF test results for the current window
        """ 
        print(f"Window {i+1}:")
        print(f"  Mean: {mean_value}")
        print(f"  Variance: {variance_value}")
        print(f"  Skewness: {skewness_value}")
        print(f"  Kurtosis: {kurtosis_value}")
        """

        # Plot the normal distribution
        plt.plot(bins, normal_dist, "r-", linewidth=2, label="Normal Distribution")
        plt.title(f"Distribution of Window {i+1}")
        plt.xlabel("Signal Value")
        plt.ylabel("Probability Density")
        plt.grid(True)

        # Annotate the histogram with the statistical moments
        plt.axvline(
            mean_value,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {mean_value:.2f}",
        )
        plt.axvline(
            mean_value + np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
            label=f"Standard Deviation: {np.sqrt(variance_value):.2f}",
        )
        plt.axvline(
            mean_value - np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
        )
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, f"histogram_window_{i+1}.png"))
        plt.show()

    # Print all statistical moments
    """
    for i, (mean_value, variance_value, skewness_value, kurtosis_value) in enumerate(
        moments
    ):
        print(
            f"Window {i+1}: Mean={mean_value}, Variance={variance_value}, Skewness={skewness_value}, Kurtosis={kurtosis_value}"
        )

    moments_observation(moments, num_windows, experiment_dir)
    
    """
    return moments


def Classical_Signal_Experiment(
    frequency, length, sampling_rate, window_length, window_overlaping
):
    experiment_dir = "images/Classical_Signal"
    os.makedirs(experiment_dir, exist_ok=True)
    # Generate the classical sine signal
    t, signal_classical = generate_classical_sine(frequency, length, sampling_rate)

    # Calculate moving averages
    moving_average_signal = incremental_average(signal_classical)

    # Plot the classical sine signal
    plt.plot(t, signal_classical)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Classical Sine Signal")
    plt.savefig(os.path.join(experiment_dir, "signal_plot.png"))

    # Plot the Moving Average for the classical sine signal
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_classical, label="Classical Sine Signal")
    plt.plot(t, moving_average_signal, label="Moving Average Signal")
    plt.title("Classical Sine Signal and its Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "moving_average_plot.png"))
    plt.show()

    # Fragment the signal into overlapping windows and calculate statistical moments
    window_size = window_length
    overlap = window_overlaping
    num_windows = (len(signal_classical) - window_size) // (window_size - overlap) + 1
    moments = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_classical[start:end]
        mean_value = np.mean(window_signal)
        variance_value = moment_variance(window_signal)
        skewness_value = moment_skew(window_signal)
        kurtosis_value = moment_skew(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))
        # Plot the histogram for the current window
        plt.figure(figsize=(10, 6))
        count, bins, ignored = plt.hist(
            window_signal,
            bins=30,
            density=True,
            edgecolor="k",
            alpha=0.7,
            label="Signal Distribution",
        )

        # Fit a normal distribution
        normal_dist = norm.pdf(bins, mean_value, np.sqrt(variance_value))

        # Plot the normal distribution
        plt.plot(bins, normal_dist, "r-", linewidth=2, label="Normal Distribution")
        plt.title(f"Distribution of Window {i+1}")
        plt.xlabel("Signal Value")
        plt.ylabel("Probability Density")
        plt.grid(True)

        # Annotate the histogram with the statistical moments
        plt.axvline(
            mean_value,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {mean_value:.2f}",
        )
        plt.axvline(
            mean_value + np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
            label=f"Standard Deviation: {np.sqrt(variance_value):.2f}",
        )
        plt.axvline(
            mean_value - np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
        )
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, f"histogram_window_{i+1}.png"))
        plt.show()

        # Print the statistical moments and ADF test results for the current window
        """
        print(f"Window {i+1}:")
        print(f"  Mean: {mean_value}")
        print(f"  Variance: {variance_value}")
        print(f"  Skewness: {skewness_value}")
        print(f"  Kurtosis: {kurtosis_value}") 
        """

    # Print all statistical moments
    """
    for i, (mean_value, variance_value, skewness_value, kurtosis_value) in enumerate(
        moments
    ):
        print(
            f"Window {i+1}: Mean={mean_value}, Variance={variance_value}, Skewness={skewness_value}, Kurtosis={kurtosis_value}"
        )
    moments_observation(moments, num_windows, experiment_dir)

    """
    return moments


def Piecewise_Signal_Experiment(
    frequency,
    variances,
    length,
    sampling_rate,
    interval,
    window_lenght,
    window_overlaping,
):
    experiment_dir = "images/Piecewise_Signal_Experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate the signal using k_piecewise
    t, signal_piecewise, amplitude = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_piecewise
    )

    moving_average_signal = incremental_average(signal_piecewise)
    moving_average_amplitude = incremental_average(amplitude)

    # Plot the signal with k_piecewise
    plt.plot(t, signal_piecewise)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Piecewise Constant Amplitude")
    plt.savefig(os.path.join(experiment_dir, "signal_plot.png"))

    # Plot the MovingAverage with k_piecewise
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_piecewise, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude, label="Moving Average Amplitud")
    plt.title("Non-stationary Signal and its Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "moving_average_plot.png"))

    # Calculate statistical moments
    mean_value = np.mean(signal_piecewise)
    variance_value = moment_variance(signal_piecewise)
    skewness_value = moment_skew(signal_piecewise)
    kurtosis_value = moment_skew(signal_piecewise)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(
        signal_piecewise,
        bins=30,
        density=True,
        edgecolor="k",
        alpha=0.7,
        label="Signal Distribution",
    )

    # Fit a normal distribution
    normal_dist = norm.pdf(bins, mean_value, np.sqrt(variance_value))

    # Plot the normal distribution
    plt.plot(bins, normal_dist, "r-", linewidth=2, label="Normal Distribution")
    plt.title("Distribution of the Non-stationary Piecewise Signal")
    plt.xlabel("Signal Value")
    plt.ylabel("Probability Density")
    plt.grid(True)

    # Annotate the histogram with the statistical moments
    plt.axvline(
        mean_value,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {mean_value:.2f}",
    )
    plt.axvline(
        mean_value + np.sqrt(variance_value),
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"Standard Deviation: {np.sqrt(variance_value):.2f}",
    )
    plt.axvline(
        mean_value - np.sqrt(variance_value), color="g", linestyle="dashed", linewidth=1
    )
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "Full Signal Distribution.png"))
    # Print the statistical moments
    """
    print(f"Mean: {mean_value}")
    print(f"Variance: {variance_value}")
    print(f"Skewness: {skewness_value}")
    print(f"Kurtosis: {kurtosis_value}")   
    """

    # Fragment the signal into overlapping windows and calculate statistical moments
    window_size = window_lenght
    overlap = window_overlaping
    num_windows = (len(signal_piecewise) - window_size) // (window_size - overlap) + 1
    moments = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_piecewise[start:end]
        mean_value = np.mean(window_signal)
        variance_value = moment_variance(window_signal)
        skewness_value = moment_skew(window_signal)
        kurtosis_value = moment_skew(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))
        # Plot the histogram for the current window
        plt.figure(figsize=(10, 6))
        count, bins, ignored = plt.hist(
            window_signal,
            bins=30,
            density=True,
            edgecolor="k",
            alpha=0.7,
            label="Signal Distribution",
        )

        # Fit a normal distribution
        normal_dist = norm.pdf(bins, mean_value, np.sqrt(variance_value))

        # Print the statistical moments and ADF test results for the current window
        """
        print(f"Window {i+1}:")
        print(f"  Mean: {mean_value}")
        print(f"  Variance: {variance_value}")
        print(f"  Skewness: {skewness_value}")
        print(f"  Kurtosis: {kurtosis_value}")     
        """

        # Plot the normal distribution
        plt.plot(bins, normal_dist, "r-", linewidth=2, label="Normal Distribution")
        plt.title(f"Distribution of Window {i+1}")
        plt.xlabel("Signal Value")
        plt.ylabel("Probability Density")
        plt.grid(True)

        # Annotate the histogram with the statistical moments
        plt.axvline(
            mean_value,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {mean_value:.2f}",
        )
        plt.axvline(
            mean_value + np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
            label=f"Standard Deviation: {np.sqrt(variance_value):.2f}",
        )
        plt.axvline(
            mean_value - np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
        )
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, f"histogram_window_{i+1}.png"))
        plt.show()
    # Print all statistical moments
    """
    for i, (mean_value, variance_value, skewness_value, kurtosis_value) in enumerate(
        moments
    ):
        print(
            f"Window {i+1}: Mean={mean_value}, Variance={variance_value}, Skewness={skewness_value}, Kurtosis={kurtosis_value}"
        )
    moments_observation(moments, num_windows, experiment_dir)
    """
    return moments


def Sine_Signal_Experiment(
    frequency,
    variances,
    length,
    sampling_rate,
    interval,
    window_lenght,
    window_overlaping,
):
    experiment_dir = "images/Sine_Signal_Experiment"
    os.makedirs(experiment_dir, exist_ok=True)

    t, signal_sin, amplitude = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_sin
    )

    moving_average_signal = incremental_average(signal_sin)
    moving_average_amplitude = incremental_average(amplitude)

    # Plot the signal with k_sin
    plt.plot(t, signal_sin)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Piecewise Constant Amplitude")
    plt.savefig(os.path.join(experiment_dir, "signal_plot.png"))

    # Plot the MovingAverage with k_sin
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_sin, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude, label="Moving Average Amplitud")
    plt.title("Non-stationary Signal and its Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "moving_average_plot.png"))
    plt.show()

    # Calculate statistical moments
    mean_value = np.mean(signal_sin)
    variance_value = moment_variance(signal_sin)
    skewness_value = moment_skew(signal_sin)
    kurtosis_value = moment_skew(signal_sin)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(
        signal_sin,
        bins=30,
        density=True,
        edgecolor="k",
        alpha=0.7,
        label="Signal Distribution",
    )

    # Fit a normal distribution
    normal_dist = norm.pdf(bins, mean_value, np.sqrt(variance_value))

    # Plot the normal distribution
    plt.plot(bins, normal_dist, "r-", linewidth=2, label="Normal Distribution")
    plt.title("Distribution of the Non-stationary Sine Signal")
    plt.xlabel("Signal Value")
    plt.ylabel("Probability Density")
    plt.savefig(os.path.join(experiment_dir, "signal_distribution_plot.png"))
    plt.grid(True)

    # Annotate the histogram with the statistical moments
    plt.axvline(
        mean_value,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {mean_value:.2f}",
    )
    plt.axvline(
        mean_value + np.sqrt(variance_value),
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"Standard Deviation: {np.sqrt(variance_value):.2f}",
    )
    plt.axvline(
        mean_value - np.sqrt(variance_value), color="g", linestyle="dashed", linewidth=1
    )
    plt.legend()
    plt.show()

    # Print the statistical moments
    """
    print(f"Mean: {mean_value}")
    print(f"Variance: {variance_value}")
    print(f"Skewness: {skewness_value}")
    print(f"Kurtosis: {kurtosis_value}")   
    """

    # Fragment the signal into overlapping windows and calculate statistical moments
    window_size = window_lenght
    overlap = window_overlaping
    num_windows = (len(signal_sin) - window_size) // (window_size - overlap) + 1
    moments = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_sin[start:end]
        mean_value = np.mean(window_signal)
        variance_value = np.var(window_signal)
        skewness_value = skew(window_signal)
        kurtosis_value = kurtosis(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))
        # Plot the histogram for the current window
        plt.figure(figsize=(10, 6))
        count, bins, ignored = plt.hist(
            window_signal,
            bins=30,
            density=True,
            edgecolor="k",
            alpha=0.7,
            label="Signal Distribution",
        )

        # Fit a normal distribution
        normal_dist = norm.pdf(bins, mean_value, np.sqrt(variance_value))
        """
        # Print the statistical moments and ADF test results for the current window
        print(f"Window {i+1}:")
        print(f"  Mean: {mean_value}")
        print(f"  Variance: {variance_value}")
        print(f"  Skewness: {skewness_value}")
        print(f"  Kurtosis: {kurtosis_value}")
        """

        # Plot the normal distribution
        plt.plot(bins, normal_dist, "r-", linewidth=2, label="Normal Distribution")
        plt.title(f"Distribution of Window {i+1}")
        plt.xlabel("Signal Value")
        plt.ylabel("Probability Density")
        plt.grid(True)

        # Annotate the histogram with the statistical moments
        plt.axvline(
            mean_value,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {mean_value:.2f}",
        )
        plt.axvline(
            mean_value + np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
            label=f"Standard Deviation: {np.sqrt(variance_value):.2f}",
        )
        plt.axvline(
            mean_value - np.sqrt(variance_value),
            color="g",
            linestyle="dashed",
            linewidth=1,
        )
        plt.legend()
        plt.savefig(os.path.join(experiment_dir, f"histogram_window_{i+1}.png"))
        plt.show()

    # Print all statistical moments
    """
    for i, (mean_value, variance_value, skewness_value, kurtosis_value) in enumerate(
        moments
    ):
        print(
            f"Window {i+1}: Mean={mean_value}, Variance={variance_value}, Skewness={skewness_value}, Kurtosis={kurtosis_value}"
        )
    moments_observation(moments, num_windows, experiment_dir)
    """
    return moments
