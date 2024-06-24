import os
import numpy as np
import matplotlib.pyplot as plt
from Incremental_Average import incremental_average
from scipy.stats import norm
from Signal_miscellaneous import *
from Stationarity_evaluation import *
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

    
    
    """
    moments_observation(moments, num_windows, experiment_dir)
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

    """
    moments_observation(moments, num_windows, experiment_dir)
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

    """
    moments_observation(moments, num_windows, experiment_dir)
    return moments


def Piecewise_Signal_Experiment_Tests(
    frequency,
    variances,
    length,
    sampling_rate,
    interval,
    window_length,
    window_overlapping,
):
    experiment_dir = "images/temp_Signal_Experiment_Window"
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate the signal using k_piecewise
    t, signal_piecewise, amplitude = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_piecewise
    )
    # Calculate moving averages
    moving_average_signal = incremental_average(signal_piecewise)
    moving_average_amplitude = incremental_average(amplitude)

    # Plot the signal with k_piecewise
    plt.plot(t, signal_piecewise)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Piecewise Constant Amplitude")
    plt.savefig(os.path.join(experiment_dir, "signal_plot.png"))
    plt.show()

    # Plot the Moving Average with k_piecewise
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_piecewise, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude, label="Moving Average Amplitude")
    plt.title("Non-stationary Signal and its Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "moving_average_plot.png"))
    plt.show()

    # Fragment the signal into overlapping windows and calculate statistical moments
    window_size = window_length
    overlap = window_overlapping
    num_windows = (len(signal_piecewise) - window_size) // (window_size - overlap) + 1
    moments = []
    stationarity_results = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_piecewise[start:end]
        mean_value = np.mean(window_signal)
        variance_value = moment_variance(window_signal)
        skewness_value = skew(window_signal)
        kurtosis_value = kurtosis(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))

        # Analyze stationarity
        stationarity_result = analyze_stationarity(window_signal)
        stationarity_results.append(stationarity_result)

        # Plot the histogram for the current window
        """
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
        """

    # Print all statistical moments
    for i, (mean_value, variance_value, skewness_value, kurtosis_value) in enumerate(
        moments
    ):
        print(
            f"Window {i+1}: Mean={mean_value}, Variance={variance_value}, Skewness={skewness_value}, Kurtosis={kurtosis_value}"
        )
    stationarity_status = []
    # Print stationarity test results and compare consecutive windows
    for i in range(1, num_windows):
        current_result = stationarity_results[i]
        prev_result = stationarity_results[i - 1]
        print(
            f"Window {i} to {i + 1}: Current (ADF p-value={current_result['adf_p']:.4f}, KPSS p-value={current_result['kpss_p']:.4f}) vs "
            f"Previous (ADF p-value={prev_result['adf_p']:.4f}, KPSS p-value={prev_result['kpss_p']:.4f})"
        )

        # Check the combinations of ADF and KPSS tests results
        # Decision logic for stationarity status
        if current_result["adf_p"] > 0.05 and current_result["kpss_p"] < 0.05:
            print(
                "Current Window: Non-stationary without a trend (ADF Positive, KPSS Negative)"
            )
            status = "Non-Stationary"
        elif current_result["adf_p"] < 0.05 and current_result["kpss_p"] > 0.05:
            print("Current Window: Stationary (ADF Negative, KPSS Positive)")
            status = "Stationary"
        elif current_result["adf_p"] > 0.05 and current_result["kpss_p"] < 0.05:
            print(
                "Current Window: Non-stationary, possibly around a trend (Both Agree on Non-stationarity)"
            )
            status = "Non-Stationary"
        elif current_result["adf_p"] < 0.05 and current_result["kpss_p"] > 0.05:
            print(
                "Current Window: Stationary around a mean without a trend (Both Agree on Stationarity)"
            )
            status = "Stationary"
        else:
            print("Current Window: Stationary results are inconclusive or conflicting.")
            status = "Non-Stationary"

        stationarity_status.append(status)  # Append the determined status
    print(stationarity_status)
    moments_observation(moments, num_windows, experiment_dir)
    return moments, stationarity_status, moving_average_signal


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

    """
    moments_observation(moments, num_windows, experiment_dir)
    return moments


def Linear_Signal_Experiment_Tests(
    frequency,
    variances,
    length,
    sampling_rate,
    interval,
    window_length,
    window_overlapping,
):
    experiment_dir = "images/Linear_Signal_Experiment_Tests_Overlapping_Window"
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
    window_size = window_length
    overlap = window_overlapping
    num_windows = (len(signal_linear) - window_size) // (window_size - overlap) + 1
    moments = []
    stationarity_results = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_linear[start:end]
        mean_value = np.mean(window_signal)
        variance_value = moment_variance(window_signal)
        skewness_value = skew(window_signal)
        kurtosis_value = kurtosis(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))

        # Analyze stationarity
        stationarity_result = analyze_stationarity(window_signal)
        stationarity_results.append(stationarity_result)

        # Plot the histogram for the current window
        """
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
        """

    # Print all statistical moments
    for i, (mean_value, variance_value, skewness_value, kurtosis_value) in enumerate(
        moments
    ):
        print(
            f"Window {i+1}: Mean={mean_value}, Variance={variance_value}, Skewness={skewness_value}, Kurtosis={kurtosis_value}"
        )

    # Print stationarity test results and compare consecutive windows
    for i in range(1, num_windows):
        current_result = stationarity_results[i]
        prev_result = stationarity_results[i - 1]
        transition_detected = False
        print(
            f"Window {i} to {i + 1}: Current (ADF p-value={current_result['adf_p']:.4f}, KPSS p-value={current_result['kpss_p']:.4f}) vs "
            f"Previous (ADF p-value={prev_result['adf_p']:.4f}, KPSS p-value={prev_result['kpss_p']:.4f})"
        )

        # Check the combinations of ADF and KPSS tests results
        if current_result["adf_p"] > 0.05 and current_result["kpss_p"] < 0.05:
            print(
                "Current Window: Non-stationary without a trend (ADF Positive, KPSS Negative)"
            )
            transition_detected = True
        elif current_result["adf_p"] < 0.05 and current_result["kpss_p"] > 0.05:
            print("Current Window: Stationary (ADF Negative, KPSS Positive)")
            transition_detected = False
        elif current_result["adf_p"] > 0.05 and current_result["kpss_p"] < 0.05:
            print(
                "Current Window: Non-stationary, possibly around a trend (Both Agree on Non-stationarity)"
            )
            transition_detected = True
        elif current_result["adf_p"] < 0.05 and current_result["kpss_p"] > 0.05:
            print(
                "Current Window: Stationary around a mean without a trend (Both Agree on Stationarity)"
            )
            transition_detected = False
        else:
            print("Current Window: Stationary results are inconclusive or conflicting.")

        # Print transition information based on detection
        print(
            f"Window {i} to {i + 1}: {'Non-Stationary' if transition_detected else 'Stationary'}"
        )

    moments_observation(moments, num_windows, experiment_dir)
    return moments, stationarity_results, signal_linear


def Piecewise_Signal_Experiment_Test_Complete(
    frequency,
    variances,
    length,
    sampling_rate,
    interval,
    window_length,
    window_overlapping,
):
    experiment_dir = "images/temp_Signal_Experiment_Window"
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate the signal using k_piecewise
    t, signal_piecewise, amplitude = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_piecewise
    )
    # Calculate moving averages
    moving_average_signal = incremental_average(signal_piecewise)
    moving_average_amplitude = incremental_average(amplitude)

    # Plot the signal with k_piecewise
    plt.plot(t, signal_piecewise)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Piecewise Constant Amplitude")
    plt.savefig(os.path.join(experiment_dir, "signal_plot.png"))
    plt.show()

    # Plot the Moving Average with k_piecewise
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_piecewise, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude, label="Moving Average Amplitude")
    plt.title("Non-stationary Signal and its Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "moving_average_plot.png"))
    plt.show()

    # Fragment the signal into overlapping windows and calculate statistical moments
    window_size = window_length
    overlap = window_overlapping
    num_windows = (len(signal_piecewise) - window_size) // (window_size - overlap) + 1
    moments = []
    stationarity_results = []
    stationarity_status = []
    independent_stationarity_status = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_piecewise[start:end]
        mean_value = np.mean(window_signal)
        variance_value = moment_variance(window_signal)
        skewness_value = skew(window_signal)
        kurtosis_value = kurtosis(window_signal)

        moments.append((mean_value, variance_value, skewness_value, kurtosis_value))

        # Analyze stationarity
        stationarity_result = analyze_stationarity(window_signal)
        stationarity_results.append(stationarity_result)

        # Determine stationarity status for the current window
        if stationarity_result["kpss_p"] < 0.05:
            independent_status = "Non-Stationary"
        elif stationarity_result["kpss_p"] > 0.05:
            independent_status = "Stationary"
        else:
            independent_status = "Non-Stationary"

        independent_stationarity_status.append(independent_status)

    # Print stationarity test results and compare consecutive windows
    for i in range(1, num_windows):
        current_result = stationarity_results[i]
        prev_result = stationarity_results[i - 1]

        # Determine current window stationarity status
        if current_result["adf_p"] > 0.05 and current_result["kpss_p"] < 0.05:
            current_status = "Non-Stationary"
        elif current_result["adf_p"] < 0.05 and current_result["kpss_p"] > 0.05:
            current_status = "Stationary"
        else:
            current_status = "Non-Stationary"

        # Determine previous window stationarity status
        if prev_result["adf_p"] > 0.05 and prev_result["kpss_p"] < 0.05:
            prev_status = "Non-Stationary"
        elif prev_result["adf_p"] < 0.05 and prev_result["kpss_p"] > 0.05:
            prev_status = "Stationary"
        else:
            prev_status = "Non-Stationary"

        if current_status != prev_status:
            print(current_status, prev_status)
            current_status = "Non-Stationary"
        else:
            current_status = "Stationary"

        # Append the current status to the stationarity status list
        stationarity_status.append(current_status)

    moments_observation(moments, num_windows, experiment_dir)
    return (
        moments,
        independent_stationarity_status,
        stationarity_status,
        moving_average_signal,
    )
