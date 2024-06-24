from Experiments import *


def main():
    # Parameters
    frequency = 1  # Frequency of 1 Hz
    variances = [0.5, 1, 1.5]  # List of variances to repeat
    length = 5 * len(variances)  # Length of the signal in seconds
    sampling_rate = 1000  # Number of samples per second
    interval = 5000  # Interval of 5000 samples

    # Half of interval
    window_length = 4000
    window_overlaping = 2000

    """
    moments_sine_signal = Sine_Signal_Experiment(
        frequency,
        variances,
        length,
        sampling_rate,
        interval,
        window_length,
        window_overlaping,
    )
    results = evaluate_stationarity(moments_sine_signal)
    print("Stationarity Results Sine Signal:")
    for key, value in results.items():
        print(f"{key}: {value}")

    moments_piecewise_signal = Piecewise_Signal_Experiment(
        frequency,
        variances,
        length,
        sampling_rate,
        interval,
        window_length,
        window_overlaping,
    )
    results = evaluate_stationarity(moments_piecewise_signal)
    print("Stationarity Results Piecewisw Signal:")
    for key, value in results.items():
        print(f"{key}: {value}")

    moments_linear_signal = Linear_Signal_Experiment(
        frequency,
        variances,
        length,
        sampling_rate,
        interval,
        window_length,
        window_overlaping,
    )
    results = evaluate_stationarity(moments_linear_signal)
    print("Stationarity Results Linear Signal:")
    for key, value in results.items():
        print(f"{key}: {value}")

    moments_clasical_signal = Classical_Signal_Experiment(
        frequency, length, sampling_rate, window_length, window_overlaping
    )
    results = evaluate_stationarity(moments_clasical_signal)
    print("Stationarity Results Classical Signal:")
    for key, value in results.items():
        print(f"{key}: {value}")
    """
    moments_test_signal, independent_status, status_array, signal = (
        Piecewise_Signal_Experiment_Test_Complete(
            frequency,
            variances,
            length,
            sampling_rate,
            interval,
            window_length,
            window_overlaping,
        )
    )
    stationarity_result = analyze_stationarity(signal)
    print("Stationarity test: ", stationarity_result)

    binary_values_Independent = [
        0 if status == "Stationary" else 1 for status in independent_status
    ]
    binary_values_statys = [
        0 if status == "Stationary" else 1 for status in status_array
    ]

    # Plotting the binary values
    plt.figure(figsize=(10, 2))
    plt.plot(
        binary_values_Independent,
        drawstyle="steps-post",
        marker="o",
        color="b",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
    )
    plt.yticks([0, 1], ["Stationary", "Non-Stationary"])
    plt.title("Binary Representation of Stationarity Status Independent")
    plt.xlabel("Window Index")
    plt.ylabel("Status")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.plot(
        binary_values_statys,
        drawstyle="steps-post",
        marker="o",
        color="b",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
    )
    plt.yticks([0, 1], ["Stationary", "Non-Stationary"])
    plt.title("Binary Representation of Stationarity Status Overlapping")
    plt.xlabel("Window Index")
    plt.ylabel("Status")
    plt.grid(True)
    plt.show()
    results = evaluate_stationarity(moments_test_signal)
    print("Stationarity Results Classical Signal:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print("Non-stationary Model generation")
    main()
