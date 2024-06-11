from Experiments import *


def main():
    # Parameters
    frequency = 1  # Frequency of 1 Hz
    variances = [0.5, 1, 1.5, 1, 0.5]  # List of variances to repeat
    length = 5 * len(variances)  # Length of the signal in seconds
    sampling_rate = 1000  # Number of samples per second
    interval = 5000  # Interval of 5000 samples

    # Half of interval
    window_length = 2500
    window_overlaping = 1250

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


if __name__ == "__main__":
    print("Non-stationary Model generation")
    main()

"""


"""
