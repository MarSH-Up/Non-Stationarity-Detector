import matplotlib.pyplot as plt
from Incremental_Average import *
from Signal_miscellaneous import generate_nonstationary_sine
from k_functions import *


def main():
    # Parameters
    frequency = 1  # Frequency of 1 Hz
    variances = [0.5, 1, 1.5]  # List of variances to repeat
    length = 5 * len(variances) * 2  # Length of the signal in seconds
    sampling_rate = 100  # Number of samples per second
    interval = 500  # Interval of 5000 samples

    # Generate the signal using k_linear:  Tendencias de varianza creciente continua.
    t, signal_linear, amplitude_linear = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_linear
    )
    moving_average_signal_linear = exponential_moving_average(signal_linear)
    moving_average_amplitude_linar = exponential_moving_average(amplitude_linear)

    # Plot the signal with k_linear
    plt.figure()
    plt.plot(t, signal_linear)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Linear K Function")

    # Plot the MovingAverage with k_linear
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_linear, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal_linear, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude_linar, label="Moving Average Amplitud")
    plt.title("K_linear Non-stationary Signal and Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Generate the signal using k_sin: patrones cíclicos o estacionales
    t, signal_sin, amplitude_sin = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_sin
    )
    moving_average_signal_sin = exponential_moving_average(signal_sin)
    moving_average_amplitude_sin = exponential_moving_average(amplitude_sin)

    # Plot the signal with k_sin
    plt.figure()
    plt.plot(t, signal_sin)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Sine K Function")

    # Plot the MovingAverage with k_sin
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_sin, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal_sin, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude_sin, label="Moving Average Amplitud")
    plt.title("K_sin Non-stationary Signal and Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Generate the signal using k_piecewise: Cambios abruptos en la varianza
    t, signal_piecewise, amplitude_piecewise = generate_nonstationary_sine(
        frequency, variances, length, sampling_rate, interval, k_func=k_piecewise
    )
    moving_average_signal_piecewise = exponential_moving_average(signal_piecewise)
    moving_average_amplitude_piecewise = exponential_moving_average(amplitude_piecewise)
    # Plot the signal with k_piecewise
    plt.figure()
    plt.plot(t, signal_piecewise)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Non-stationary Sine Signal with Piecewise slices")

    # Plot the MovingAverage with k_piecewise
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_piecewise, label="Non-stationary Signal")
    plt.plot(t, moving_average_signal_piecewise, label="Moving Average Signal")
    plt.plot(t, moving_average_amplitude_piecewise, label="Moving Average Amplitud")
    plt.title("K_piecewise Non-stationary Signal and Moving Average")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Non-stationary Model generation")
    main()
