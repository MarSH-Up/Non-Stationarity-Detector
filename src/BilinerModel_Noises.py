import numpy as np


def awgn(signal, snr_db, signal_power=None):
    """
    Add white Gaussian noise to a signal to achieve a given SNR.

    Parameters:
    - signal: The input signal
    - snr_db: Desired SNR in dB
    - signal_power: If 'measured', the power is calculated from the signal. If None, it's assumed to be 1.

    Returns:
    - Noisy signal
    """
    if signal_power == "measured":
        power = np.mean(signal**2)
    else:
        power = 1.0

    noise_power = power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    return signal + noise
