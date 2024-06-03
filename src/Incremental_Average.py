def incremental_average(signal):
    Q_k = 0  # Initial average
    Q_k_values = []
    for k, r_k_plus_1 in enumerate(signal, start=1):
        Q_k = Q_k + (1 / k) * (r_k_plus_1 - Q_k)
        Q_k_values.append(Q_k)
        # print(f"After {k} rewards, the incremental average is: {Q_k}")

    return Q_k_values


def exponential_moving_average(signal, alpha=0.1):
    ema_values = []
    ema = 0
    for value in signal:
        ema = alpha * value + (1 - alpha) * ema
        ema_values.append(ema)
    return ema_values
