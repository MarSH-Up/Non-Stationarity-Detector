# Analysis of Graphs in Terms of Amplitude, Variance, and Non-Stationarity

## 1. Graph with Linear Scaling Function (`k_linear`):
**Amplitude**: The amplitude of the signal increases linearly over time. Initially, the amplitude is small and grows steadily as time progresses.

**Variance**: The variance of the signal also increases linearly. This indicates a continuously and predictably increasing variance.

**Non-Stationarity**: Non-stationarity is present because the amplitude and, therefore, the variance are continuously changing.

## 2. Graph with Sine Scaling Function (`k_sin`):

**Amplitude**: The amplitude of the signal varies according to a sine function. This means that the amplitude oscillates periodically, reaching peaks and valleys at regular intervals.

**Variance**: The variance of the signal follows an oscillating pattern similar to the amplitude. This creates a pattern of cyclical non-stationarity.

**Non-Stationarity**: The signal is non-stationary because the variance changes cyclically.

## 3. Graph with Piecewise Constant Scaling Function (`k_piecewise`):

**Amplitude**: The amplitude of the signal remains constant during certain intervals and then abruptly changes to a new value. Each segment of the signal has a fixed amplitude, but this changes abruptly at the beginning of each new interval.

**Variance**: The variance of the signal is constant within each interval but changes abruptly at the transition points between intervals. This results in a non-stationary signal with segments of constant variance separated by abrupt changes.

**Non-Stationarity**: This signal shows segmented non-stationarity. The variance is constant within each segment but changes abruptly between segments.

## Conclusion

Each graph represents a different scenario of non-stationarity:

- **`k_linear`**: Presents a variance that continuously increases due to the linear growth of the amplitude, creating an environment of non-stationarity with progressively growing variance.
- **`k_sin`**: The variance oscillates periodically, reflecting the cyclical changes in amplitude. This type of signal is useful for modeling processes with seasonality or repetitive patterns.
- **`k_piecewise`**: The variance is constant within each segment and changes abruptly at the transition points. This type of signal is suitable for modeling processes with sudden changes in behavior, such as discrete events that abruptly alter the variance.
