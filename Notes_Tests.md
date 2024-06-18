# Stationarity Tests in Time Series Analysis

Understanding whether a time series is stationary is crucial for effective modeling and forecasting. Two popular tests used to analyze stationarity are the Augmented Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. These tests are complementary, as they are based on different null hypotheses concerning stationarity.

## 1. Augmented Dickey-Fuller (ADF) Test

### Purpose
The ADF test is employed to test for the presence of a unit root in a time series, which is a critical indicator of non-stationarity.

### Definition of a Unit Root
A unit root in a time series model implies that the series is defined by a stochastic process that can be expressed as:

\[
y_t = \rho y_{t-1} + u_t
\]

where:
- \( y_t \) is the value of the series at time t,
- \( \rho \) is the coefficient of \( y_{t-1} \) (lagged value),
- \( u_t \) is the error term.

If \( \rho = 1 \), the series has a unit root, indicating that it follows a random walk and is non-stationary. This condition implies that shocks to the series have a permanent effect.

### Null Hypothesis (H0)
The series has a unit root (non-stationary).

### Alternative Hypothesis (H1)
The series does not have a unit root (stationary).

### Methodology
The ADF test extends the Dickey-Fuller test by including lagged differences of the series to handle autoregressive processes. The regression model used is:

\[
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t
\]

- \( \Delta \): first difference operator.
- \( \alpha \): constant term.
- \( \beta t \): coefficient on a time trend.
- \( \gamma \): coefficient on lagged level of the series.
- \( \delta_i \): coefficients of the lagged differences.
- \( \epsilon_t \): error term.

### Interpretation
- If \( \gamma \) is significantly different from zero (specifically, less than zero), it suggests that the series does not have a unit root, supporting stationarity.

## 2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test

### Purpose
The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test primarily assesses the stationarity of a time series, but with a specific focus. Unlike tests such as the Augmented Dickey-Fuller (ADF) test that look for a unit root, KPSS is designed to test the null hypothesis that a time series is stationary around a deterministic trend. Here’s how this works:

### Null Hypothesis (H0)
The time series is stationary around a trend. This means that although the series may exhibit a long-term trend (i.e., it might increase or decrease over time), the fluctuations around this trend are stable and consistent over time. These fluctuations are essentially stationary stochastic processes superimposed on a deterministic trend line.

### Alternative Hypothesis (H1)
 The time series has a unit root, meaning it is non-stationary, and any trends in the series are not around a stable mean but are themselves part of the non-stationarity. This could mean that the series has a stochastic trend or other forms of non-stationary behavior such as a random walk.

### Methodology
The KPSS test categorizes a series as stationary on the basis of the stationarity around a mean or linear trend. The test statistic is calculated from the series decomposed as:

\[
y_t = \mu + \beta t + \rho_t + \epsilon_t
\]

- \( \mu \): intercept (mean of the series).
- \( \beta t \): slope of the trend.
- \( \rho_t \): stationary stochastic process.
- \( \epsilon_t \): noise.

### Interpretation
- A significant test statistic suggests the presence of a unit root, indicating non-stationarity.

### Cautions

A major disadvantage for the KPSS test is that it has a high rate of Type I errors (it tends to reject the null hypothesis too often). If attempts are made to control these errors (by having larger p-values), then that negatively impacts the test’s power.
One way to deal with the potential for high Type I errors is to combine the KPSS with an ADF test. If the result from both tests suggests that the time series in stationary, then it probably is.

## Complementing Each Other

- **ADF Positive, KPSS Negative**: Indicates non-stationarity as ADF suggests no unit root but KPSS indicates a trend.
- **ADF Negative, KPSS Positive**: Both tests agree that the series is stationary.
- **Mixed Results**: When results from ADF and KPSS contradict each other, deeper analysis is required, considering the specific characteristics of the data and the tests' sensitivity to different types of non-stationarity.

Using ADF and KPSS together provides a more comprehensive understanding of the time series' properties, especially when assessing for stationarity, which is pivotal for accurate modeling and forecasting in various applications such as economics, finance, and meteorology.
