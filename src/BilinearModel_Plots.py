import matplotlib.pyplot as plt
import numpy as np


def plot_neurodynamics(Z, timestamps, fig, ax):
    """
    Plots the neurodynamic values over time for multiple regions.

    Parameters:
    - Z: 2D numpy array of shape (nTimestamps, nRegions) containing neurodynamic values.
    - timestamps: 1D numpy array of shape (nTimestamps,) containing the time instances.
    - fig: Matplotlib figure object to which the plot will be added.
    - ax: Axis object for the figure to add the plots to.

    Returns:
    - None. Displays the plot on the given figure and axis.
    """
    nRegions = Z.shape[1]
    for i in range(nRegions):
        ax.plot(timestamps, Z[:, i], label=f"Region {i+1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurodynamic Value")
    ax.set_title("Neurodynamics vs Time")
    ax.legend()
    ax.grid(True)
    # fig.show()


def plot_Stimulus(U_stimulus, timestamps, fig, ax):
    """
    Plots the stimulus values over time for one or multiple regions.

    Parameters:
    - U_stimulus: 2D numpy array of shape (nRegions, nTimestamps) or 1D array of shape (nTimestamps,) containing stimulus values.
    - timestamps: 1D numpy array of shape (nTimestamps,) containing the time instances.
    - fig: Matplotlib figure object to which the plot will be added.
    - ax: Axis object for the figure to add the plots to.

    Returns:
    - None. Displays the plot on the given figure and axis.
    """
    nRegions = U_stimulus.shape[0] if U_stimulus.ndim == 2 else 1

    for i in range(nRegions):
        ax.plot(timestamps, U_stimulus[i] if nRegions > 1 else U_stimulus)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stimulus Value")
    ax.set_title("Stimulus vs Time")
    ax.legend([f"Region {i+1}" for i in range(nRegions)])
    ax.grid(True)
    # fig.show()


def plot_DHDQ(dq, dh, timestamps, fig, ax):
    """
    Plots dq (change in deoxyhemoglobin concentration) and dh (change in oxyhemoglobin concentration)
    over time for a specified number of regions on the given figure and axis.

    Parameters:
    - dq (numpy.ndarray): 2D array of shape (nRegions, nTimestamps) containing the dq values
      (change in deoxyhemoglobin concentration) for each region.
    - dh (numpy.ndarray): 2D array of shape (nRegions, nTimestamps) containing the dh values
      (change in oxyhemoglobin concentration) for each region.
    - timestamps (numpy.ndarray): 1D array of shape (nTimestamps,) containing the timestamps
      corresponding to dq and dh values.
    - fig (matplotlib.figure.Figure): The figure object on which the plot will be drawn.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axis object on which the plot will be drawn.

    Returns:
    - None. Modifies the given figure and axis to display the plot.
    """
    nRegions = dq.shape[0]
    for r in range(nRegions):
        ax.plot(timestamps, dq[r, :], label=f"dq Region {r+1}")
        ax.plot(timestamps, dh[r, :], label=f"dh Region {r+1}")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Relative Hemoglobin Concentration")
    ax.set_title("dq and dh over time")
    ax.legend()
    ax.grid(True)
    # fig.show()


def plot_Y(Y, timestamps, fig, ax):
    """
    Plots the optical density changes of dxy-Hb and oxy-Hb over time for a specified number of regions
    on the given figure and axis.

    Parameters:
    - Y (numpy.ndarray): 2D array of shape (2 * nRegions, nTimestamps) containing the optical density changes
      for dxy-Hb (odd rows) and oxy-Hb (even rows) for each region.
    - timestamps (numpy.ndarray): 1D array of shape (nTimestamps,) containing the timestamps
      corresponding to the optical density changes in Y.
    - fig (matplotlib.figure.Figure): The figure object on which the plot will be drawn.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axis object on which the plot will be drawn.

    Returns:
    - None. Modifies the given figure and axis to display the plot.
    """

    nRegions = Y.shape[0] // 2
    for r in range(nRegions):
        ax.plot(timestamps, Y[2 * r, :], label=f"dxy-Hb Region {r+1}")
        ax.plot(timestamps, Y[2 * r + 1, :], label=f"oxy-Hb Region {r+1}")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Optical Density Changes")
    ax.set_title("Optical Density Changes over time for dxy-Hb and oxy-Hb")
    ax.legend()
    ax.grid(True)
    # fig.show()


# Example usage:
# fig, ax = plt.subplots(figsize=(10,6))
# plot_Y(Y, timestamps, fig, ax)
