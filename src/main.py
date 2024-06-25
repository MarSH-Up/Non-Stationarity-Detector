from Experiments import *
import numpy as np
import matplotlib.pyplot as plt
import lingam
import networkx as nx


from BilinearModel_CSVGenerator import write_to_csv
from BilinearModel_Hemodynamics import Hemodynamics
from BilinearModel_Neurodynamics import Neurodynamics
from BilinearModel_Optics import *
from BilinearModel_SemisyntheticNoise import *
from BilinearModel_StimulusGenerator import *
from BilinearModel_SyntheticNoise import *
from Parameters.Parameters_DAG3 import Parameters_DAG3
from Parameters.Parameters_DAG3_0 import Parameters
from BilinearModel_Plots import *
from fNIRS_Experiment import fNIRS_processing


def generateSignals_full(params, noise_type="Synthetic", percentNoise=1):
    parameters_dict = params
    parameters_dict["A"] = np.array(params["A"])
    parameters_dict["B"] = np.array(params["B"])
    parameters_dict["C"] = np.array(params["C"])

    U_stimulus, timestamps, Z, dq, dh, Y = fNIRS_Process(
        parameters_dict, noise_type, percentNoise
    )

    response = {
        "U_stimulus": U_stimulus.tolist(),
        "timestamps": timestamps.tolist(),
        "Z": Z.T.tolist(),
        "dq": dq.tolist(),
        "dh": dh.tolist(),
        "Y": Y.tolist(),
    }
    return timestamps, U_stimulus, Z, dq, dh, Y, response


def combine_signals():
    # Generate signals for the first set of parameters (DAG3)
    timestamps1, U_stimulus1, Z1, dq1, dh1, Y1, _ = generateSignals_full(
        Parameters_DAG3, "Synthetic", 1
    )

    # Generate signals for the second set of parameters (original)
    timestamps2, U_stimulus2, Z2, dq2, dh2, Y2, _ = generateSignals_full(
        Parameters, "Synthetic", 1
    )

    # Adjust the start time for the second run to ensure sequential timestamps
    start_time2 = timestamps1[-1] + (timestamps1[1] - timestamps1[0])
    timestamps2 += start_time2  # Adjust timestamps for the second dataset

    # Concatenate results along the correct dimension
    final_timestamps = np.concatenate([timestamps1, timestamps2])
    final_U_stimulus = np.concatenate(
        [U_stimulus1, U_stimulus2], axis=1
    )  # assuming U_stimulus is (nRegions, nTimestamps)
    final_Z = np.concatenate([Z1, Z2], axis=0)
    final_dq = np.concatenate(
        [dq1, dq2], axis=1
    )  # assuming dq is (nRegions, nTimestamps)
    final_dh = np.concatenate(
        [dh1, dh2], axis=1
    )  # assuming dh is (nRegions, nTimestamps)
    final_Y = np.concatenate(
        [Y1, Y2], axis=1
    )  # assuming Y is (2 * nRegions, nTimestamps)

    # Plotting the results
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    plot_Stimulus(final_U_stimulus, final_timestamps, fig, axs[0])
    plot_neurodynamics(final_Z, final_timestamps, fig, axs[1])
    plot_DHDQ(final_dq, final_dh, final_timestamps, fig, axs[2])
    plot_Y(final_Y, final_timestamps, fig, axs[3])
    plt.tight_layout()
    plt.show()

    return final_timestamps, final_U_stimulus, final_Z, final_dq, final_dh, final_Y


def apply_LiNGAM(dq_data, window_label):

    # Assume dq_data is already transposed to have shape (n_samples, n_regions)
    n_samples, n_regions = dq_data.shape

    # Apply the LiNGAM algorithm
    model = lingam.DirectLiNGAM()
    model.fit(dq_data)

    # Display the results
    print("Estimated adjacency matrix:")
    print(model.adjacency_matrix_)

    # Extract the causal ordering
    ordering = model.causal_order_
    print("Estimated causal ordering:", ordering)

    # Show the corresponding DAG (optional, requires networkx and matplotlib)
    G = nx.DiGraph()

    # Add nodes
    for i in range(n_regions):
        G.add_node(f"{i+1}")

    # Add edges based on the adjacency matrix
    adj_matrix = model.adjacency_matrix_
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                G.add_edge(f"{i+1}", f"{j+1}")

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=20,
    )
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    title = "Estimated Causal Graph:" + " " + window_label
    plt.title(title)
    plt.show()


def fNIRS_Process(Parameters, NoiseSelection, percentNoise=0):
    """
    Process the fNIRS data.

    Returns:
        U_stimulus: Stimulus signal
        timestamps: Array of timestamps
        Z: Neurodynamics
        dq, dh: Derivatives of blood volume and deoxyhemoglobin concentration
        Y: Optics output
        qj, pj: Hemodynamics data
    """
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        Parameters["freq"],
        Parameters["actionTime"],
        Parameters["restTime"],
        Parameters["cycles"],
        Parameters["A"].shape[0],
    )
    # Initialize the state of the neurodynamics
    Z0 = np.zeros([Parameters["A"].shape[0]])
    # Compute the neurodynamics of the system
    Z = Neurodynamics(
        Z0, timestamps, Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus
    )

    # Process hemodynamics
    qj, pj = Hemodynamics(Z.T, Parameters["P_SD"], Parameters["step"])

    pj_noise = pj.copy()
    qj_noise = qj.copy()
    dq, dh = calculate_hemoglobin_changes(pj, qj)
    if percentNoise > 0:
        # PhysiologicalNoise Inclusion
        if NoiseSelection == "Synthetic":
            noise_types = [
                "heart",
                "breathing",
                "vasomotion",
                "white",
            ]  # Types of noise to generate
            # Example percent error
            noises_with_gains = synthetic_physiological_noise_model(
                timestamps, noise_types, pj_noise, percentNoise
            )
            # Labels corresponding to the noise types
            labels = ["Heart Rate", "Vasomotion", "Breathing Rate", "White"]

            # Combine the noises into hemodyanmics
            combined_noises = combine_noises(noises_with_gains, pj_noise.shape[0])

            # Add the combined noise to qj and pj
            qj_noise = qj_noise + combined_noises
            pj_noise = pj_noise + combined_noises

        elif NoiseSelection == "Semisynthetic":
            semisynthetic_noises = semisynthecticDataExtraction(
                Parameters["A"].shape[0], Parameters["freq"], len(timestamps)
            )

            qj_noise, pj_noise = add_noise_to_hemodynamics(
                qj_noise, pj_noise, semisynthetic_noises, percentNoise, timestamps
            )

    # Process optics
    Y, dq, dh = BilinearModel_Optics(pj_noise, qj_noise, U_stimulus, Parameters["A"])

    return U_stimulus, timestamps, Z, dq, dh, Y


def analyze_signal_windows(
    signal_piecewise,
    window_size,
    overlap,
    analyze_stationarity,
    moment_variance,
):

    num_windows = (signal_piecewise.shape[1] - window_size) // (
        window_size - overlap
    ) + 1

    moments = []
    stationarity_results = []
    stationarity_status = []
    independent_stationarity_status = []

    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = start + window_size
        window_signal = signal_piecewise[0, start:end]
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
            # independent_status = "Non-Stationary"
            independent_status = "Stationary"
        elif stationarity_result["kpss_p"] > 0.05:
            # independent_status = "Stationary"
            independent_status = "Non-Stationary"
        else:
            # independent_status = "Non-Stationary"
            independent_status = "Stationary"

        independent_stationarity_status.append(independent_status)

    # Print stationarity test results and compare consecutive windows
    for i in range(1, num_windows):
        current_result = stationarity_results[i]
        prev_result = stationarity_results[i - 1]
        transition_detected = False
        current_window_signal_full = signal_piecewise[
            :, i * (window_size - overlap) : i * (window_size - overlap) + window_size
        ]
        """
        print(
            f"Window {i} to {i + 1}: Current (ADF p-value={current_result['adf_p']:.4f}, KPSS p-value={current_result['kpss_p']:.4f}) vs "
            f"Previous (ADF p-value={prev_result['adf_p']:.4f}, KPSS p-value={prev_result['kpss_p']:.4f})"
        )
        """
        current_status = ""
        # Determine current window stationarity status
        if current_result["adf_p"] > 0.05 and current_result["kpss_p"] < 0.05:
            # current_status = "Non-Stationary"
            current_status = "Stationary"
        elif current_result["adf_p"] < 0.05 and current_result["kpss_p"] > 0.05:
            # current_status = "Stationary"
            current_status = "Non-Stationary"

        else:
            # current_status = "Non-Stationary"
            current_status = "Stationary"

        # Determine previous window stationarity status
        if prev_result["adf_p"] > 0.05 and prev_result["kpss_p"] < 0.05:
            # prev_status = "Non-Stationary"
            prev_status = "Stationary"
        elif prev_result["adf_p"] < 0.05 and prev_result["kpss_p"] > 0.05:
            # prev_status = "Stationary"
            prev_status = "Non-Stationary"
        else:
            # current_status = "Non-Stationary"
            prev_status = "Stationary"

        # Check for transitions between the windows
        if current_status != prev_status:
            transition_detected = True
            print(
                f"Transition detected from {prev_status} to {current_status} in window {i + 1}."
            )
            print("Trigger LinGAM")
            # Here trigger Lingam

        else:
            transition_detected = False
            print(f"No transition detected in window {i + 1}.")

        # Print transition information based on detection
        print(
            f"Window {i} to {i + 1}: {'Non-Stationary' if current_status == 'Non-Stationary' else 'Stationary'}"
        )

        if transition_detected:
            apply_LiNGAM(current_window_signal_full.T, f"Window {i}")

        # Append the current status to the stationarity status list
        stationarity_status.append(current_status)

    moments_observation(moments, num_windows)
    return (moments, independent_stationarity_status, stationarity_status)


def main():

    # Timestamps, Hemodynamics, OD = fNIRS_processing()
    final_timestamps, final_U_stimulus, final_Z, final_dq, final_dh, final_Y = (
        combine_signals()
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for better distinction
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    initial = final_dh[:, : len(final_dh[0]) // 4]
    oxyhemo = final_dh

    apply_LiNGAM(initial.T, "Full_Signal")

    window_size = 20 * 10
    overlap = 10 * 10

    moments, independent_stationarity_status, stationarity_status = (
        analyze_signal_windows(
            oxyhemo, window_size, overlap, analyze_stationarity, moment_variance
        )
    )
    binary_values_Independent = [
        0 if status == "Stationary" else 1 for status in independent_stationarity_status
    ]
    binary_values_statys = [
        0 if status == "Stationary" else 1 for status in stationarity_status
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

    results = evaluate_stationarity(moments)
    print("Stationarity Results Classical Signal:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print("Non-stationary Model generation")
    main()
