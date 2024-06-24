from Experiments import *
import matplotlib.pyplot as plt
import lingam
import networkx as nx

from fNIRS_Experiment import fNIRS_processing


def apply_LiNGAM(dq_data):

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

    plt.title("Estimated Causal Graph")
    plt.show()


def analyze_signal_windows(
    signal_piecewise,
    window_size,
    overlap,
    analyze_stationarity,
    moment_variance,
):
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

        print(
            f"Window {i} to {i + 1}: Current (ADF p-value={current_result['adf_p']:.4f}, KPSS p-value={current_result['kpss_p']:.4f}) vs "
            f"Previous (ADF p-value={prev_result['adf_p']:.4f}, KPSS p-value={prev_result['kpss_p']:.4f})"
        )
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
            prev_status = "Stationary"

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

        # Append the current status to the stationarity status list
        stationarity_status.append(current_status)

    moments_observation(moments, num_windows)
    return (moments, independent_stationarity_status, stationarity_status)


def main():
    n_region = 3
    Timestamps, Hemodynamics, OD = fNIRS_processing()
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

    for i in range(0, n_region * 2, 2):
        region_index = i // 2 + 1
        ax.plot(
            Timestamps,
            OD[i, :],
            label=f"OD{region_index}",
            color=colors[i % len(colors)],
        )
        ax.plot(
            Timestamps,
            OD[i + 1, :],
            label=f"OD{region_index}",
            color=colors[(i + 1) % len(colors)],
        )

    ax.set_title("Optics Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("OD Changes")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(0, n_region * 2, 2):
        region_index = i // 2 + 1
        ax.plot(
            Timestamps,
            Hemodynamics[i, :],
            label=f"HbO_Region{region_index}",
            color=colors[i % len(colors)],
        )
        ax.plot(
            Timestamps,
            Hemodynamics[i + 1, :],
            label=f"HbR_Region{region_index}",
            color=colors[(i + 1) % len(colors)],
        )

    ax.set_title("Hemodynamics over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Concentration Changes")
    ax.legend()
    plt.show()

    indices = [i * 2 + 1 for i in range(n_region)]
    oxyhemo = Hemodynamics[indices, :]
    print(len(oxyhemo))
    window_size = 20 * 10
    overlap = 10 * 10

    moments, independent_stationarity_status, stationarity_status = (
        analyze_signal_windows(
            oxyhemo[0], window_size, overlap, analyze_stationarity, moment_variance
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
