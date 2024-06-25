import numpy as np


def Hemodynamics(Z, P_SD, Step):
    """
    Simulate the hemodynamic response for multiple brain regions using a bilinear model.

    Parameters:
    - Z: Neural activity. A 2D array of shape (nRegions, simulationLength) where each row represents a brain region and each column represents a time point.
    - P_SD: Parameters for each region. A 2D array where each row represents a different parameter and each column represents a brain region.
    - Step: Time step for the Euler method.

    Returns:
    - qj: Deoxyhemoglobin concentration. A 2D array of shape (nRegions, simulationLength) where each row represents a brain region and each column represents a time point.
    - pj: Total hemoglobin concentration. A 2D array of shape (nRegions, simulationLength) where each row represents a brain region and each column represents a time point.
    """

    # Determine the number of brain regions and simulation length
    nRegions, simulationLength = Z.shape

    # Initialization of hemodynamic state variables for each of the nRegions over the simulation time

    # Vasodilatory Signal variable for all regions
    Sj = np.full((nRegions, simulationLength), np.nan)
    Sj[:, 0] = np.zeros(nRegions)

    # Rate of blood volume for all regions
    Vj = np.zeros((nRegions, simulationLength))
    Vj[:, 0] = np.exp(np.zeros(nRegions))

    # HbT concentration (Rate) for all regions
    pj = np.full((nRegions, simulationLength), np.nan)
    pj[:, 0] = np.exp(np.zeros(nRegions))

    # HbR concentration (Rate) for all regions
    qj = np.full((nRegions, simulationLength), np.nan)
    qj[:, 0] = np.exp(np.zeros(nRegions))

    # Inflow for all regions
    fjin = np.full((nRegions, simulationLength), np.nan)
    fjin[:, 0] = np.exp(np.zeros(nRegions))
    fjout_s = np.full((nRegions, simulationLength), np.nan)
    fjout_s1 = np.full((nRegions, simulationLength), np.nan)

    # Define constants for the hemodynamic model. These are standard values in the literature.
    H = [0.64, 0.32, 2.00, 0.32, 0.32, 2.00]

    # Extract and adjust parameters for each of the nRegions

    Kj = H[0] * np.exp(np.concatenate(([0], [P_SD[0, 0] for _ in range(nRegions - 1)])))
    Yj = H[1] * np.exp(np.concatenate(([0], [P_SD[1, 0] for _ in range(nRegions - 1)])))
    Tj = H[2] * np.exp(np.concatenate(([0], [P_SD[2, 0] for _ in range(nRegions - 1)])))
    Tjv = H[5] * np.exp(
        np.concatenate(([0], [P_SD[3, 0] for _ in range(nRegions - 1)]))
    )
    phi = H[3]

    # Hemodynamics simulation for each time point using the Euler method for all regions
    for t in range(1, simulationLength):
        # Compute changes in hemodynamic state variables across all regions
        Sj_dot = Z[:, t - 1] - Kj * Sj[:, t - 1] - Yj * (fjin[:, t - 1] - 1)
        fjin_dot = Sj[:, t - 1]

        fv_s = Vj[:, t - 1] ** (1 / phi)
        Vj_dot = (fjin[:, t - 1] - fv_s) / (Tj * Tjv * Vj[:, t - 1])
        fjout = fv_s + Tjv * Vj_dot
        Efp = (1 - (1 - H[4]) ** (1 / fjin[:, t - 1])) / H[4]
        qj_dot = ((fjin[:, t - 1] * Efp - fjout * qj[:, t - 1]) / Vj[:, t - 1]) / (
            Tj * qj[:, t - 1]
        )
        pj_dot = (fjin[:, t - 1] - (fjout * pj[:, t - 1]) / Vj[:, t - 1]) / Tj

        # Update the state variables for the next time point for all regions using the Euler method
        Sj[:, t] = Sj[:, t - 1] + Step * Sj_dot
        Vj[:, t] = Vj[:, t - 1] + Step * Vj_dot
        fjin[:, t] = fjin[:, t - 1] + Step * fjin_dot
        qj[:, t] = qj[:, t - 1] + Step * qj_dot
        pj[:, t] = pj[:, t - 1] + Step * pj_dot
        fjout_s[:, t] = fjout
        fjout_s1[:, t] = Efp

    return qj, pj


def HemoglobinConcentrations(qj, pj):
    deltaQ = (qj - 1) * (71 * (1 - 0, 65))
    deltaP = (pj - 1) * 71
    deltaH = deltaP - deltaQ

    return deltaH, deltaQ
