def MBLL(e_coefficients, DPF, Distance, OD):
    import numpy as np

    """
    Calculates hemodynamics from optical density (OD) changes for N regions using
    the modified Beer-Lambert law (MBLL), keeping the output shape consistent with OD.

    Parameters:
    - e_coefficients (numpy.ndarray): 2x2 array of molar extinction coefficients for 780nm and 850nm.
    - DPF (numpy.ndarray): 2-element array containing the differential path length factors for 780nm and 850nm.
    - Distance (float): The distance between emitter and detector in cm.
    - OD (numpy.ndarray): (Nregions * 2) x timestamps_length array of optical density changes, where
      each pair of rows corresponds to 780nm and 850nm for each region.

    Returns:
    - Hemodynamics (numpy.ndarray): (Nregions * 2) x timestamps_length array of hemodynamic changes,
      maintaining the input shape.
    """

    A_matrix = np.array(
        [
            [
                e_coefficients[0, 0] * Distance * DPF[0],
                e_coefficients[0, 1] * Distance * DPF[0],
            ],
            [
                e_coefficients[1, 0] * Distance * DPF[1],
                e_coefficients[1, 1] * Distance * DPF[1],
            ],
        ]
    )

    Nregions = OD.shape[0] // 2
    Hemodynamics = np.zeros_like(OD)

    # Process each region's OD data pair-wise
    for i in range(Nregions):
        # Extract OD data for the current region pair (780nm and 850nm)
        OD_region = OD[2 * i : 2 * i + 2, :]

        # Solve the linear system A * Hemodynamics = OD for each region pair
        Hemodynamics_region = np.linalg.solve(A_matrix, OD_region)

        # Place the result back in the corresponding positions
        Hemodynamics[2 * i : 2 * i + 2, :] = Hemodynamics_region

    return Hemodynamics
