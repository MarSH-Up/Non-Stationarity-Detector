import numpy as np
from scipy.integrate import odeint


def Neurodynamics_Model(Z, t, A, B, C, U):
    """
    Neurodynamics model for multiple brain regions.

    Parameters:
    - Z: The state of the system. Shape: (nRegions,).
    - t: Current time.
    - A: Connectivity matrix. Shape: (nRegions, nRegions).
    - B: Influence matrix. Shape: (nRegions, nRegions, number of inputs).
    - C: Input effect matrix. Shape: (nRegions, number of inputs).
    - U: Input matrix. Shape: (number of inputs, number of timestamps).

    Returns:
    - dZdt: The rate of change of the system's state.
    """

    nRegions = A.shape[0]
    nInputs = B.shape[2]
    index = min(int(t * 10), U.shape[1] - 1)

    # J_t computation according to the provided equation
    J_t = np.copy(A)

    for i in range(nInputs):
        J_t += U[i, index] * B[:, :, i]

    # Modify the diagonal of the connectivity matrix A
    SI = np.diag(A)
    new_diag = np.exp(SI) / 2 + SI
    A -= np.diagflat(new_diag)
    J_t = A + J_t

    # Calculate the rate of change of the system's state
    dZdt = np.dot(J_t, Z) + np.dot(C, U[:, index])
    return dZdt


def Neurodynamics(Z0, timestamps, A, B, C, U_stimulus):
    """
    Integrate the neurodynamics across all brain regions.

    Parameters:
    - Z0: Initial state of the system. Shape: (nRegions,).
    - timestamps: Array of time points.
    - A: Connectivity matrix. Shape: (nRegions, nRegions).
    - B: Influence matrix. Shape: (nRegions, nRegions, number of inputs).
    - C: Input effect matrix. Shape: (nRegions, number of inputs).
    - U_stimulus: Stimulus input matrix. Shape: (number of inputs, number of timestamps).

    Returns:
    - Z: The system's state at each timestamp. Shape: (number of timestamps, nRegions).
    """

    Z = odeint(Neurodynamics_Model, Z0, timestamps, args=(A, B, C, U_stimulus))

    return Z
