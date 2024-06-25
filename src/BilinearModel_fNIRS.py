import sys

import numpy as np

sys.path.append(
    "/Users/mariodelossantos/Desktop/Research/PhD/PhD-Repository/Bilinear_model_fNIRS/src/components"
)
from BilinearModel_Hemodynamics import *
from BilinearModel_Neurodynamics_v1 import *
from BilinearModel_Optics import *
from BilinearModel_StimulusGenerator import *
from BilinearModel_SemisyntheticNoise import (
    add_noise_to_hemodynamics,
    semisynthecticDataExtraction,
)

from BilinearModel_SyntheticNoise import (
    combine_noises,
    synthetic_physiological_noise_model,
)

P_SD = np.array(
    [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
)


def fNIRS_Process_test(Parameters, NoiseSelection: str, percentNoise: str):
    """
    Process the fNIRS data.

    Returns:
        U_stimulus: Stimulus signal
        timestamps: Array of timestamps
        Z: Neurodynamics
        dq, dh: Derivatives of blood volume and deoxyhemoglobin concentration
        Y: Optics output
    """

    # Generate stimulus train
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
    qj, pj = Hemodynamics(Z.T, P_SD, Parameters["step"])

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
