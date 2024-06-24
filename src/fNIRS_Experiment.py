from fNIRS_processing import MBLL


def fNIRS_processing():
    import os
    import numpy as np
    import pandas as pd

    file_name = "data/DAG_Connectivity.csv"
    df = pd.read_csv(file_name)

    # Data preprocessing
    Y_columns = [col for col in df.columns if col.startswith("Y")]
    Y_df = df[Y_columns]
    Timestamps = df["Timestamps"].values
    Y_array = Y_df.values.T
    OD = Y_array

    # Analysis setup
    e_coefficients = np.array([[735.8251, 1104.715], [1159.306, 785.8993]])
    DPF = np.array([10.3, 8.4])
    Distance = 3.0
    Hemodynamics = MBLL(e_coefficients, DPF, Distance, OD)

    return Timestamps, Hemodynamics, OD
