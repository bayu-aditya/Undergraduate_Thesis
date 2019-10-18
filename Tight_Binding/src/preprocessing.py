# Author : Bayu Aditya
import numpy as np
import pandas as pd
from .concat import concatenate_atom_orbital

def extract(filename_hr_file, max_cubic_cell=[100,100,100]):
    # Menentukan letak baris awal pembacaan parameter
    #   output : num_row
    with open(filename_hr_file) as f:
        data = f.readlines()
    for num_row, row in enumerate(data):
        if (len(row.split()) == 7):
            break  

    # Pembacaan hr_file dimulai dari baris "num_row"
    num_init_row = num_row
    data = pd.read_csv(
        filename_hr_file, 
        names=["X", "Y", "Z", "A", "B", "Re", "Im"], 
        skiprows=num_init_row,
        delim_whitespace=True
        )

    # Seleksi parameter berdasarkan "max_cubic_cell"
    max_X = max_cubic_cell[0]
    max_Y = max_cubic_cell[1]
    max_Z = max_cubic_cell[2]
    filter_X = abs(data.X) <= max_X
    filter_Y = abs(data.Y) <= max_Y
    filter_Z = abs(data.Z) <= max_Z
    data = data[filter_X & filter_Y & filter_Z]
    num_orbitals = len(data.A.unique())
    print(
        "[INFO] input data, max_X : ", data.X.unique(),
        ", max_Y : ", data.Y.unique(),
        ", max_Z : ", data.Z.unique())
    print("[INFO] parameter tight-binding has been extracted.")
    return num_orbitals, data

def generate_input_hamiltonian(data_parameter_TB, filename_atomic_position, filename_orbital_index, a, b, c):
    # membaca dataframe dan merge
    atom_pos_df = pd.read_csv(filename_atomic_position)
    orbital_df = pd.read_csv(filename_orbital_index)
    orbital_df = orbital_df.merge(atom_pos_df)
    
    # menyatukan dataframe atomic_position dan orbital_index
    orbital_A, orbital_B = concatenate_atom_orbital(filename_atomic_position, filename_orbital_index, a, b, c)

    # merge parameter and orbital
    merge_df = data_parameter_TB.merge(orbital_A)
    merge_df = merge_df.merge(orbital_B)

     # Mendapatkan vektor lattice
    merge_df["Rx"] = (merge_df["Bx"]-merge_df["Ax"]) + a*merge_df["X"]
    merge_df["Ry"] = (merge_df["By"]-merge_df["Ay"]) + b*merge_df["Y"]
    merge_df["Rz"] = (merge_df["Bz"]-merge_df["Az"]) + c*merge_df["Z"]
    merge_df = merge_df.drop(["Ax", "Ay", "Az", "Bx", "By", "Bz"], axis=1)
    merge_df = merge_df.drop(["X", "Y", "Z"], axis=1)

    input_hamiltonian = merge_df
    input_hamiltonian = input_hamiltonian.to_numpy()
    print("[INFO] input for construct hamiltonian has been created.")
    return input_hamiltonian