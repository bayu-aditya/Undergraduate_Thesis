# author : Bayu Aditya
import numpy as np
import pandas as pd
from tqdm import tqdm

def hamiltonian(k, input_dataframe):
    """Hamiltonian Tight Binding berdasarkan single K vector.
    
    Arguments:
        k {numpy.array} -- vektor k dengan ukuran (3,)
        input_dataframe {pandas.DataFrame} -- DataFrame berdasarkan vector lattice dari input_data
    
    Returns:
        numpy.array -- Matriks Hamiltonian tight binding dengan ukuran banyaknya jenis orbital.
    """
    a = input_dataframe

    hamiltonian = np.zeros(shape=(len(a.A.unique()), len(a.B.unique())),
                        dtype=np.complex128)

    for i, A in enumerate(a.A.unique()):
        for j, B in enumerate(a.B.unique()):
            filter_A = a.A == A
            filter_B = a.B == B
            df = a[filter_A & filter_B]
            
            sum_H = 0+0j
            for idx in range(len(df)):
                e = df.Re.iloc[idx]
                t = df.Im.iloc[idx]
                R = [df.Rx.iloc[idx], df.Ry.iloc[idx], df.Rz.iloc[idx]]
                H_idx = e + t*np.exp(1j*np.dot(k,R))
                sum_H += H_idx        
            hamiltonian[i,j] = sum_H
    return hamiltonian

def multiple_hamiltonian(k_path_grid, input_hamiltonian, hamiltonian_func):
    multiple_hamiltonian = []
    for i in tqdm(range(len(k_path_grid))):
        ham = hamiltonian_func(k_path_grid[i], input_hamiltonian)
        multiple_hamiltonian.append(ham)
    multiple_hamiltonian = np.array(multiple_hamiltonian, dtype=np.complex128)
    return multiple_hamiltonian