# author : Bayu Aditya
import numpy as np
import pandas as pd

def hamiltonian(k, input_dataframe):
    """Hamiltonian Tight Binding berdasarkan vektor K.
    
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