# author : Bayu Aditya
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial

def hamiltonian(k, input_dataframe):
    """Hamiltonian Tight Binding berdasarkan single K vector.
    
    Arguments:
        k {numpy.array} -- vektor k dengan ukuran (3,)
        input_dataframe {pandas.DataFrame} -- DataFrame berdasarkan vector lattice dari input_data
    
    Returns:
        numpy.array -- Matriks Hamiltonian tight binding dengan ukuran banyaknya jenis orbital.
    """
    mat = input_dataframe.to_numpy()

    hamiltonian = np.zeros(
        shape=(len(np.unique(mat[:,3])), len(np.unique(mat[:,3]))),dtype=np.complex128)

    for i, A in enumerate(np.unique(mat[:,3])):
        for j, B in enumerate(np.unique(mat[:,4])):
            df = mat[(mat[:,3]==A) & (mat[:,4]==B)]
            
            R = np.array([df[:,7], df[:,8], df[:,9]])
            sum_H = (df[:,5] + df[:,6]*1j)*np.exp(-1j*np.dot(k,R))
            sum_H = sum_H.sum()
            
            hamiltonian[i,j] = sum_H
    return hamiltonian

# def multiple_hamiltonian(k_path_grid, input_hamiltonian, hamiltonian_func):
#     multiple_hamiltonian = []
#     for i in tqdm(range(len(k_path_grid))):
#         ham = hamiltonian_func(k_path_grid[i], input_hamiltonian)
#         multiple_hamiltonian.append(ham)
#     multiple_hamiltonian = np.array(multiple_hamiltonian, dtype=np.complex128)
#     return multiple_hamiltonian

def multiple_hamiltonian(k_path_grid, input_hamiltonian, hamiltonian_func):
    num_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cpu)
    hamiltonian = partial(hamiltonian_func, input_dataframe=input_hamiltonian)
    multiple_hamiltonian = pool.map(hamiltonian, k_path_grid)
    multiple_hamiltonian = np.array(multiple_hamiltonian, dtype=np.complex128)
    return multiple_hamiltonian