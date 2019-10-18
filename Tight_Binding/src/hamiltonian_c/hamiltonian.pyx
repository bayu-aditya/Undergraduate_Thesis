# author Bayu Aditya
import numpy as np
cimport numpy as np
import cython

DTYPE1 = np.float64
ctypedef np.float64_t DTYPE1_t

DTYPE2 = np.complex128
ctypedef np.complex128_t DTYPE2_t

def hamiltonian_cython(np.ndarray[DTYPE1_t, ndim=1] k, np.ndarray[DTYPE1_t, ndim=2] input_dataframe):

    cdef double[:,:] mat = input_dataframe
    cdef np.ndarray[DTYPE2_t, ndim=2] ham
    cdef np.ndarray[DTYPE1_t, ndim=1] R
    cdef np.int i, j, idx

    ham = np.zeros(shape=(len(np.unique(mat[:,0])), len(np.unique(mat[:,1]))),dtype=np.complex128)

    for idx in range(len(mat)):
        i, j = np.int(mat[idx,0]-1), np.int(mat[idx,1]-1)
        R = np.array([mat[idx,4], mat[idx,5], mat[idx,6]])
        ham[i,j] += (mat[idx,2] + mat[idx,3]*1j)*np.exp(-1j*np.dot(k,R))
    return ham