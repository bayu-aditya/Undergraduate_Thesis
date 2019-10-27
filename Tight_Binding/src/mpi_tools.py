# author : Bayu Aditya
import numpy as np
from mpi4py import MPI

def scatter_index(communication, index):
    rank = communication.Get_rank()
    size = communication.Get_size()
    index = np.array(index, dtype=np.int)
    if rank == 0:
        split = np.array_split(index, size)
        split_size = [len(split[i]) for i in range(len(split))]
        split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]
    else:
        split = None
        split_size = None
        split_disp = None
    split_size = communication.bcast(split_size, root = 0)
    split_disp = communication.bcast(split_disp, root = 0)  
    idx_local = np.zeros(split_size[rank], dtype=np.int)
    communication.Scatterv([index, split_size, split_disp, MPI.INTEGER8], idx_local, root=0)
    return idx_local