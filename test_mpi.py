import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
n = 77

if rank == 0:
    d1 = np.arange(0, n, dtype=np.float64)
    split = np.array_split(d1, size)
    split_size = [len(split[i]) for i in range(len(split))]
    split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]
    time.sleep(10)

else:
#Create variables on other cores
    d1 = None
    split = None
    split_size = None
    split_disp = None

split_size = comm.bcast(split_size, root = 0)
split_disp = comm.bcast(split_disp, root = 0)
d1_local = np.zeros(split_size[rank], dtype=np.float64)
comm.Scatterv([d1, split_size, split_disp, MPI.DOUBLE], d1_local, root=0)
print('rank ', rank, ': ', d1_local)