# Author : Bayu Aditya
import numpy as np
import time
from mpi4py import MPI

from src.preprocessing import extract, generate_input_hamiltonian
from src.k_path import k_mesh_orthorombic

from src.hamiltonian import hamiltonian_v3
from src.hamiltonian import hamiltonian_v4

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#________________________________________________________________________________

if rank == 0:
    print("[INFO] Using ", size, "Processor")

    k_grid = k_mesh_orthorombic(
        n1 = 23, n2 = 3, n3 = 17,
        a = 10.86909,
        b = 5.763864449563357*10.86909,
        c = 0.703931807842342*10.86909)
    n = len(k_grid)

    num_orbitals, parameterTB = extract(
        "/home/bayu/Documents/Undergraduate_Thesis/Data/Sr10Nb10O34/hr_files/Sr10_Nb10_O34_5x1x8_hr.dat", 
        max_cubic_cell=[3,3,3]
        )

    input_hamiltonian = generate_input_hamiltonian(
        data_parameter_TB = parameterTB, 
        filename_atomic_position = "/home/bayu/Documents/Undergraduate_Thesis/Data/Sr10Nb10O34/atomic_position.csv",
        filename_orbital_index="/home/bayu/Documents/Undergraduate_Thesis/Data/Sr10Nb10O34/orbital_index.csv",
        a = 10.86909,
        b = 5.763864449563357*10.86909,
        c = 0.703931807842342*10.86909
        )

    hamiltonian = np.memmap(
        filename = "/home/bayu/Documents/Undergraduate_Thesis/hamiltonian.npm",
        dtype = np.complex128, mode = 'w+',
        shape = (n, num_orbitals, num_orbitals)
        )

    idx = np.arange(0, n, dtype=np.int)
    split = np.array_split(idx, size)
    split_size = [len(split[i]) for i in range(len(split))]
    split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]

else:
#Create variables on other cores
    k_grid = None
    n = None
    idx = None
    split = None
    split_size = None
    split_disp = None
    num_orbitals = None
    input_hamiltonian = None
    hamiltonian = None

k_grid = comm.bcast(k_grid, root=0)
n = comm.bcast(n, root=0)
split_size = comm.bcast(split_size, root = 0)
split_disp = comm.bcast(split_disp, root = 0)
num_orbitals = comm.bcast(num_orbitals, root=0)
input_hamiltonian = comm.bcast(input_hamiltonian, root=0)

idx_local = np.zeros(split_size[rank], dtype=np.int)
comm.Scatterv([idx, split_size, split_disp, MPI.INTEGER8], idx_local, root=0)
print('rank ', rank, ': ', idx_local)
#______________________________________________________________________________
hamiltonian = np.memmap(
    filename = "/home/bayu/Documents/Undergraduate_Thesis/hamiltonian.npm",
    dtype = np.complex128, mode = 'r+',
    shape = (n, num_orbitals, num_orbitals)
    )

num = 0
for i in idx_local:
    start = time.time()
    # hamiltonian[i,:,:] = hamiltonian_v3(k_grid[i], input_hamiltonian)
    hamiltonian[i,:,:] = hamiltonian_v4(k_grid[i], input_hamiltonian, num_orbitals)
    num += 1
    print("[LOG] rank", rank, ", num", num, time.time() - start, "sec.")

print("[INFO] Generate hamiltonian over k-points in rank :", rank, "has been finished")