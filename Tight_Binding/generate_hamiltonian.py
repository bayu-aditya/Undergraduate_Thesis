# Author : Bayu Aditya
import numpy as np
import argparse
import time
from mpi4py import MPI

from src.variable import variable
from src.preprocessing import extract, generate_input_hamiltonian
from src.k_path import k_mesh_orthorombic, k_path_custom
from src.mpi_tools import scatter_index

# from src.hamiltonian import hamiltonian_v3
from src.hamiltonian import hamiltonian_v4

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#===============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file JSON", type=str)
args = parser.parse_args()

var = variable(comm, args.input)

#===============================================================================

if rank == 0:
    var.summary()
    print("GENERATE HAMILTONIAN FOR")
    print("    Band Structures          : {}".format(var.mode_bands))
    print("    Density of States        : {}\n".format(var.mode_dos))

    num_orbitals, parameterTB = extract(
        filename_hr_file = var.hr_file, 
        max_cubic_cell = var.max_cell
        )
    var._data["num_orbitals"] = num_orbitals
    var.generate_json()

    input_hamiltonian = generate_input_hamiltonian(
        data_parameter_TB = parameterTB, 
        filename_atomic_position = var.atomic_position,
        filename_orbital_index = var.orbital_index,
        a = var.a, b = var.b, c = var.c
        )

    if var.mode_dos:
        k_grid = k_mesh_orthorombic(
            n1 = var.kpoint[0], n2 = var.kpoint[1], n3 = var.kpoint[2],
            a = var.a, b = var.b, c = var.c)
        num_k_dos = len(k_grid)
        hamiltonian_dos = np.memmap(
            filename = var.hamiltonian_dos,
            dtype = np.complex128, mode = 'w+',
            shape = (num_k_dos, num_orbitals, num_orbitals)
            )
    if var.mode_bands:
        k_path = k_path_custom(
            k_point_selection = var.kpath, n = var.num_kpath
        )
        num_k_band = len(k_path)
        hamiltonian_band = np.memmap(
            filename = var.hamiltonian_bands,
            dtype = np.complex128, mode = 'w+',
            shape = (num_k_band, num_orbitals, num_orbitals)
            )
else:
#Create variables on other cores
    k_grid = None
    k_path = None
    num_k_dos = None
    num_k_band = None
    num_orbitals = None
    input_hamiltonian = None

num_orbitals = comm.bcast(num_orbitals, root=0)
input_hamiltonian = comm.bcast(input_hamiltonian, root=0)

if var.mode_dos:
    k_grid = comm.bcast(k_grid, root=0)
    num_k_dos = comm.bcast(num_k_dos, root=0)
    idx_kp_dos = np.arange(0, num_k_dos, dtype=np.int)
    idx_kp_dos_local = scatter_index(comm, idx_kp_dos)
    print('[INFO] Rank ', rank, 'generate hamiltonian dos by', len(idx_kp_dos_local), "k-points")
    comm.barrier()
    if rank == 0:
        print(70*"=")
comm.barrier()
if var.mode_bands:
    k_path = comm.bcast(k_path, root=0)
    num_k_band = comm.bcast(num_k_band, root=0)
    idx_kp_band = np.arange(0, num_k_band, dtype=np.int)
    idx_kp_band_local = scatter_index(comm, idx_kp_band)
    print('[INFO] Rank ', rank, 'generate hamiltonian band by', len(idx_kp_band_local), "k-points")
    comm.barrier()
    if rank == 0:
        print(70*"=")

#===============================================================================
if var.mode_dos:
    hamiltonian_dos = np.memmap(
        filename = var.hamiltonian_dos,
        dtype = np.complex128, mode = 'r+',
        shape = (num_k_dos, num_orbitals, num_orbitals)
        )
    num = 0
    for i in idx_kp_dos_local:
        start = time.time()
        # hamiltonian[i,:,:] = hamiltonian_v3(k_grid[i], input_hamiltonian)
        hamiltonian_dos[i,:,:] = hamiltonian_v4(k_grid[i], input_hamiltonian, num_orbitals)
        num += 1
        print("[LOG] rank", rank, ", num", num, time.time() - start, "sec.")

    print("[INFO] Generate hamiltonian dos over k-points in rank :", rank, "has been finished. Waiting for other processor...")
    comm.barrier()
    if rank == 0:
        print("[INFO] Finishing for generate hamiltonian density of states\n", 70*"=")

# ==============================================================================

if var.mode_bands:
    hamiltonian_band = np.memmap(
        filename = var.hamiltonian_bands,
        dtype = np.complex128, mode = 'r+',
        shape = (num_k_band, num_orbitals, num_orbitals)
        )
    num = 0
    for i in idx_kp_band_local:
        start = time.time()
        # hamiltonian[i,:,:] = hamiltonian_v3(k_grid[i], input_hamiltonian)
        hamiltonian_band[i,:,:] = hamiltonian_v4(k_path[i], input_hamiltonian, num_orbitals)
        num += 1
        print("[LOG] rank", rank, ", num", num, time.time() - start, "sec.")

    print("[INFO] Generate hamiltonian band over k-points in rank :", rank, "has been finished. Waiting for other processor...")
    comm.barrier()
    if rank == 0:
        print("[INFO] Finishing for generate hamiltonian band structures\n", 70*"=")