# Author : Bayu Aditya
import numpy as np
import os
import time
import argparse
from mpi4py import MPI
from tools.generator import matrix_generator
from src.variable import variable

parser = argparse.ArgumentParser()
parser.add_argument("input", help="input file 'information.json'", type=str)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

var = variable(comm, args.input)
LOC_HAMILTONIAN = var.hamiltonian_dos
N_KPOINTS = int(var.num_k)
NUM_ORBITALS = var.num_orbitals
OUT_DIR = var.outdir
STEP = var.step
RANGE_FREQ = [var.start, var.stop]
NUM_FREQ = var.num_freq
ZERO_PLUS = var.zero_plus

if rank == 0:
    if not os.path.exists(os.path.join(var.outdir, "temp")):
        os.makedirs(os.path.join(var.outdir, "temp"))

    frequency = np.linspace(RANGE_FREQ[0], RANGE_FREQ[1], num=NUM_FREQ, dtype=np.float64)
    dos = np.zeros_like(frequency)
    pdos = np.zeros(shape=(len(frequency),NUM_ORBITALS), dtype=np.float64)

    idx = np.arange(0, N_KPOINTS, dtype=np.int)
    split = np.array_split(idx, size)
    split_size = [len(split[i]) for i in range(len(split))]
    split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]

else:
    frequency = None
    dos = None
    pdos = None
    idx = None
    split = None
    split_size = None
    split_disp = None

frequency = comm.bcast(frequency, root = 0)
split_size = comm.bcast(split_size, root = 0)
split_disp = comm.bcast(split_disp, root = 0)

idx_local = np.zeros(split_size[rank], dtype=np.int)
comm.Scatterv([idx, split_size, split_disp, MPI.INTEGER8], idx_local, root=0)
print('rank ', rank, 'calculate', len(idx_local), "k-points")

#____________________________READING-HAMILTONIAN__________________________________
hamiltonian = np.memmap(
    LOC_HAMILTONIAN, 
    shape=(N_KPOINTS, NUM_ORBITALS, NUM_ORBITALS),
    dtype=np.complex128, mode="r"
    )

dir_ham_local = os.path.join(OUT_DIR, "temp", str(rank))
ham_local = np.memmap(
    dir_ham_local, shape = (len(idx_local), NUM_ORBITALS, NUM_ORBITALS),
    dtype=np.complex128, mode="w+"
    )

for i in range(len(idx_local)):
    ham_local[i,:,:] = hamiltonian[i,:,:]
#________________________________________________________________________________`

for i, freq in enumerate(frequency):
    if rank == 0:
        begin = time.time()

    green_diag_per_process = np.zeros(shape=NUM_ORBITALS, dtype=np.float64)
    for ham in matrix_generator(ham_local, step=STEP):
        identity = np.eye(NUM_ORBITALS, dtype=np.complex128)
        green_batch = np.linalg.inv(
            (freq+ZERO_PLUS*1j)*identity - ham
        )
        green_batch = np.imag(np.diag(np.sum(green_batch, axis=0)))
        green_diag_per_process += green_batch

    green_diag = comm.gather(green_diag_per_process, root=0)

    if rank == 0:
        green_diag = np.array(green_diag, dtype=np.float64).sum()
        pdos[i] = -(1.0/np.pi)*green_diag
        dos[i] = -(1.0/np.pi)*green_diag.sum()
        print(time.time() - begin, "Sec. Frequency", i+1, "of", len(frequency))

if rank == 0:
    loc_save = os.path.join(OUT_DIR, "DOS_array.npz")
    np.savez(loc_save, frequency, dos, pdos)
    print("[INFO] File has been saved in :", loc_save)