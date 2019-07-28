# author : Bayu Aditya
import numpy as np
import time
import matplotlib.pyplot as plt
from Tight_Binding.tight_binding.extract_parameter import _concatenate_atom_orbital
from Tight_Binding.tight_binding.extract_parameter import input_data
from Tight_Binding.tight_binding.k_path import k_path
from Tight_Binding.tight_binding.hamiltonian import hamiltonian
from Tight_Binding.tight_binding.hamiltonian import multiple_hamiltonian
# import tensorflow as tf

# gpus = tf.config.experimental.get_visible_devices()
# tf.config.experimental.set_memory_growth(gpus[1], True)

parameter = input_data(
    parameter_TB="Data/SrNbO3/hr_files/srnbo3_hr_15x15x15.dat", 
    atomic_position="Data/SrNbO3/atomic_position.csv", 
    orbital_index="Data/SrNbO3/orbital_index.csv",
    a = 4.089,
    b = 4.089,
    c = 4.089)
input_hamiltonian = parameter.vec_lattice()

a = 4.089
pi = np.pi
k = np.array(
    [[0.0, 0.0, pi/a],
     [0.0, 0.0, 0.0],
     [pi/a, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, pi/a]]
     )
k_p = k_path(k, 100)
print(len(k_p))

start = time.time()
ham = multiple_hamiltonian(k_p, input_hamiltonian, hamiltonian)
print(time.time() - start)
np.save("Data/SrNbO3/hamiltonian/hamiltonian.npy", ham)
print("Selesai")

# print("Solve Eigenvalues")
# ham = np.load("hamiltonian_15x15x15.npy")
# eig = np.linalg.eigvals(ham)
# # ham_d = tf.convert_to_tensor(ham)
# # eig_d = tf.linalg.eigvalsh(ham_d)
# # eig = eig_d.numpy()

# for i in range(eig.shape[-1:][0]):
#     plt.plot(eig[:,i], '-')
# plt.ylim([13.1845-7.0, 13.1845+2.0])
# plt.show()

# print("Selesai")