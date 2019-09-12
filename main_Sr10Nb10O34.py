# author : Bayu Aditya
import numpy as np
import time
import matplotlib.pyplot as plt
from Tight_Binding.tight_binding.extract_parameter import _concatenate_atom_orbital
from Tight_Binding.tight_binding.extract_parameter import input_data
from Tight_Binding.tight_binding.k_path import k_path, k_path_custom
from Tight_Binding.tight_binding.hamiltonian import hamiltonian
from Tight_Binding.tight_binding.hamiltonian import multiple_hamiltonian
# import tensorflow as tf

# gpus = tf.config.experimental.get_visible_devices()
# tf.config.experimental.set_memory_growth(gpus[1], True)

parameter = input_data(
    parameter_TB="Data/Sr10Nb10O34/hr_files/Sr10_Nb10_O34_5x1x8_hr.dat", 
    atomic_position="Data/Sr10Nb10O34/atomic_position.csv", 
    orbital_index="Data/Sr10Nb10O34/orbital_index.csv",
    a = 10.86909,
    b = 5.763864449563357*10.86909,
    c = 0.703931807842342*10.86909)
input_hamiltonian = parameter.vec_lattice()

a = 10.86909
b = 5.763864449563357*10.86909
c = 0.703931807842342*10.86909
pi = np.pi
k = np.array(
    [[0.0, 0.0, pi/c],
     [0.0, 0.0, 0.0],
     [pi/a, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, pi/b, 0.0]]
     )
# k_p = k_path(k, 10)
k_p = k_path_custom(k, [30,28,28,5])
print(len(k_p))

# start = time.time()
# ham = multiple_hamiltonian(k_p, input_hamiltonian, hamiltonian, num_process="all")
# print(time.time() - start)
# np.save("Data/Sr10Nb10O34/hamiltonian/hamiltonian.npy", ham)
# print("Selesai")

# print("Solve Eigenvalues")
# ham = np.load("Data/Sr10Nb10O34/hamiltonian/hamiltonian_band_5x1x8.npy")
# ham_d = tf.convert_to_tensor(ham)
# eig_d = tf.linalg.eigvalsh(ham_d)
# eig = eig_d.numpy()

# for i in range(eig.shape[-1:][0]):
#     plt.plot(eig[:,i], color='black')
# # plt.ylim([13.1845-7.0, 13.1845+2.0])
# plt.show()
# print("Selesai")