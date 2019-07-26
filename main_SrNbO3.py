# author : Bayu Aditya
import numpy as np
import time
import matplotlib.pyplot as plt
from Tight_Binding.tight_binding.extract_parameter import _concatenate_atom_orbital
from Tight_Binding.tight_binding.extract_parameter import input_data
from Tight_Binding.tight_binding.k_path import k_path
from Tight_Binding.tight_binding.hamiltonian import hamiltonian
from Tight_Binding.tight_binding.hamiltonian import multiple_hamiltonian
import tensorflow as tf

gpus = tf.config.experimental.get_visible_devices()
tf.config.experimental.set_memory_growth(gpus[1], True)

# orbitalA, orbitalB = _concatenate_atom_orbital(
#     "Data/SrNbO3/atomic_position.csv", 
#     "Data/SrNbO3/orbital_index.csv", 
#     a = 4.089,
#     b = 4.089,
#     c = 4.089)

parameter = input_data(
    parameter_TB="Data/SrNbO3/srnbo3_hr.dat", 
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
k_p = k_path(k, 30)
print(len(k_p))

start = time.time()
ham = multiple_hamiltonian(k_p, input_hamiltonian, hamiltonian)
print(time.time() - start)

print("Solve Eigenvalues")
# eig = np.linalg.eigvals(ham)

ham_d = tf.convert_to_tensor(ham)
eig_d = tf.linalg.eigvalsh(ham_d)
eig = eig_d.numpy()

for i in range(eig.shape[-1:][0]):
    plt.plot(eig[:,i])
plt.show()

print("Selesai")