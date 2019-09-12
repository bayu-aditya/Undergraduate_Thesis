# author : Bayu Aditya
import numpy as np
import matplotlib.pyplot as plt
from Tight_Binding.tight_binding.extract_parameter import input_data
from Tight_Binding.tight_binding.k_path import k_path
from Tight_Binding.tight_binding.hamiltonian import multiple_hamiltonian
from Tight_Binding.tight_binding.hamiltonian import hamiltonian

loc_parameter = "Data/example/sample_parameter.dat"
loc_atom_position = "Data/example/atomic_position.csv"
loc_orbital_index = "Data/example/orbital_index.csv"

parameter = input_data(loc_parameter, loc_atom_position, loc_orbital_index, 
           a = 1.0,
           b = 1.0,
           c = 1.0)
input_hamiltonian = parameter.vec_lattice()

a = 1.0
pi = np.pi
k = np.array(
    [[0.0, 0.0, 0.0],
     [0.0, pi/a, 0.0],
     [pi/a, pi/a, 0.0],
     [0.0, 0.0, 0.0],
     [pi/a, pi/a, pi/a]]
     )
k_p = k_path(k, 1000)

ham = multiple_hamiltonian(k_p, input_hamiltonian, hamiltonian)
eig = np.linalg.eigvals(ham)
plt.plot(eig[:,0])
plt.plot(eig[:,1])
plt.show()