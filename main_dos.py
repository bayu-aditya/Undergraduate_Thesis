# author : Bayu Aditya
from Tight_Binding.tight_binding.extract_parameter import input_data
from Tight_Binding.tight_binding.hamiltonian import multiple_hamiltonian
from Tight_Binding.tight_binding.hamiltonian import hamiltonian

from Tight_Binding.tight_binding.percobaan_dos import density_of_states
# from Tight_Binding.DOS.dos import density_of_states

loc_parameter = "Data/example/sample_parameter.dat"
loc_atom_position = "Data/example/atomic_position.csv"
loc_orbital_index = "Data/example/orbital_index.csv"

parameter = input_data(loc_parameter, loc_atom_position, loc_orbital_index, 
           a = 1.0,
           b = 1.0,
           c = 1.0)
input_hamiltonian = parameter.vec_lattice()


cls_dos = density_of_states(30, input_hamiltonian)
cls_dos.get_dos(0, 2)

# cls_dos = density_of_states(10)
# cls_dos.get_dos(0, 2)
print("Selesai")