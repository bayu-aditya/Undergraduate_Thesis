# author : Bayu Aditya
from Tight_Binding.tight_binding.extract_parameter import _concatenate_atom_orbital

orbitalA, orbitalB = _concatenate_atom_orbital(
    "Data/SrNbO3/atomic_position.csv", 
    "Data/SrNbO3/orbital_index.csv", 
    a = 4.089,
    b = 4.089,
    c = 4.089)

print("Selesai")