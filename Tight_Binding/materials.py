# author Bayu Aditya
from Tight_Binding.tight_binding.extract_parameter import input_data
from Tight_Binding.DOS.dos import dos_orthorombic

class orthorombic(input_data, dos_orthorombic):
    def __init__(self, parameter_TB, atomic_position, orbital_index, a, b, c, max_X=100, max_Y=100, max_Z=100):
        input_data.__init__(self, parameter_TB, atomic_position, orbital_index, a, b, c, max_X, max_Y, max_Z)
        dos_orthorombic.__init__(self)
