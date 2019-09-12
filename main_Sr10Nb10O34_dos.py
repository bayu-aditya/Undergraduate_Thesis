# author Bayu Aditya
from Tight_Binding.materials import orthorombic

sr10nb10o34 = orthorombic(
    parameter_TB = "Data/Sr10Nb10O34/hr_files/Sr10_Nb10_O34_10x2x16_hr.dat",
    atomic_position = "Data/Sr10Nb10O34/atomic_position.csv",
    orbital_index = "Data/Sr10Nb10O34/orbital_index.csv",
    a = 10.86909,
    b = 5.763864449563357*10.86909,
    c = 0.703931807842342*10.86909,
    max_X=3, max_Y=1, max_Z=3
)

# Create Hamiltonian for Density of states
# sr10nb10o34.dos_generate_hamiltonian(
#     n1=23, n2=3, n3=17,
#     filename_output = "hamiltonian_dos_10x2x16_23x3x17.npy"
#     )

# sr10nb10o34.plot_dos(0.0, 20.0, 10, "Data/Sr10Nb10O34/hamiltonian/hamiltonian_dos.npy")
# sr10nb10o34.plot_dos_gpu(-50.0, 25.0, 1000, "Data/Sr10Nb10O34/hamiltonian/hamiltonian_dos.npy")
sr10nb10o34.dos_generate_greenfunc_tf1(-50.0, 25.0, 10000, "Data/Sr10Nb10O34/hamiltonian/hamiltonian_dos_10x2x16_23x3x17.npy", step=2500)

# ---------------- Reduce until 133 orbital ---------------------- 
# sr10nb10o34.reduce_dos_band(
#     3, 13.5, 0.0, 
#     [29, 56, 83],
#     "Data/Sr10Nb10O34/DOS/DOS_array_5x1x8_23x3x17.npz",
#     "Data/Sr10Nb10O34/hamiltonian/hamiltonian_band_5x1x8.npy")
