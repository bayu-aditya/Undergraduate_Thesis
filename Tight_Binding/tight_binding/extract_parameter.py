# author : Bayu Aditya (2019)
#%%
import numpy as np
import pandas as pd

class input_data():
    def __init__(self, parameter_TB, atomic_position, orbital_index, a, b, c):
        """Input data parameter Tight Binding dari Wannier90, posisi atom, index orbital, dan ukuran lattice parameter.
        
        Arguments:
            parameter_TB {str} -- nama dan lokasi file hr
            atomic_position {str} -- lokasi file atomic_position
            orbital_index {str} -- lokasi file orbital_index
            a {numpy.float64} -- lattice parameter arah X
            b {numpy.float64} -- lattice parameter arah Y
            c {numpy.float64} -- lattice parameter arah Z
        """
        self.a = a
        self.b = b
        self.c = c
        # Import parameter_TB from wannier90
        self.parameter = extract_parameter(parameter_TB).get_data()
        # menyatukan dataframe atomic_position dan orbital_index
        self.orbitalA, self.orbitalB = _concatenate_atom_orbital(
            atomic_position, orbital_index, a, b, c)

    def get_parameter_TB(self):
        return self.parameter

    def show_parameter_TB(self, n=5):
        return self.parameter.head(n)
    
    def get_orbital_position(self):
        return self.orbitalA

    def show_orbital_position(self, n=5):
        return self.orbitalA.head(n)

    def merge(self):
        # merge parameter and orbital
        merge_df = self.parameter.merge(self.orbitalA)
        merge_df = merge_df.merge(self.orbitalB)
        return merge_df

    def vec_lattice(self):
        """Dataframe parameter tight binding beserta vector lattice untuk hamiltonian tight binding.

        | X   | Y   | Z   | A   | B   | Re  | Im  | Rx  | Ry  | Rz  |
        +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        | 0   | 0   | 0   | 1   | 1   | 0.5 | 0.5 | 0.4 | 0.3 | 0.2 | 
        | 0   | 0   | 0   | 1   | 1   | 0.5 | 0.5 | 0.4 | 0.3 | 0.2 |
        | :   | :   | :   | :   | :   | :   | :   | :   | :   | :   |
        
        Returns:
            pandas.DataFrame -- parameter tight binding dan vector lattice
        """
        # Mendapatkan vektor lattice
        merge_df = self.merge()
        merge_df["Rx"] = (merge_df["Bx"]-merge_df["Ax"]) + self.a*merge_df["X"]
        merge_df["Ry"] = (merge_df["By"]-merge_df["Ay"]) + self.b*merge_df["Y"]
        merge_df["Rz"] = (merge_df["Bz"]-merge_df["Az"]) + self.c*merge_df["Z"]
        merge_df = merge_df.drop(["Ax", "Ay", "Az", "Bx", "By", "Bz"], axis=1)
        return merge_df


class extract_parameter():
    """
    Ekstrak Parameter dari file output Wannier90 *_hr.dat . Data parameter ditampilkan dalam bentuk DataFrame.

    Args:
        hr_files (str): nama dan lokasi file hr
    """
    def __init__(self, hr_files):
        num_init_row = self._check_initial_row(hr_files)
        self.data = pd.read_csv(
            hr_files, 
            names=["X", "Y", "Z", "A", "B", "Re", "Im"], 
            skiprows=num_init_row,
            delim_whitespace=True
            )


    def get_data(self):
        """
        Mengambil parameter berbentuk DataFrame

        Returns:
            pandas.DataFrame: Parameter Tight Binding
        """
        return self.data


    def _check_initial_row(self, hr_files):
        """
        Menentukan letak baris awal pembacaan parameter

        Args:
            hr_files (str): nama dan lokasi file hr
        
        Returns:
            int: nomor baris awal pembacaan parameter
        """
        with open(hr_files) as f:
            data = f.readlines()
        for num_row, row in enumerate(data):
            if (len(row.split()) == 7):
                break            
        return num_row
    

def _concatenate_atom_orbital(atomic_position, orbital_index, a, b, c):
    """Menggabungkan dataframe dari file "atomic_position" dan "orbital_index".
    
    Example:
        INPUT
        a = 1.0
        b = 1.0
        c = 1.0
        atomic_position
            | atom | posX | posY | posZ |
            +------+------+------+------+
            | Sra  |  0.0 |  0.0 |  0.0 |
            | Nb   |  0.5 |  0.5 |  0.5 |
        orbital_index
            | atom | n | orbital | orbital_index |
            +------+---+---------+---------------+
            | Sra  | 4 |    s    |      1        |
            | Sra  | 3 |    px   |      2        |
            | Sra  | 3 |    py   |      3        |
            | Sra  | 3 |    pz   |      4        |
            | Nb   | 4 |    s    |      5        |
        OUTPUT
        OrbitalA
            |A  | Ax |  Ay  |   Az  |
            +---+----+------+-------+
            |1	|0.0 |	0.0 |	0.0 |
            |2	|0.0 |	0.0 |	0.0 |
            |3	|0.0 |	0.0 |	0.0 |
            |4	|0.0 |	0.0 |	0.0 |
            |5	|0.5 |	0.5 |	0.5 |
        OrbitalB
            |B  | Bx |  By  |   Bz  |
            +---+----+------+-------+
            |1	|0.0 |	0.0 |	0.0 |
            |2	|0.0 |	0.0 |	0.0 |
            |3	|0.0 |	0.0 |	0.0 |
            |4	|0.0 |	0.0 |	0.0 |
            |5	|0.5 |	0.5 |	0.5 |
    
    Arguments:
        atomic_position {str} -- lokasi file atomic_position
        orbital_index {str} -- lokasi file orbital_index
        a {numpy.float64} -- lattice parameter arah X
        b {numpy.float64} -- lattice parameter arah Y
        c {numpy.float64} -- lattice parameter arah Z
    
    Returns:
        tuple -- tuple ukuran [2,] dengan tuple[0] dan tuple[1] merupakan pandas.DataFrame
    """
    # read dataframe
    atom = pd.read_csv(atomic_position)     # position atoms
    orbital = pd.read_csv(orbital_index)    # orbital indexs
    # merge dataframe
    atom_orbital = orbital.merge(atom)
    # only choose certain columns
    atom_orbital = atom_orbital[["orbital_index", "posX", "posY", "posZ"]]
    # times positions with lattice parameter
    atom_orbital["posX"] = atom_orbital["posX"]*a
    atom_orbital["posY"] = atom_orbital["posY"]*b
    atom_orbital["posZ"] = atom_orbital["posZ"]*c
    # create new dataframe and rename columns
    orbitalA, orbitalB = atom_orbital.copy(), atom_orbital.copy()
    orbitalA.columns = ["A", "Ax", "Ay", "Az"]
    orbitalB.columns = ["B", "Bx", "By", "Bz"]
    return orbitalA, orbitalB