# Author : Bayu Aditya
import pandas as pd

def concatenate_atom_orbital(atomic_position, orbital_index, a, b, c):
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