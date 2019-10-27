# Author : Bayu Aditya
import os
import json
import numpy as np
import argparse
from mpi4py import MPI

# # DEBUG MODE
# parser = argparse.ArgumentParser()
# parser.add_argument("input", help="input file JSON", type=str)
# args = parser.parse_args()


class mpi_information:
    def __init__(self, communication):
        self.comm = communication
        self.size = communication.Get_size()
        self.rank = communication.Get_rank()
    
    def summary_mpi(self):
        print("PARALLELIZATION INFO")
        print("    Parallel version (MPI4PY) running on {} processors\n".format(self.size))


class jsonio:
    def __init__(self, input_json):
        self._data = self.read_json(input_json)
        self._input_json = input_json

    @property
    def hr_file(self):
        return self._data["system"]["hr_file"]

    @property
    def atomic_position(self):
        return self._data["system"]["atomic_position"]

    @property
    def orbital_index(self):
        return self._data["system"]["orbital_index"]

    @property
    def mode_bands(self):
        return self._data["generate_hamiltonian"]["bands"]

    @property
    def mode_dos(self):
        return self._data["generate_hamiltonian"]["dos"]

    @property
    def a(self):
        return self._data["cell_parameter"]["a"]

    @property
    def b(self):
        return self._data["cell_parameter"]["b"]

    @property
    def c(self):
        return self._data["cell_parameter"]["c"]

    @property
    def max_cell(self):
        return self._data["cell_parameter"]["max_cell"]

    @property
    def kpoint(self):
        return self._data["k-points"]["nk"]

    @property
    def num_k(self):
        return self.kpoint[0] * self.kpoint[1] * self.kpoint[2]

    @property
    def start(self):
        return self._data["DOS"]["start"]
        
    @property
    def stop(self):
        return self._data["DOS"]["stop"]

    @property
    def num_freq(self):
        return self._data["DOS"]["num"]
        
    @property
    def zero_plus(self):
        return self._data["DOS"]["zero_plus"]

    @property
    def step(self):
        return self._data["DOS"]["step"]

    @property
    def kpath(self):
        kpath = np.array(self._data["BANDS"]["kpath"])
        kpath[:,0] *= 2.0*np.pi/self.a
        kpath[:,1] *= 2.0*np.pi/self.b
        kpath[:,2] *= 2.0*np.pi/self.c
        return kpath

    @property
    def num_kpath(self):
        return self._data["BANDS"]["num"]

    @property
    def outdir(self):
        return self._data["output"]["loc"]

    @property
    def hamiltonian_dos(self):
        return os.path.join(self.outdir, "hamiltonian_dos.npm")

    @property
    def hamiltonian_bands(self):
        return os.path.join(self.outdir, "hamiltonian_bands.npm")

    @property
    def num_orbitals(self):
        return self._data["num_orbitals"]

    def summary_IO(self):
        print("INPUT CONFIGURATION")
        print("    input file JSON          : {}".format(self._input_json))
        print("    hr file                  : {}".format(self.hr_file))
        print("    atomic position          : {}".format(self.atomic_position))
        print("    orbital index            : {}\n".format(self.orbital_index))
        print("OUTPUT CONFIGURATION")
        print("    directory                : {}".format(self.outdir))
        print("    hamiltonian DOS          : {}".format(self.hamiltonian_dos))
        print("    hamiltonian bands        : {}\n".format(self.hamiltonian_bands))

    @staticmethod
    def read_json(input_json):
        with open(input_json, "r") as read:
            data = json.load(read)
        return data


class variable(jsonio, mpi_information):
    def __init__(self, communication, input_json):
        mpi_information.__init__(self, communication)
        jsonio.__init__(self, input_json)

    def generate_json(self):
        assert type(self._data["num_orbitals"]) != type(None), "num_orbitals is None"
        loc_json_output = os.path.join(self.outdir, "information.json")
        with open(loc_json_output, 'w') as json_file:
            json.dump(self._data, json_file, indent=4)

    def summary(self):
        self.summary_mpi()
        self.summary_IO()


# if __name__ == "__main__":
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()

#     var = mpi_information(comm)
#     var.summary_mpi()

#     var = jsonio(args.input)
#     var.summary_IO()

#     var = variable(comm, args.input)
#     var.generate_json()